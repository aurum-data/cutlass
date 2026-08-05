from __future__ import annotations

import math
import os
import warnings
from multiprocessing import Pool, shared_memory
from time import perf_counter
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._backend import (
    cuda_supports,
    estimate_cuda_bytes,
    estimate_work,
    normalise_backend,
    normalise_device,
    normalise_dtype,
)
from ._math import _binary_log_loss_from_logits, _sigmoid
from ._solvers import _CDLogistic, _FISTALogistic, _RidgeLogistic
from .acceleration import FitProgress
from .exceptions import (
    BackendConfigurationError,
    BackendExecutionError,
    BackendUnavailableError,
    CutlassBackendWarning,
    FitCancelledError,
)
from .metrics import calculate_youden_j


def _check_cancel(cancel_callback: Optional[Callable[[], bool]]) -> None:
    if cancel_callback is not None and bool(cancel_callback()):
        raise FitCancelledError("CUTLASS fit was cancelled.")


def _emit_progress(
    callback: Optional[Callable[[dict[str, Any]], None]],
    *,
    phase: str,
    completed: int,
    total: int,
    backend: str,
    started_at: float,
    fold: Optional[int] = None,
    c_index: Optional[int] = None,
    message: str = "",
) -> None:
    if callback is None:
        return
    callback(
        FitProgress(
            phase=phase,
            completed=int(completed),
            total=int(total),
            backend=backend,
            fold=fold,
            c_index=c_index,
            message=message,
            elapsed_seconds=max(0.0, perf_counter() - started_at),
        ).to_dict()
    )


def _fold_path_worker_shm(args):
    """
    Run ONE fold across the ENTIRE C-path, warm-starting within the worker.

    Returns
    -------
    val_losses : 1D np.ndarray of length len(Cs_path) with validation log-loss
                 for this fold at each C in the provided order.
    """
    (shm_name_X, shape_X, dtype_X, order_X,
     shm_name_y, shape_y, dtype_y,
     val_idx, Cs_path, solver_name, tol, max_iter, all_pm1) = args

    # Attach to shared arrays (zero-copy)
    shmX = shared_memory.SharedMemory(name=shm_name_X)
    X = np.ndarray(shape_X, dtype=np.dtype(dtype_X), buffer=shmX.buf, order=order_X)
    shmy = shared_memory.SharedMemory(name=shm_name_y)
    y = np.ndarray(shape_y, dtype=np.dtype(dtype_y), buffer=shmy.buf)

    try:
        n, p = X.shape
        # Build train/validation views for this fold (one time)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        Xtr, ytr = X[mask, :], y[mask]
        Xva, yva = X[val_idx, :], y[val_idx]
        ntr = Xtr.shape[0]

        # Warm-start state kept inside the worker across the C-path
        w_ws = np.zeros(p, dtype=np.float64)
        b_ws = None
        active_ws = np.array([], dtype=np.int32)

        val_losses = np.empty(len(Cs_path), dtype=np.float64)
        prev_C = None

        for t, Ci in enumerate(Cs_path):
            lam_ci   = 1.0 / (float(Ci) * float(ntr))
            lam_prev = (1.0 / (float(prev_C) * float(ntr))) if prev_C is not None else None

            if solver_name in ("saga", "liblinear", "cd"):
                solver = _CDLogistic(lam=lam_ci, tol=tol, max_iter=max_iter,
                                     verbose=False, kkt_tol=1e-4, all_pm1=all_pm1)
                solver.fit(Xtr, ytr, w0=w_ws, b0=b_ws, lam_prev=lam_prev, active_init=active_ws)
            else:
                solver = _FISTALogistic(lam=lam_ci, tol=tol, max_iter=max(max_iter, 4000), verbose=False)
                solver.fit(Xtr, ytr, w0=w_ws, b0=b_ws)

            # Validation loss at this C
            z_val = Xva @ solver.w_ + solver.b_
            val_losses[t] = _binary_log_loss_from_logits(yva.astype(float), z_val)

            # Warm-start for the next C
            w_ws = solver.w_
            b_ws = solver.b_
            active_ws = np.where(np.abs(w_ws) > 0)[0].astype(np.int32)
            prev_C = Ci

        return val_losses
    finally:
        # Detach from shared memory (the parent unlinks)
        shmX.close()
        shmy.close()


def _fit_adaptive_pilot(
    X: np.ndarray,
    y: np.ndarray,
    *,
    pilot_C: float,
    eps: float,
    tol: float,
    max_iter: int,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if pilot_C <= 0.0:
        raise ValueError("adaptive_pilot_C must be positive.")
    if eps <= 0.0:
        raise ValueError("adaptive_eps must be positive.")

    n = X.shape[0]
    lam = 1.0 / (float(pilot_C) * max(float(n), 1.0))
    pilot = _RidgeLogistic(lam=lam, tol=tol, max_iter=max_iter, verbose=False)
    pilot.fit(X, y)
    feature_scales = np.maximum(np.abs(pilot.w_) + float(eps), float(eps))
    return feature_scales, pilot


def _adaptive_fold_path_worker_shm(args):
    """
    Run one adaptive-L1 fold across the full C path.

    Each fold gets its own L2 pilot fit, then the L1 path is fit on
    X * (abs(beta_pilot) + eps). The returned validation losses are computed
    on the same fold-specific weighted representation.
    """
    (shm_name_X, shape_X, dtype_X, order_X,
     shm_name_y, shape_y, dtype_y,
     val_idx, Cs_path, solver_name, tol, max_iter,
     adaptive_eps, adaptive_pilot_C) = args

    shmX = shared_memory.SharedMemory(name=shm_name_X)
    X = np.ndarray(shape_X, dtype=np.dtype(dtype_X), buffer=shmX.buf, order=order_X)
    shmy = shared_memory.SharedMemory(name=shm_name_y)
    y = np.ndarray(shape_y, dtype=np.dtype(dtype_y), buffer=shmy.buf)

    try:
        n, p = X.shape
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        Xtr, ytr = X[mask, :], y[mask]
        Xva, yva = X[val_idx, :], y[val_idx]
        ntr = Xtr.shape[0]

        feature_scales, _ = _fit_adaptive_pilot(
            Xtr,
            ytr,
            pilot_C=float(adaptive_pilot_C),
            eps=float(adaptive_eps),
            tol=float(tol),
            max_iter=int(max_iter),
        )
        Xtr_weighted = np.asarray(Xtr * feature_scales[None, :], dtype=np.float64)
        Xva_weighted = np.asarray(Xva * feature_scales[None, :], dtype=np.float64)

        w_ws = np.zeros(p, dtype=np.float64)
        b_ws = None
        active_ws = np.array([], dtype=np.int32)

        val_losses = np.empty(len(Cs_path), dtype=np.float64)
        prev_C = None

        for t, Ci in enumerate(Cs_path):
            lam_ci = 1.0 / (float(Ci) * float(ntr))
            lam_prev = (1.0 / (float(prev_C) * float(ntr))) if prev_C is not None else None

            if solver_name in ("saga", "liblinear", "cd"):
                solver = _CDLogistic(lam=lam_ci, tol=tol, max_iter=max_iter,
                                     verbose=False, kkt_tol=1e-4, all_pm1=False)
                solver.fit(
                    Xtr_weighted,
                    ytr,
                    w0=w_ws,
                    b0=b_ws,
                    lam_prev=lam_prev,
                    active_init=active_ws,
                )
            else:
                solver = _FISTALogistic(lam=lam_ci, tol=tol, max_iter=max(max_iter, 4000), verbose=False)
                solver.fit(Xtr_weighted, ytr, w0=w_ws, b0=b_ws)

            z_val = Xva_weighted @ solver.w_ + solver.b_
            val_losses[t] = _binary_log_loss_from_logits(yva.astype(float), z_val)

            w_ws = solver.w_
            b_ws = solver.b_
            active_ws = np.where(np.abs(w_ws) > 0)[0].astype(np.int32)
            prev_C = Ci

        return val_losses
    finally:
        shmX.close()
        shmy.close()


class CutlassLogisticCV:
    """
    Minimal drop-in to replace sklearn.linear_model.CutlassLogisticCV (binary, L1).
    Adaptive L1 is available as an optional mode via ``penalty="adaptive_l1"``.

    Supported args:
      - Cs (int or array-like): if int, uses np.logspace(-4, 4, Cs)
      - penalty='l1' (default) or 'adaptive_l1'
      - solver: 'cd' (recommended), 'fista', or 'hybrid';
        'saga'/'liblinear' map to CPU coordinate descent
      - scoring='neg_log_loss' (only)
      - cv (int folds), tol, max_iter, random_state
      - refit (bool)
      - cv_rule: 'min' (best mean) or '1se' (strongest within one SE of best)
      - zero_clamp: set |w|<=threshold to 0 after final fit (cleanup only)
      - adaptive_eps: stabilizer for adaptive-L1 weights, abs(beta_pilot) + eps
      - adaptive_pilot_C: C value for the L2 pilot used by adaptive L1
      - backend: 'cpu' (default), 'cuda', or 'auto'
      - device: optional zero-based CUDA device id
      - dtype: 'float64' (the parity-qualified precision)
      - allow_cpu_fallback: restart unsupported/failed CUDA fits on CPU

    ``fista`` can run its CV path and final refit on CUDA. ``hybrid`` can run
    FISTA CV on CUDA and performs its final coordinate-descent refit on CPU.
    Logical polishing and public prediction remain CPU operations.

    Attributes include coef_, intercept_, C_, Cs_, classes_, backend_report_,
    fit_timings_, and backend selection/fallback diagnostics.
    
    Includes optional logical post-processing ("logical polish") after the final refit:
      - logic_polish: bool, off by default
      - logic_scale: float, magnitude K for active coefficients (default 10.0)
      - logic_target: float or None, early-exit if Youden's J >= this (e.g., 0.97)
      - logic_maxk: int or None, cap the number of top features to test (speed)
      - logic_rel_tol: float, Adoption tolerance (relative). Adopt logical model if
          J_logical >= J_baseline * (1 - logic_rel_tol). Example: 0.01 allows up to ~1% lower J.
      - logic_plot   : bool, Produce a J vs k plot, and optionally J vs K if a grid is given.
      - logic_plot_dir: Optional[str], If provided and writable, figures are saved there (PNG).
          Otherwise, figures are returned on self.logic_figs_ for the caller.
      - logic_Ks_plot: Optional[Sequence[float]], If provided, also generate 
          J vs K using these K values.
      - logic_k_policy : {'global','premin'}, how to choose k after scanning
            'global'  -> best J over all k (previous behavior)
            'premin'  -> best J for k in [1, k_first_min], where k_first_min is
                         the first \u201creal\u201d valley of the smoothed J_k curve
      - logic_smooth_k : int, odd window size for moving-average smoothing of J_k
      - logic_firstmin_drop : float, declare a valley once the smoothed J has
                         fallen by at least this absolute amount from its
                         running maximum (robust valley detection)
      - logic_firstmin_frac : float in (0,1], fallback: if no valley is found,
                         treat the first \u2308frac\xb7L\u2309 points as the search window

    """
    # -------------------------------------------------------------------------
    # Theory tie-in:
    #   \u2022 CV across C (equivalently \u03bb = 1/(Cn)) controls sparsity. On rectified
    #     designs, lower cross-correlation helps avoid false inclusions (IC).
    #   \u2022 "1se" rule prefers simpler models within one SE of best loss (paper\u2019s
    #     preference for sparse, interpretable solutions).
    #   \u2022 logic_polish compresses the L1 model into a rule: fixed-magnitude \xb1K
    #     on top\u2011k features + an intercept policy (e.g., m\u2011of\u2011k). This reflects
    #     the Boolean framing discussed in the manuscript.
    # -------------------------------------------------------------------------
    def __init__(self, Cs=10, penalty="l1", solver="cd", scoring="neg_log_loss",
                 cv=3, n_jobs=-1, tol=1e-4, max_iter=2000, refit=True,
                 random_state=42, verbose=True,
                 # existing custom knobs
                 cv_rule="min", zero_clamp=0.0,
                 # logical-polish knobs
                 logic_polish=False,
                 logic_scale=10.0,
                 logic_target=None,
                 logic_maxk=None,
                 logic_rel_tol=0.01,         # 1% tolerance default
                 logic_plot=False,
                 logic_plot_dir=None,
                 logic_Ks_plot=None,
                 # k-selection behavior
                 logic_k_policy="",          # "" or "global" or "premin"
                 logic_smooth_k=3,
                 logic_firstmin_drop=0.05,
                 logic_firstmin_frac=0.5,
                 # NEW: intercept policy
                 logic_intercept="mean",     # "mean" | "mofk" | "maxj"
                 logic_m=None,               # for "mofk": integer m (default: m=k)
                 logic_m_frac=None,          # alternatively, fraction in (0,1]; m=ceil(frac*k)
                 # adaptive-L1 knobs
                 adaptive_eps=1e-3,
                 adaptive_pilot_C=1.0,
                 # execution backend
                 backend="cpu",
                 device=None,
                 dtype="float64",
                 allow_cpu_fallback=True):
        self.Cs = Cs
        self.penalty = penalty
        self.solver = solver
        self.scoring = scoring
        self.cv = int(cv)
        self.n_jobs = n_jobs
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.refit = bool(refit)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
    
        # extras
        self.cv_rule = str(cv_rule)
        self.zero_clamp = float(zero_clamp)
    
        # logical-polish parameters
        self.logic_polish   = bool(logic_polish)
        self.logic_scale    = float(logic_scale)
        self.logic_target   = None if logic_target is None else float(logic_target)
        self.logic_maxk     = None if logic_maxk is None else int(logic_maxk)
        self.logic_rel_tol  = float(logic_rel_tol)
        self.logic_plot     = bool(logic_plot)
        self.logic_plot_dir = logic_plot_dir
        self.logic_Ks_plot  = None if logic_Ks_plot is None else list(map(float, logic_Ks_plot))
    
        # k-selection
        self.logic_k_policy      = (str(logic_k_policy) or "global").lower()
        self.logic_smooth_k      = int(logic_smooth_k)
        self.logic_firstmin_drop = float(logic_firstmin_drop)
        self.logic_firstmin_frac = float(logic_firstmin_frac)
    
        # NEW: intercept policy
        self.logic_intercept = str(logic_intercept).lower()  # "mean" | "mofk" | "maxj"
        self.logic_m         = None if logic_m is None else int(logic_m)
        self.logic_m_frac    = None if logic_m_frac is None else float(logic_m_frac)
        self.adaptive_eps = float(adaptive_eps)
        self.adaptive_pilot_C = float(adaptive_pilot_C)
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
    
        # learned attributes
        self.classes_   = np.array([0, 1], dtype=int)
        self.Cs_        = None
        self.C_         = None
        self.coef_      = None
        self.intercept_ = None
    
        # diagnostics from the logical step (populated if run)
        self.logic_diag_  = {}
        self.logic_figs_  = []
        self.adaptive_feature_scales_ = None
        self.adaptive_penalty_weights_ = None
        self.adaptive_pilot_coef_ = None
        self.adaptive_pilot_intercept_ = None
        self.adaptive_weighted_coef_ = None
        self.cv_fold_losses_ = None
        self.cv_mean_losses_ = None
        self.cv_se_losses_ = None
        self.cv_path_Cs_ = None

        # Populated after every successful fit.
        self.backend_requested_ = None
        self.backend_used_ = None
        self.backend_provider_ = None
        self.device_id_ = None
        self.device_name_ = None
        self.dtype_ = None
        self.n_jobs_effective_ = None
        self.fallback_reason_ = None
        self.auto_decision_ = None
        self.fit_timings_ = {}
        self.peak_device_memory_bytes_ = None
        self.runtime_versions_ = {}
        self.backend_report_ = {}


    def _kfold_indices(self, n):
        k = self.cv
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        return np.array_split(idx, k)

    def _make_solver(self, lam, X):
        all_pm1 = np.all((X == 1) | (X == -1))
        s = self.solver.lower()
        if s in ("saga", "liblinear", "cd"):
            return _CDLogistic(lam=lam, tol=self.tol, max_iter=self.max_iter,
                               verbose=self.verbose, kkt_tol=1e-4, all_pm1=all_pm1)
        elif s == "fista":
            return _FISTALogistic(lam=lam, tol=self.tol, max_iter=max(self.max_iter, 4000),
                                  verbose=self.verbose)
        else:
            raise ValueError(f"Unsupported solver '{self.solver}'. Use 'cd' or 'fista'.")

    def _logical_polish(self, X, y, w, b,
                        K=10.0, target=None, rel_tol=0.01, maxk=None,
                        make_plots=False, Ks_plot=None, plot_dir=None,
                        verbose=True):
        """
        Compress final L1 model into a Boolean-style rule on the top-k features.
        NEW: Intercept policy via self.logic_intercept:
            - "mean": mean-matching (existing behavior)
            - "mofk": put 0.5 boundary at m-of-k votes (default m=k -> AND)
            - "maxj": choose b that maximizes Youden's J at threshold 0.5
        Also fixes k off-by-one and restores adoption guard.
        """
        import matplotlib.pyplot as plt
        # ---------------------------------------------------------------------
        # Theory tie-in:
        #  After L1 finds a sparse linear model, we can "quantize" it into a rule:
        #    sign(w_j) \u2192 vote direction on feature j (top-|w| features only),
        #    magnitude \u2192 constant K,
        #    decision threshold controlled by intercept policy (mean/m-of-k/maxJ).
        #  This mirrors the paper\u2019s Boolean framing: sparse interpretable logic
        #  consistent with threshold-based phenomena (Section 7, discussion).
        # ---------------------------------------------------------------------
    
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        y_bool = (y >= 1)
    
        # --- Baseline J from the original L1 logits (threshold at 0.5 => z>=0) ---
        z0 = X @ w + b
        yhat0 = (z0 >= 0.0)
        J0 = calculate_youden_j(y_bool, yhat0)
    
        # Order features by |w| (drop exact zeros)
        order_all = np.argsort(-np.abs(w))
        mask_nz = np.abs(w[order_all]) > 0
        if not np.any(mask_nz):
            return w, b, float(J0), False, [], {"J0": J0, "k_best": 0}
    
        order = order_all[mask_nz]
        L = len(order)
        if (maxk is not None) and (maxk > 0):
            L = min(L, int(maxk))
    
        # Precompute signed columns and cumulative sums for fast top\u2011k votes
        signs = np.sign(w[order][:L])
        X_sel = X[:, order[:L]] * signs       # (n, L) -> aligned votes are +1
        cum = np.cumsum(X_sel, axis=1)        # (n, L), vote sum s in {-k,-k+2,...,k}
    
        # ---------- Intercept helpers ----------
        ybar = float(np.mean(y_bool))
    
        def solve_intercept_mean_match(t, b_init=0.0):
            # Newton + bracket to solve mean(sigmoid(t + b)) = ybar
            b_cur = float(b_init)
            for _ in range(8):
                z = t + b_cur
                p_hat = _sigmoid(z)
                g = float(np.mean(p_hat - y_bool))
                h = float(np.mean(p_hat * (1.0 - p_hat)))
                if h < 1e-12:
                    break
                step = g / h
                b_cur -= step
                if abs(step) < 1e-8:
                    return b_cur
            # robust bracket (\xb140 is fine; sigmoid is already clipped there)
            lo, hi = -40.0, 40.0
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                mval = float(np.mean(_sigmoid(t + mid)) - ybar)
                if mval > 0.0: hi = mid
                else:          lo = mid
            return 0.5 * (lo + hi)
    
        def b_for_mofk(k, m=None):
            """
            Intercept for an 'at-least m-of-k' rule on \xb11 votes with weight magnitude K.
            Place the 0.5 boundary halfway between the m and (m-1) vote levels so that
            examples with exactly m aligned votes have a *positive margin* of +K.
              s_m   = 2m - k
              s_m1  = 2(m-1) - k
              s_thr = (s_m + s_m1)/2 = 2m - k - 1
              b     = -K * s_thr
            """
            if m is None:
                m = k  # strict AND by default
            s_thr = 2*int(m) - k - 1
            return -K * s_thr  # no epsilon needed; s_m yields z = +K
    
        def b_for_max_J(t, y_bool):
            # Scan thresholds on t to maximize Youden's J at 0.5
            vals = np.unique(np.sort(t))
            if vals.size == 0:
                return 0.0, calculate_youden_j(y_bool, np.zeros_like(y_bool, dtype=bool))
            mids = 0.5 * (vals[:-1] + vals[1:])
            thr_candidates = np.r_[vals[0] - 1e-9, mids, vals[-1] + 1e-9]
            bestJ, best_thr = -np.inf, float(thr_candidates[0])
            for thr in thr_candidates:
                pred = (t >= thr)          # equivalent to z = t + b >= 0 with b = -thr
                J = calculate_youden_j(y_bool, pred)
                if J > bestJ:
                    bestJ, best_thr = J, float(thr)
            return -best_thr, bestJ
    
        def youden_from_logit(t, b_local):
            pred = (t + b_local >= 0.0)
            return calculate_youden_j(y_bool, pred)
    
        # ----- Scan k = 1..L, record J_k and b_k under the chosen intercept policy -----
        mode = getattr(self, "logic_intercept", "mean")
        m_fixed = getattr(self, "logic_m", None)
        m_frac  = getattr(self, "logic_m_frac", None)
    
        J_k = np.empty(L, dtype=np.float64)
        b_k = np.empty(L, dtype=np.float64)
    
        best_J_overall = float(J0)
        best_k_overall = 0
        best_b_overall = float(b)
    
        for k in range(1, L + 1):
            t_k = K * cum[:, k - 1]
    
            if mode == "mofk":
                if (m_fixed is not None) and (m_fixed >= 1):
                    m_here = int(np.clip(m_fixed, 1, k))
                elif (m_frac is not None) and (0.0 < m_frac <= 1.0):
                    m_here = int(np.ceil(m_frac * k))
                else:
                    m_here = k
                b_here = b_for_mofk(k, m_here)
                J_here = youden_from_logit(t_k, b_here)

            elif mode == "maxj":
                b_here, J_here = b_for_max_J(t_k, y_bool)
    
            else:  # "mean"
                b_init = best_b_overall if best_k_overall > 0 else b
                b_here = solve_intercept_mean_match(t_k, b_init=b_init)
                J_here = youden_from_logit(t_k, b_here)
    
            J_k[k - 1] = J_here
            b_k[k - 1] = b_here
    
            if J_here > best_J_overall:
                best_J_overall, best_k_overall, best_b_overall = J_here, k, b_here
    
            if (target is not None) and (J_here >= float(target)):
                best_J_overall, best_k_overall, best_b_overall = J_here, k, b_here
                break
    
        # ----- Choose k by policy -----
        def _smooth(y, win):
            win = max(3, int(win))
            if win % 2 == 0: win += 1
            if len(y) < win: return y.copy()
            ker = np.ones(win, dtype=np.float64) / win
            ys = np.convolve(y, ker, mode='same')
            half = win // 2
            ys[:half] = y[:half]; ys[-half:] = y[-half:]
            return ys
    
        Js = _smooth(J_k, self.logic_smooth_k)
        # First valley of *smoothed* curve
        runmax = np.maximum.accumulate(Js)
        drop = runmax - Js
        k_first_min = None
        for i in range(1, L - 1):
            if drop[i] >= self.logic_firstmin_drop and Js[i] <= Js[i-1] and Js[i] <= Js[i+1]:
                k_first_min = i + 1  # 1-based
                break
        if k_first_min is None:
            k_first_min = max(1, int(np.ceil(self.logic_firstmin_frac * L)))
    
        if self.logic_k_policy == "premin":
            window_end = max(1, min(k_first_min, L))
            k_cand = int(np.argmax(J_k[:window_end])) + 1
            J_cand = float(J_k[k_cand - 1])
            b_cand = float(b_k[k_cand - 1])
            if verbose:
                print(f"[logical] premin policy: first valley at k~={k_first_min}, "
                      f"candidate k={k_cand} with J={J_cand:.4f}")
        else:  # "global"
            k_cand = int(np.argmax(J_k)) + 1
            J_cand = float(J_k[k_cand - 1])
            b_cand = float(b_k[k_cand - 1])
            if verbose:
                print(f"[logical] global policy: candidate k={k_cand} with J={J_cand:.4f}")
    
        # ----- Decide adoption (relative tolerance or hard target) -----
        adopt_threshold = (float(target) if target is not None else float(J0) * (1.0 - float(rel_tol)))
        adopted = (k_cand > 0) and (J_cand >= adopt_threshold)
    
        # Build adopted weight vector if accepted
        if adopted:
            w_new = np.zeros_like(w)
            topk_idx = order[:k_cand]                 # <-- fixed off-by-one
            w_new[topk_idx] = K * np.sign(w[topk_idx])
            b_new = float(b_cand)
            J_adopt = float(J_cand)
            if verbose:
                tau = -b_new / float(K)
                print(f"[logical] adopted rule-like model: J={J_adopt:.4f} "
                      f"| K={K:g}, k={k_cand}, 0.5 boundary at vote sum >= {tau:.2f}")
        else:
            w_new, b_new, J_adopt = w, b, float(J0)
    
        # ----- Plots & diagnostics (unchanged apart from fixed labels) -----
        figs = []
        diag = {
            "J0": float(J0),
            "J_k": J_k.copy(),
            "J_k_smooth": Js.copy(),
            "b_k": b_k.copy(),
            "order": order.copy(),
            "best_k_overall": int(best_k_overall),
            "best_J_overall": float(best_J_overall),
            "policy": self.logic_k_policy,
            "intercept_policy": mode,
            "k_first_min": int(k_first_min),
            "k_chosen": int(k_cand),
            "J_chosen": float(J_cand),
            "adopted": bool(adopted),
            "K_used": float(K),
            "adopt_threshold": float(adopt_threshold),
        }
    
        def _save_or_collect(fig, fname):
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
                out = os.path.join(plot_dir, fname)
                fig.savefig(out, dpi=150)
                if verbose:
                    print(f"[logical] saved: {out}")
                plt.close(fig)
            else:
                figs.append(fig)
    
        if make_plots:
            ks = np.arange(1, L + 1)
            fig1, ax1 = plt.subplots(figsize=(6.8, 4.2))
            ax1.plot(ks, J_k, marker='o', linewidth=1, label='J (raw)')
            ax1.plot(ks, Js, linewidth=2, alpha=0.6, label='J (smoothed)')
            ax1.axhline(J0, color='k', linestyle='--', linewidth=1.5, label=f"Baseline J0={J0:.3f}")
            if k_first_min is not None and 1 <= k_first_min <= L:
                ax1.axvline(k_first_min, color='gray', linestyle=':', linewidth=1.5, label=f"first min @ k={k_first_min}")
            if adopted:
                ax1.axvline(k_cand, color='C3', linestyle='--', linewidth=1.0, label=f"chosen k={k_cand}")
            ax1.set_xlabel("k (top-|w| features kept)")
            ax1.set_ylabel("Youden's J")
            ax1.set_title(f"Youden's J vs k (logical compression, intercept={mode})")
            ax1.legend(loc='best')
            fig1.tight_layout()
            _save_or_collect(fig1, "logical_J_vs_k.png")
    
            if Ks_plot is not None and len(Ks_plot) > 0:
                bestJ_vs_K = []
                for Ktest in Ks_plot:
                    J_best_thisK = -np.inf
                    b_best_thisK = 0.0
                    for k in range(1, L + 1):
                        t_k = Ktest * cum[:, k - 1]
                        if mode == "mofk":
                            if (self.logic_m is not None) and (self.logic_m >= 1):
                                m_here = int(np.clip(self.logic_m, 1, k))
                            elif (self.logic_m_frac is not None) and (0.0 < self.logic_m_frac <= 1.0):
                                m_here = int(np.ceil(self.logic_m_frac * k))
                            else:
                                m_here = k
                            b_here = b_for_mofk(k, m_here)
                            J_here = youden_from_logit(t_k, b_here)
                        elif mode == "maxj":
                            b_here, J_here = b_for_max_J(t_k, y_bool)
                        else:
                            b_here = solve_intercept_mean_match(t_k, b_init=b_best_thisK)
                            J_here = youden_from_logit(t_k, b_here)
                        if J_here > J_best_thisK:
                            J_best_thisK = J_here
                            b_best_thisK = b_here
                    bestJ_vs_K.append(J_best_thisK)
                bestJ_vs_K = np.array(bestJ_vs_K, dtype=np.float64)
                diag["Ks_plot"] = list(map(float, Ks_plot))
                diag["bestJ_vs_K"] = bestJ_vs_K.copy()
    
                fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
                ax2.plot(Ks_plot, bestJ_vs_K, marker='o', linewidth=1)
                ax2.axhline(J0, color='k', linestyle='--', linewidth=1.5, label=f"Baseline J0={J0:.3f}")
                ax2.set_xlabel("K (magnitude of logical weights)")
                ax2.set_ylabel("Best Youden's J across k")
                ax2.set_title(f"Youden's J vs K (k re-optimized, intercept={mode})")
                ax2.legend(loc='best')
                fig2.tight_layout()
                _save_or_collect(fig2, "logical_J_vs_K.png")
    
        return w_new, b_new, J_adopt, adopted, figs, diag



    def _fit_cpu(
        self,
        X,
        y,
        *,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        started_at: Optional[float] = None,
    ):
        """
        Cross-validated L1 logistic with **persistent** process pool:
          - one pool for the whole CV run,
          - one task per fold (each task walks the entire C-path),
          - no double-submission,
          - no process respawn per C,
          - no per-C pickling of warm-starts.
    
        Final refit + logical polish: unchanged.
        """
        import math
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import shared_memory, cpu_count

        cpu_method_started = perf_counter()
        progress_started = cpu_method_started if started_at is None else started_at
        _check_cancel(cancel_callback)
        _emit_progress(
            progress_callback,
            phase="input_validation",
            completed=0,
            total=1,
            backend="cpu",
            started_at=progress_started,
        )
    
        # Parent-side arrays (we keep X as Fortran for fast column slices in CD)
        X = np.asarray(X, dtype=np.float64, order='F')
        y = np.asarray(y, dtype=int)
        n, p = X.shape
    
        # Build C grid (same semantics)
        if isinstance(self.Cs, (int, np.integer)):
            Cs = np.logspace(-4, 4, int(self.Cs))
        else:
            Cs = np.asarray(self.Cs, dtype=np.float64)
        self.Cs_ = Cs.copy()
    
        penalty = self.penalty.lower()
        if penalty not in {"l1", "adaptive_l1"}:
            raise ValueError("Only penalty='l1' and penalty='adaptive_l1' are supported.")
        if self.scoring not in ("neg_log_loss",):
            raise ValueError("Only scoring='neg_log_loss' is supported.")
        if penalty == "adaptive_l1":
            if self.adaptive_eps <= 0.0:
                raise ValueError("adaptive_eps must be positive.")
            if self.adaptive_pilot_C <= 0.0:
                raise ValueError("adaptive_pilot_C must be positive.")

        self.adaptive_feature_scales_ = None
        self.adaptive_penalty_weights_ = None
        self.adaptive_pilot_coef_ = None
        self.adaptive_pilot_intercept_ = None
        self.adaptive_weighted_coef_ = None
        self._last_cpu_phase_timings = {
            "input_validation": perf_counter() - cpu_method_started
        }
        cv_started = perf_counter()
    
        folds = self._kfold_indices(n)  # list of validation indices
        all_pm1 = np.all((X == 1) | (X == -1))
    
        # Path order: large \u03bb\u2192small \u03bb (i.e., small C\u2192large C)
        lam_full = 1.0 / (np.maximum(Cs, 1e-12) * float(n))
        order = np.argsort(lam_full)[::-1]
        Cs_path = Cs[order]
    
        # Hybrid policy: FISTA for CV (fast), CD for final refit
        use_hybrid = (self.solver.lower() == "hybrid")
        solver_name = ("fista" if use_hybrid else self.solver.lower())
    
        # Share X, y ONCE across the whole pool
        shmX = shared_memory.SharedMemory(create=True, size=X.nbytes)
        X_sh = np.ndarray(X.shape, dtype=X.dtype, buffer=shmX.buf, order='F')
        X_sh[...] = X  # one-time copy
    
        shmy = shared_memory.SharedMemory(create=True, size=y.nbytes)
        y_sh = np.ndarray(y.shape, dtype=y.dtype, buffer=shmy.buf)
        y_sh[...] = y  # one-time copy
    
        # Decide pool size
        max_workers = self.n_jobs
        if max_workers in (None, -1):
            max_workers = min(self.cv, max(1, cpu_count() - 1))
        # ^ Use at most (cv) workers to avoid idle processes; leave 1 core free.
    
        try:
            # Prepare arguments: **one task per fold**
            args_list = []
            for f, val_idx in enumerate(folds):
                if penalty == "adaptive_l1":
                    args_list.append((
                        shmX.name, X.shape, X.dtype.str, 'F',
                        shmy.name, y.shape, y.dtype.str,
                        val_idx.astype(np.int64, copy=False),
                        Cs_path, solver_name, float(self.tol), int(self.max_iter),
                        float(self.adaptive_eps), float(self.adaptive_pilot_C)
                    ))
                else:
                    args_list.append((
                        shmX.name, X.shape, X.dtype.str, 'F',
                        shmy.name, y.shape, y.dtype.str,
                        val_idx.astype(np.int64, copy=False),
                        Cs_path, solver_name, float(self.tol), int(self.max_iter), bool(all_pm1)
                    ))
    
            # Launch ONCE; each worker walks the entire path for its fold
            if self.cv == 1 or max_workers == 1:
                # trivial sequential fallback (no pool)
                worker = _adaptive_fold_path_worker_shm if penalty == "adaptive_l1" else _fold_path_worker_shm
                fold_losses = [ worker(a) for a in args_list ]
            else:
                worker = _adaptive_fold_path_worker_shm if penalty == "adaptive_l1" else _fold_path_worker_shm
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    fold_losses = list(ex.map(worker, args_list))
    
            # Aggregate CV stats across folds for each C on the path
            fold_losses = np.vstack(fold_losses)              # (cv, nC)
            mean_losses = np.mean(fold_losses, axis=0)        # (nC,)
            if self.cv > 1:
                se_losses = np.std(fold_losses, axis=0, ddof=1) / math.sqrt(self.cv)
            else:
                se_losses = np.zeros_like(mean_losses)

            self.cv_fold_losses_ = fold_losses.copy()
            self.cv_mean_losses_ = mean_losses.copy()
            self.cv_se_losses_ = se_losses.copy()
            self.cv_path_Cs_ = Cs_path.copy()
            _check_cancel(cancel_callback)
            _emit_progress(
                progress_callback,
                phase="cv_path",
                completed=int(self.cv),
                total=int(self.cv),
                backend="cpu",
                started_at=progress_started,
                message="CPU cross-validation paths completed.",
            )
    
            # Emit a compact progress log (optional)
            if self.verbose:
                for i, Ci in enumerate(Cs_path):
                    lam_dbg = 1.0 / (float(Ci) * float(n))
                    print(f"[CV|parallel] C={Ci:.4g} (lam~={lam_dbg:.3e}) -> "
                          f"mean={mean_losses[i]:.6f} +/- {se_losses[i]:.6f}")
    
            # Pick C by rule
            if self.cv_rule.lower() == "1se":
                j_best = int(np.argmin(mean_losses))
                thr = mean_losses[j_best] + se_losses[j_best]
                candidates = np.where(mean_losses <= thr)[0]
                j_pick = int(np.min(candidates)) if candidates.size else j_best
                chosen_C = float(Cs_path[j_pick])
            else:
                chosen_C = float(Cs_path[int(np.argmin(mean_losses))])
    
            self.C_ = chosen_C
            self._last_cpu_phase_timings["cv_path"] = perf_counter() - cv_started

            _check_cancel(cancel_callback)
            _emit_progress(
                progress_callback,
                phase="final_refit",
                completed=0,
                total=1,
                backend="cpu",
                started_at=progress_started,
            )
    
            # ----- Final refit on ALL data -----
            final_refit_started = perf_counter()
            lam_final = 1.0 / (float(self.C_) * float(n))
            X_refit = X
            feature_scales = None
            if penalty == "adaptive_l1":
                feature_scales, pilot = _fit_adaptive_pilot(
                    X,
                    y,
                    pilot_C=float(self.adaptive_pilot_C),
                    eps=float(self.adaptive_eps),
                    tol=float(self.tol),
                    max_iter=int(self.max_iter),
                )
                X_refit = np.asarray(X * feature_scales[None, :], dtype=np.float64, order='F')
                self.adaptive_feature_scales_ = feature_scales.copy()
                self.adaptive_penalty_weights_ = (1.0 / feature_scales).copy()
                self.adaptive_pilot_coef_ = pilot.w_.reshape(1, -1).copy()
                self.adaptive_pilot_intercept_ = np.array([pilot.b_], dtype=np.float64)

            if use_hybrid:
                # CD for the final refit to get crisp sparsity
                all_pm1_full = penalty == "l1" and np.all((X_refit == 1) | (X_refit == -1))
                solver = _CDLogistic(lam=lam_final, tol=self.tol, max_iter=self.max_iter,
                                     verbose=self.verbose, kkt_tol=1e-4, all_pm1=all_pm1_full)
                solver.fit(X_refit, y, w0=None, b0=None, lam_prev=None, active_init=None)
            else:
                solver = self._make_solver(lam_final, X_refit)
                if isinstance(solver, _FISTALogistic):
                    solver.fit(X_refit, y, w0=None, b0=None)
                else:
                    solver.fit(X_refit, y, w0=None, b0=None, lam_prev=None, active_init=None)
    
            w_final = solver.w_.copy()
            if feature_scales is not None:
                self.adaptive_weighted_coef_ = w_final.reshape(1, -1).copy()
                w_final = w_final * feature_scales
            b_final = float(solver.b_)
    
            # Optional presentation clamp
            if self.zero_clamp > 0.0:
                w_final[np.abs(w_final) <= self.zero_clamp] = 0.0
            self._last_cpu_phase_timings["final_refit"] = (
                perf_counter() - final_refit_started
            )

            # Optional logical polish (unchanged)
            logical_started = perf_counter()
            if self.logic_polish:
                w_new, b_new, j_new, adopted, figs, diag = self._logical_polish(
                    X=X, y=y, w=w_final, b=b_final,
                    K=self.logic_scale,
                    target=self.logic_target,
                    rel_tol=self.logic_rel_tol,
                    maxk=self.logic_maxk,
                    make_plots=self.logic_plot,
                    Ks_plot=self.logic_Ks_plot,
                    plot_dir=self.logic_plot_dir,
                    verbose=self.verbose
                )
                self.logic_figs_ = figs
                self.logic_diag_ = diag
                if adopted and self.verbose:
                    print(f"[logical] adopted rule-like model: J={j_new:.4f}")
                if adopted:
                    w_final, b_final = w_new, b_new
            self._last_cpu_phase_timings["logical_polish_cpu"] = (
                perf_counter() - logical_started
            )
    
            self.coef_ = w_final.reshape(1, -1)
            self.intercept_ = np.array([b_final], dtype=np.float64)
            self._last_cpu_phase_timings["cpu_fit"] = (
                perf_counter() - cpu_method_started
            )
            _emit_progress(
                progress_callback,
                phase="final_refit",
                completed=1,
                total=1,
                backend="cpu",
                started_at=progress_started,
            )
            return self
    
        finally:
            # Clean shared memory segments
            try:
                X_sh = None; y_sh = None
                shmX.close(); shmy.close()
                shmX.unlink(); shmy.unlink()
            except Exception:
                pass


    def _c_count(self) -> int:
        if isinstance(self.Cs, (int, np.integer)):
            return int(self.Cs)
        return int(np.asarray(self.Cs).size)


    def _run_cpu_backend(
        self,
        X,
        y,
        *,
        requested: str,
        dtype_name: str,
        fit_started: float,
        fallback_reason: Optional[dict[str, Any]],
        auto_decision: Optional[dict[str, Any]],
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
        cancel_callback: Optional[Callable[[], bool]],
    ):
        cpu_started = perf_counter()
        self._fit_cpu(
            X,
            y,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            started_at=fit_started,
        )
        cpu_elapsed = perf_counter() - cpu_started
        total_elapsed = perf_counter() - fit_started
        if self.n_jobs in (None, -1):
            effective_jobs = min(self.cv, max(1, (os.cpu_count() or 1) - 1))
        else:
            effective_jobs = min(self.cv, max(1, int(self.n_jobs)))

        self.backend_requested_ = requested
        self.backend_used_ = "cpu"
        self.backend_provider_ = "numpy"
        self.device_id_ = None
        self.device_name_ = None
        self.dtype_ = dtype_name
        self.n_jobs_effective_ = effective_jobs
        self.fallback_reason_ = fallback_reason
        self.auto_decision_ = auto_decision
        self.fit_timings_ = {
            key: float(value)
            for key, value in getattr(self, "_last_cpu_phase_timings", {}).items()
        }
        self.fit_timings_["cpu_fit"] = float(cpu_elapsed)
        self.fit_timings_["total"] = float(total_elapsed)
        self.peak_device_memory_bytes_ = None
        self.runtime_versions_ = {"numpy": str(np.__version__)}
        rows, features = np.shape(X)[:2]
        self.backend_report_ = {
            "requested": requested,
            "used": "cpu",
            "provider": "numpy",
            "decision": auto_decision,
            "fallback": fallback_reason,
            "device": None,
            "dtype": dtype_name,
            "shape": {"rows": int(rows), "features": int(features)},
            "cv": int(self.cv),
            "c_values": self._c_count(),
            "solver": str(self.solver).lower(),
            "penalty": str(self.penalty).lower(),
            "n_jobs_effective": int(effective_jobs),
            "timings_seconds": dict(self.fit_timings_),
            "peak_device_memory_bytes": None,
            "runtime_versions": dict(self.runtime_versions_),
        }
        _emit_progress(
            progress_callback,
            phase="complete",
            completed=1,
            total=1,
            backend="cpu",
            started_at=fit_started,
        )
        return self


    def _fit_cuda(
        self,
        X,
        y,
        *,
        cuda_backend,
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
        cancel_callback: Optional[Callable[[], bool]],
        fit_started: float,
    ) -> dict[str, Any]:
        """Fit supported FISTA/hybrid configurations in one CUDA-owning process."""

        from ._cuda_solvers import (
            _CuPyFISTALogistic,
            _CuPyRidgeLogistic,
            cuda_binary_log_loss,
        )

        cp = cuda_backend.cp
        timings: dict[str, float] = {}
        transfer_bytes = 0
        _check_cancel(cancel_callback)
        _emit_progress(
            progress_callback,
            phase="host_to_device",
            completed=0,
            total=1,
            backend="cuda",
            started_at=fit_started,
        )

        if cuda_backend.is_device_array(X):
            X_source = X
        else:
            X_source = np.asarray(X, dtype=np.float64, order="F")
            transfer_bytes += int(X_source.nbytes)
        if cuda_backend.is_device_array(y):
            y_source = y
        else:
            y_source = np.asarray(y, dtype=int)
            transfer_bytes += int(y_source.nbytes)

        def transfer_inputs():
            X_device = cuda_backend.asarray(X_source, dtype=cp.float64, order="F")
            y_device = cuda_backend.asarray(y_source, dtype=cp.int64)
            return X_device, y_device

        (X_device, y_device), timings["host_to_device"] = cuda_backend.timed_host(
            transfer_inputs
        )
        if X_device.ndim != 2 or y_device.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X_device.shape[0] != y_device.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        n, p = map(int, X_device.shape)
        y_device_float = y_device.astype(cp.float64)
        _emit_progress(
            progress_callback,
            phase="host_to_device",
            completed=1,
            total=1,
            backend="cuda",
            started_at=fit_started,
        )

        if isinstance(self.Cs, (int, np.integer)):
            Cs = np.logspace(-4, 4, int(self.Cs))
        else:
            Cs = np.asarray(self.Cs, dtype=np.float64)
        self.Cs_ = Cs.copy()

        penalty = str(self.penalty).lower()
        if penalty not in {"l1", "adaptive_l1"}:
            raise ValueError("Only penalty='l1' and penalty='adaptive_l1' are supported.")
        if self.scoring != "neg_log_loss":
            raise ValueError("Only scoring='neg_log_loss' is supported.")
        if penalty == "adaptive_l1":
            if self.adaptive_eps <= 0.0:
                raise ValueError("adaptive_eps must be positive.")
            if self.adaptive_pilot_C <= 0.0:
                raise ValueError("adaptive_pilot_C must be positive.")

        self.adaptive_feature_scales_ = None
        self.adaptive_penalty_weights_ = None
        self.adaptive_pilot_coef_ = None
        self.adaptive_pilot_intercept_ = None
        self.adaptive_weighted_coef_ = None

        folds = self._kfold_indices(n)
        lam_full = 1.0 / (np.maximum(Cs, 1e-12) * float(n))
        order = np.argsort(lam_full)[::-1]
        Cs_path = Cs[order]
        total_paths = int(len(folds) * len(Cs_path))
        fold_losses_host: list[np.ndarray] = []
        pilot_elapsed = 0.0
        cv_started = perf_counter()

        for fold_index, val_idx in enumerate(folds):
            _check_cancel(cancel_callback)
            mask_host = np.ones(n, dtype=bool)
            mask_host[val_idx] = False
            mask = cuda_backend.asarray(mask_host, dtype=cp.bool_)
            val_device = cuda_backend.asarray(
                val_idx.astype(np.int64, copy=False), dtype=cp.int64
            )
            Xtr = X_device[mask, :]
            ytr = y_device_float[mask]
            Xva = X_device[val_device, :]
            yva = y_device_float[val_device]
            ntr = int(Xtr.shape[0])

            if penalty == "adaptive_l1":
                pilot_start = perf_counter()
                pilot_lam = 1.0 / (
                    float(self.adaptive_pilot_C) * max(float(ntr), 1.0)
                )
                pilot = _CuPyRidgeLogistic(
                    cp,
                    lam=pilot_lam,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    verbose=False,
                )
                pilot.fit(Xtr, ytr, cancel_callback=cancel_callback)
                feature_scales = cp.maximum(
                    cp.abs(pilot.w_) + float(self.adaptive_eps),
                    float(self.adaptive_eps),
                )
                Xtr_fit = Xtr * feature_scales[None, :]
                Xva_fit = Xva * feature_scales[None, :]
                cuda_backend.synchronize()
                pilot_elapsed += perf_counter() - pilot_start
            else:
                Xtr_fit = Xtr
                Xva_fit = Xva

            w_ws = cp.zeros(p, dtype=cp.float64)
            b_ws = None
            val_losses = np.empty(len(Cs_path), dtype=np.float64)
            for c_index, Ci in enumerate(Cs_path):
                _check_cancel(cancel_callback)
                lam_ci = 1.0 / (float(Ci) * float(ntr))
                solver = _CuPyFISTALogistic(
                    cp,
                    lam=lam_ci,
                    tol=self.tol,
                    max_iter=max(self.max_iter, 4000),
                    verbose=False,
                )
                solver.fit(
                    Xtr_fit,
                    ytr,
                    w0=w_ws,
                    b0=b_ws,
                    cancel_callback=cancel_callback,
                )
                z_val = Xva_fit @ solver.w_ + solver.b_
                val_losses[c_index] = float(
                    cuda_binary_log_loss(cp, yva, z_val).item()
                )
                w_ws = solver.w_
                b_ws = solver.b_
                completed = fold_index * len(Cs_path) + c_index + 1
                _emit_progress(
                    progress_callback,
                    phase="cv_path",
                    completed=completed,
                    total=total_paths,
                    backend="cuda",
                    started_at=fit_started,
                    fold=fold_index,
                    c_index=c_index,
                )
            fold_losses_host.append(val_losses)
            del mask, val_device, Xtr, ytr, Xva, yva, Xtr_fit, Xva_fit, w_ws, b_ws
            cuda_backend.observe_memory()

        cuda_backend.synchronize()
        timings["cv_path"] = perf_counter() - cv_started
        timings["adaptive_pilot"] = pilot_elapsed
        fold_losses = np.vstack(fold_losses_host)
        mean_losses = np.mean(fold_losses, axis=0)
        if self.cv > 1:
            se_losses = np.std(fold_losses, axis=0, ddof=1) / math.sqrt(self.cv)
        else:
            se_losses = np.zeros_like(mean_losses)
        self.cv_fold_losses_ = fold_losses.copy()
        self.cv_mean_losses_ = mean_losses.copy()
        self.cv_se_losses_ = se_losses.copy()
        self.cv_path_Cs_ = Cs_path.copy()

        if self.verbose:
            for i, Ci in enumerate(Cs_path):
                lam_dbg = 1.0 / (float(Ci) * float(n))
                print(
                    f"[CV|cuda] C={Ci:.4g} (lam~={lam_dbg:.3e}) -> "
                    f"mean={mean_losses[i]:.6f} +/- {se_losses[i]:.6f}"
                )

        selection_started = perf_counter()
        if self.cv_rule.lower() == "1se":
            j_best = int(np.argmin(mean_losses))
            threshold = mean_losses[j_best] + se_losses[j_best]
            candidates = np.where(mean_losses <= threshold)[0]
            j_pick = int(np.min(candidates)) if candidates.size else j_best
            self.C_ = float(Cs_path[j_pick])
        else:
            self.C_ = float(Cs_path[int(np.argmin(mean_losses))])
        timings["cv_selection"] = perf_counter() - selection_started

        _check_cancel(cancel_callback)
        _emit_progress(
            progress_callback,
            phase="final_refit",
            completed=0,
            total=1,
            backend="cuda",
            started_at=fit_started,
        )
        final_started = perf_counter()
        lam_final = 1.0 / (float(self.C_) * float(n))
        X_refit = X_device
        feature_scales = None
        if penalty == "adaptive_l1":
            pilot_lam = 1.0 / (
                float(self.adaptive_pilot_C) * max(float(n), 1.0)
            )
            pilot = _CuPyRidgeLogistic(
                cp,
                lam=pilot_lam,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=False,
            )
            pilot.fit(X_device, y_device_float, cancel_callback=cancel_callback)
            feature_scales = cp.maximum(
                cp.abs(pilot.w_) + float(self.adaptive_eps),
                float(self.adaptive_eps),
            )
            X_refit = X_device * feature_scales[None, :]
            self.adaptive_feature_scales_ = cuda_backend.to_numpy(feature_scales)
            self.adaptive_penalty_weights_ = 1.0 / self.adaptive_feature_scales_
            self.adaptive_pilot_coef_ = cuda_backend.to_numpy(pilot.w_).reshape(1, -1)
            self.adaptive_pilot_intercept_ = np.array(
                [float(pilot.b_.item())], dtype=np.float64
            )

        solver_name = str(self.solver).lower()
        if solver_name == "hybrid":
            transfer_started = perf_counter()
            X_refit_host = cuda_backend.to_numpy(X_refit)
            y_host = cuda_backend.to_numpy(y_device).astype(int, copy=False)
            timings["device_to_host"] = perf_counter() - transfer_started
            cpu_refit_started = perf_counter()
            all_pm1 = penalty == "l1" and np.all(
                (X_refit_host == 1) | (X_refit_host == -1)
            )
            final_solver = _CDLogistic(
                lam=lam_final,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose,
                kkt_tol=1e-4,
                all_pm1=all_pm1,
            )
            final_solver.fit(
                X_refit_host,
                y_host,
                w0=None,
                b0=None,
                lam_prev=None,
                active_init=None,
            )
            w_final = final_solver.w_.copy()
            b_final = float(final_solver.b_)
            timings["final_refit_cpu"] = perf_counter() - cpu_refit_started
        else:
            gpu_refit_started = perf_counter()
            final_solver = _CuPyFISTALogistic(
                cp,
                lam=lam_final,
                tol=self.tol,
                max_iter=max(self.max_iter, 4000),
                verbose=self.verbose,
            )
            final_solver.fit(
                X_refit,
                y_device_float,
                w0=None,
                b0=None,
                cancel_callback=cancel_callback,
            )
            cuda_backend.synchronize()
            timings["final_refit_gpu"] = perf_counter() - gpu_refit_started
            transfer_started = perf_counter()
            w_final = cuda_backend.to_numpy(final_solver.w_)
            b_final = float(final_solver.b_.item())
            timings["device_to_host"] = perf_counter() - transfer_started

        if feature_scales is not None:
            self.adaptive_weighted_coef_ = w_final.reshape(1, -1).copy()
            w_final = w_final * self.adaptive_feature_scales_
        if self.zero_clamp > 0.0:
            w_final[np.abs(w_final) <= self.zero_clamp] = 0.0
        timings["final_refit"] = perf_counter() - final_started

        logical_started = perf_counter()
        if self.logic_polish:
            if cuda_backend.is_device_array(X):
                X_logical = cuda_backend.to_numpy(X_device)
            else:
                X_logical = np.asarray(X, dtype=np.float64)
            y_logical = cuda_backend.to_numpy(y_device).astype(int, copy=False)
            w_new, b_new, j_new, adopted, figs, diag = self._logical_polish(
                X=X_logical,
                y=y_logical,
                w=w_final,
                b=b_final,
                K=self.logic_scale,
                target=self.logic_target,
                rel_tol=self.logic_rel_tol,
                maxk=self.logic_maxk,
                make_plots=self.logic_plot,
                Ks_plot=self.logic_Ks_plot,
                plot_dir=self.logic_plot_dir,
                verbose=self.verbose,
            )
            self.logic_figs_ = figs
            self.logic_diag_ = diag
            if adopted:
                w_final, b_final = w_new, b_new
                if self.verbose:
                    print(f"[logical] adopted rule-like model: J={j_new:.4f}")
        timings["logical_polish_cpu"] = perf_counter() - logical_started

        self.coef_ = np.asarray(w_final, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.array([b_final], dtype=np.float64)
        cuda_backend.synchronize()
        _emit_progress(
            progress_callback,
            phase="final_refit",
            completed=1,
            total=1,
            backend="cuda",
            started_at=fit_started,
        )
        return {
            "timings": timings,
            "transfer_bytes": int(transfer_bytes),
            "synchronization_count": int(cuda_backend.synchronization_count),
        }


    def fit(
        self,
        X,
        y,
        *,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ):
        """Fit using the requested CPU, CUDA, or automatically selected backend."""

        fit_started = perf_counter()
        requested = normalise_backend(self.backend)
        dtype_name = normalise_dtype(self.dtype)
        device_id = normalise_device(self.device)
        self.backend_requested_ = requested
        self.backend_used_ = None
        self.backend_provider_ = None
        self.device_id_ = None
        self.device_name_ = None
        self.dtype_ = dtype_name
        self.n_jobs_effective_ = None
        self.fallback_reason_ = None
        self.auto_decision_ = None
        self.fit_timings_ = {}
        self.peak_device_memory_bytes_ = None
        self.runtime_versions_ = {}
        self.backend_report_ = {}
        self.Cs_ = None
        self.C_ = None
        self.coef_ = None
        self.intercept_ = None
        self.cv_fold_losses_ = None
        self.cv_mean_losses_ = None
        self.cv_se_losses_ = None
        self.cv_path_Cs_ = None
        self.logic_diag_ = {}
        self.logic_figs_ = []
        self.adaptive_feature_scales_ = None
        self.adaptive_penalty_weights_ = None
        self.adaptive_pilot_coef_ = None
        self.adaptive_pilot_intercept_ = None
        self.adaptive_weighted_coef_ = None
        callback_failures: list[Exception] = []

        if progress_callback is not None:
            user_progress_callback = progress_callback

            def progress_proxy(event: dict[str, Any]) -> None:
                try:
                    user_progress_callback(event)
                except Exception as exc:
                    callback_failures.append(exc)
                    raise

            progress_callback = progress_proxy

        if cancel_callback is not None:
            user_cancel_callback = cancel_callback

            def cancel_proxy() -> bool:
                try:
                    return bool(user_cancel_callback())
                except Exception as exc:
                    callback_failures.append(exc)
                    raise

            cancel_callback = cancel_proxy

        _check_cancel(cancel_callback)

        shape = np.shape(X)
        if len(shape) != 2:
            raise ValueError("X must be a 2D array.")
        rows, features = map(int, shape)
        work = estimate_work(
            rows,
            features,
            self.cv,
            self._c_count(),
            self.penalty,
        )
        required_bytes = estimate_cuda_bytes(
            rows, features, self.cv, self.penalty
        )
        auto_decision: Optional[dict[str, Any]] = None
        try:
            auto_threshold = max(
                1, int(os.environ.get("CUTLASS_CUDA_AUTO_MIN_WORK", "75000000"))
            )
        except ValueError:
            auto_threshold = 75_000_000

        if requested == "cpu":
            return self._run_cpu_backend(
                X,
                y,
                requested=requested,
                dtype_name=dtype_name,
                fit_started=fit_started,
                fallback_reason=None,
                auto_decision=None,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        supported, capability_code = cuda_supports(self.solver, dtype_name)
        if not supported:
            message = (
                f"CUDA does not support solver='{self.solver}' with dtype='{dtype_name}'."
            )
            if requested == "auto":
                auto_decision = {
                    "selected": "cpu",
                    "reason": capability_code,
                    "estimated_work": int(work),
                    "estimated_device_bytes": int(required_bytes),
                }
                return self._run_cpu_backend(
                    X,
                    y,
                    requested=requested,
                    dtype_name=dtype_name,
                    fit_started=fit_started,
                    fallback_reason=None,
                    auto_decision=auto_decision,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
            if not self.allow_cpu_fallback:
                raise BackendConfigurationError(message)
            fallback = {
                "code": capability_code,
                "phase": "backend_resolution",
                "message": message,
            }
            warnings.warn(message + " Falling back to CPU.", CutlassBackendWarning, stacklevel=2)
            return self._run_cpu_backend(
                X,
                y,
                requested=requested,
                dtype_name=dtype_name,
                fit_started=fit_started,
                fallback_reason=fallback,
                auto_decision=None,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        if requested == "auto" and work < auto_threshold:
            auto_decision = {
                "selected": "cpu",
                "reason": "below_crossover",
                "estimated_work": int(work),
                "threshold": int(auto_threshold),
                "estimated_device_bytes": int(required_bytes),
                "policy_version": 1,
            }
            return self._run_cpu_backend(
                X,
                y,
                requested=requested,
                dtype_name=dtype_name,
                fit_started=fit_started,
                fallback_reason=None,
                auto_decision=auto_decision,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        try:
            from ._cuda_backend import CudaBackend

            cuda_backend = CudaBackend(device=device_id)
        except Exception as exc:
            if isinstance(exc, FitCancelledError):
                raise
            message = f"CUDA backend is unavailable ({type(exc).__name__})."
            if requested == "auto":
                auto_decision = {
                    "selected": "cpu",
                    "reason": "cuda_unavailable",
                    "estimated_work": int(work),
                    "estimated_device_bytes": int(required_bytes),
                }
                return self._run_cpu_backend(
                    X,
                    y,
                    requested=requested,
                    dtype_name=dtype_name,
                    fit_started=fit_started,
                    fallback_reason=None,
                    auto_decision=auto_decision,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
            if not self.allow_cpu_fallback:
                if isinstance(exc, BackendUnavailableError):
                    raise
                raise BackendUnavailableError(message) from exc
            fallback = {
                "code": "cuda_unavailable",
                "phase": "backend_resolution",
                "message": message,
            }
            warnings.warn(message + " Falling back to CPU.", CutlassBackendWarning, stacklevel=2)
            return self._run_cpu_backend(
                X,
                y,
                requested=requested,
                dtype_name=dtype_name,
                fit_started=fit_started,
                fallback_reason=fallback,
                auto_decision=None,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        memory = cuda_backend.memory_info()
        memory_safe = required_bytes <= int(memory["free_bytes"] * 0.8)

        if requested == "auto":
            if not memory_safe:
                auto_decision = {
                    "selected": "cpu",
                    "reason": "memory_preflight",
                    "estimated_work": int(work),
                    "threshold": int(auto_threshold),
                    "estimated_device_bytes": int(required_bytes),
                    "free_device_bytes": int(memory["free_bytes"]),
                    "policy_version": 1,
                }
                return self._run_cpu_backend(
                    X,
                    y,
                    requested=requested,
                    dtype_name=dtype_name,
                    fit_started=fit_started,
                    fallback_reason=None,
                    auto_decision=auto_decision,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
            auto_decision = {
                "selected": "cuda",
                "reason": "supported_workload",
                "estimated_work": int(work),
                "threshold": int(auto_threshold),
                "estimated_device_bytes": int(required_bytes),
                "free_device_bytes": int(memory["free_bytes"]),
                "policy_version": 1,
            }
        elif not memory_safe:
            message = "CUDA memory preflight rejected the fit."
            if not self.allow_cpu_fallback:
                raise BackendExecutionError(message)
            fallback = {
                "code": "memory_preflight",
                "phase": "backend_resolution",
                "message": message,
            }
            warnings.warn(message + " Falling back to CPU.", CutlassBackendWarning, stacklevel=2)
            return self._run_cpu_backend(
                X,
                y,
                requested=requested,
                dtype_name=dtype_name,
                fit_started=fit_started,
                fallback_reason=fallback,
                auto_decision=None,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        try:
            with cuda_backend.device:
                details = self._fit_cuda(
                    X,
                    y,
                    cuda_backend=cuda_backend,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                    fit_started=fit_started,
                )
        except FitCancelledError:
            raise
        except Exception as exc:
            if callback_failures and exc is callback_failures[-1]:
                raise
            out_of_memory = cuda_backend.is_out_of_memory(exc)
            code = "cuda_out_of_memory" if out_of_memory else "cuda_execution_failed"
            if out_of_memory:
                cuda_backend.clear_unused()
            message = f"CUDA fit failed during execution ({type(exc).__name__})."
            if not self.allow_cpu_fallback:
                raise BackendExecutionError(message) from exc
            fallback = {
                "code": code,
                "phase": "cuda_fit",
                "message": message,
                "process_restart_recommended": not out_of_memory,
            }
            warnings.warn(message + " Falling back to CPU.", CutlassBackendWarning, stacklevel=2)
            try:
                X_cpu = cuda_backend.to_numpy(X) if cuda_backend.is_device_array(X) else X
                y_cpu = cuda_backend.to_numpy(y) if cuda_backend.is_device_array(y) else y
            except Exception as transfer_exc:
                raise BackendExecutionError(
                    "CUDA failed and device input could not be recovered for CPU fallback."
                ) from transfer_exc
            return self._run_cpu_backend(
                X_cpu,
                y_cpu,
                requested=requested,
                dtype_name=dtype_name,
                fit_started=fit_started,
                fallback_reason=fallback,
                auto_decision=auto_decision,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        total_elapsed = perf_counter() - fit_started
        timings = dict(details["timings"])
        timings["total"] = float(total_elapsed)
        self.backend_requested_ = requested
        self.backend_used_ = "cuda"
        self.backend_provider_ = "cupy"
        self.device_id_ = int(cuda_backend.device_id)
        self.device_name_ = cuda_backend.device_name
        self.dtype_ = dtype_name
        self.n_jobs_effective_ = 1
        self.fallback_reason_ = None
        self.auto_decision_ = auto_decision
        self.fit_timings_ = timings
        self.peak_device_memory_bytes_ = int(cuda_backend.peak_used_bytes)
        self.runtime_versions_ = cuda_backend.versions
        self.backend_report_ = {
            "requested": requested,
            "used": "cuda",
            "provider": "cupy",
            "decision": auto_decision,
            "fallback": None,
            "device": {
                "id": int(cuda_backend.device_id),
                "name": cuda_backend.device_name,
                "compute_capability": cuda_backend.status.compute_capability,
            },
            "dtype": dtype_name,
            "shape": {"rows": rows, "features": features},
            "cv": int(self.cv),
            "c_values": self._c_count(),
            "solver": str(self.solver).lower(),
            "penalty": str(self.penalty).lower(),
            "n_jobs_effective": 1,
            "transfer_bytes": int(details["transfer_bytes"]),
            "synchronization_count": int(details["synchronization_count"]),
            "timings_seconds": dict(timings),
            "peak_device_memory_bytes": int(cuda_backend.peak_used_bytes),
            "peak_reserved_device_memory_bytes": int(cuda_backend.peak_reserved_bytes),
            "runtime_versions": dict(self.runtime_versions_),
        }
        _emit_progress(
            progress_callback,
            phase="complete",
            completed=1,
            total=1,
            backend="cuda",
            started_at=fit_started,
        )
        return self


    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.coef_.ravel() + float(self.intercept_[0])
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])
