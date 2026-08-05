"""CuPy implementations of CUTLASS's GPU-suitable logistic solvers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .exceptions import FitCancelledError


def _check_cancel(cancel_callback: Optional[Callable[[], bool]]) -> None:
    if cancel_callback is not None and bool(cancel_callback()):
        raise FitCancelledError("CUTLASS fit was cancelled.")


def _sigmoid(cp, z):
    z = cp.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + cp.exp(-z))


def _softplus(cp, z):
    return cp.log1p(cp.exp(-cp.abs(z))) + cp.maximum(z, 0.0)


def _binary_log_loss_from_logits(cp, y, z):
    return cp.mean(_softplus(cp, z) - y * z)


def _soft_threshold(cp, w, threshold):
    return cp.sign(w) * cp.maximum(cp.abs(w) - threshold, 0.0)


class _CuPyFISTALogistic:
    """CUDA FISTA matching the NumPy reference solver's update order."""

    def __init__(
        self,
        cp,
        lam: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 4000,
        step: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.cp = cp
        self.lam = float(lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.step = step
        self.verbose = bool(verbose)
        self.w_ = None
        self.b_ = None
        self.n_iter_ = 0

    def _estimate_L(self, X) -> float:
        cp = self.cp
        n, p = X.shape
        v_host = np.random.default_rng(123).standard_normal(p)
        v_host /= np.linalg.norm(v_host) + 1e-12
        v = cp.asarray(v_host, dtype=cp.float64)
        for _ in range(12):
            Xv = X @ v
            v = X.T @ Xv
            nv = cp.linalg.norm(v)
            nv_host = float(nv.item())
            if nv_host == 0.0:
                break
            v /= nv
        smax_sq = float((cp.linalg.norm(X @ v) ** 2).item())
        return 0.25 * smax_sq / max(n, 1) + 0.25

    def fit(
        self,
        X,
        y,
        *,
        w0=None,
        b0=None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> "_CuPyFISTALogistic":
        cp = self.cp
        X = cp.asarray(X, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.float64)
        n, p = X.shape
        w = cp.zeros(p, dtype=cp.float64) if w0 is None else cp.asarray(w0, dtype=cp.float64).copy()

        if b0 is None:
            py = float(cp.clip(cp.mean(y), 1e-6, 1 - 1e-6).item())
            b = cp.asarray(np.log(py / (1.0 - py)), dtype=cp.float64)
        else:
            b = cp.asarray(b0, dtype=cp.float64).copy()

        L = self._estimate_L(X) if self.step is None else 1.0 / self.step
        tstep = 0.9 / L
        w_y = w.copy()
        b_y = b.copy()
        t = 1.0
        prev_obj = np.inf

        for it in range(1, self.max_iter + 1):
            if it == 1 or it % 10 == 0:
                _check_cancel(cancel_callback)
            z = X @ w_y + b_y
            p_hat = _sigmoid(cp, z)
            grad_w = (X.T @ (p_hat - y)) / max(float(n), 1.0)
            grad_b = cp.sum(p_hat - y) / max(float(n), 1.0)

            found = False
            bt = 0
            while not found and bt < 20:
                w_new = _soft_threshold(cp, w_y - tstep * grad_w, tstep * self.lam)
                b_new = b_y - tstep * grad_b
                z_new = X @ w_new + b_new
                obj_new = _binary_log_loss_from_logits(cp, y, z_new) + self.lam * cp.sum(cp.abs(w_new))
                dz_w = w_new - w_y
                dz_b = b_new - b_y
                quad = (
                    _binary_log_loss_from_logits(cp, y, z)
                    + cp.dot(grad_w, dz_w)
                    + grad_b * dz_b
                    + (cp.linalg.norm(dz_w) ** 2 + dz_b * dz_b) / (2 * tstep)
                )
                if bool((obj_new <= quad + 1e-12).item()):
                    found = True
                else:
                    tstep *= 0.5
                    bt += 1

            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            w_acc = w_new + ((t - 1.0) / t_new) * (w_new - w)
            b_acc = b_new + ((t - 1.0) / t_new) * (b_new - b)
            restart = cp.dot(w_acc - w_new, w_new - w) + (b_acc - b_new) * (b_new - b)
            if bool((restart > 0).item()):
                w_y, b_y = w_new, b_new
                t = 1.0
            else:
                w_y, b_y = w_acc, b_acc
                t = t_new

            dw = cp.linalg.norm(w_new - w)
            db = cp.abs(b_new - b)
            w, b = w_new, b_new
            self.n_iter_ = it
            threshold = self.tol * (1.0 + cp.linalg.norm(w) + cp.abs(b))
            if bool((dw + db <= threshold).item()):
                break

            z_curr = X @ w + b
            obj = float(
                (_binary_log_loss_from_logits(cp, y, z_curr) + self.lam * cp.sum(cp.abs(w))).item()
            )
            if obj > prev_obj + 1e-10:
                t = 1.0
                w_y, b_y = w, b
            prev_obj = obj

        self.w_ = w
        self.b_ = b
        return self

    def predict_proba(self, X):
        if self.w_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        p = _sigmoid(self.cp, X @ self.w_ + self.b_)
        return self.cp.column_stack([1.0 - p, p])


class _CuPyRidgeLogistic:
    """CUDA L2 logistic solver used as the adaptive-L1 pilot."""

    def __init__(
        self,
        cp,
        lam: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 2000,
        verbose: bool = False,
    ) -> None:
        self.cp = cp
        self.lam = float(lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.w_ = None
        self.b_ = None
        self.n_iter_ = 0

    def _estimate_L(self, X) -> float:
        cp = self.cp
        n, p = X.shape
        v_host = np.random.default_rng(123).standard_normal(p)
        norm = np.linalg.norm(v_host)
        if norm == 0.0:
            return 0.25 + self.lam
        v = cp.asarray(v_host / norm, dtype=cp.float64)
        for _ in range(12):
            Xv = X @ v
            v = X.T @ Xv
            nv = cp.linalg.norm(v)
            nv_host = float(nv.item())
            if nv_host == 0.0:
                break
            v /= nv
        smax_sq = float((cp.linalg.norm(X @ v) ** 2).item())
        return 0.25 * smax_sq / max(n, 1) + 0.25 + self.lam

    def _objective(self, X, y, w, b):
        z = X @ w + b
        return _binary_log_loss_from_logits(self.cp, y, z) + 0.5 * self.lam * self.cp.dot(w, w)

    def fit(
        self,
        X,
        y,
        *,
        w0=None,
        b0=None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> "_CuPyRidgeLogistic":
        cp = self.cp
        X = cp.asarray(X, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.float64)
        n, p = X.shape
        w = cp.zeros(p, dtype=cp.float64) if w0 is None else cp.asarray(w0, dtype=cp.float64).copy()

        if b0 is None:
            py = float(cp.clip(cp.mean(y), 1e-6, 1 - 1e-6).item())
            b = cp.asarray(np.log(py / (1.0 - py)), dtype=cp.float64)
        else:
            b = cp.asarray(b0, dtype=cp.float64).copy()

        L = max(self._estimate_L(X), 1e-12)
        step = 0.9 / L
        w_y = w.copy()
        b_y = b.copy()
        t = 1.0
        prev_obj = float(self._objective(X, y, w, b).item())

        for it in range(1, self.max_iter + 1):
            if it == 1 or it % 10 == 0:
                _check_cancel(cancel_callback)
            z = X @ w_y + b_y
            p_hat = _sigmoid(cp, z)
            grad_w = (X.T @ (p_hat - y)) / max(float(n), 1.0) + self.lam * w_y
            grad_b = cp.mean(p_hat - y)
            w_new = w_y - step * grad_w
            b_new = b_y - step * grad_b
            obj_new = float(self._objective(X, y, w_new, b_new).item())

            if obj_new > prev_obj + 1e-10:
                w_y = w.copy()
                b_y = b.copy()
                t = 1.0
                step *= 0.5
                continue

            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            w_acc = w_new + ((t - 1.0) / t_new) * (w_new - w)
            b_acc = b_new + ((t - 1.0) / t_new) * (b_new - b)
            dw = cp.linalg.norm(w_new - w)
            db = cp.abs(b_new - b)
            w, b = w_new, b_new
            w_y, b_y = w_acc, b_acc
            t = t_new
            prev_obj = obj_new
            self.n_iter_ = it

            threshold = self.tol * (1.0 + cp.linalg.norm(w) + cp.abs(b))
            if bool((dw + db <= threshold).item()):
                break

        self.w_ = w
        self.b_ = b
        return self

    def predict_proba(self, X):
        if self.w_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        p = _sigmoid(self.cp, X @ self.w_ + self.b_)
        return self.cp.column_stack([1.0 - p, p])


def cuda_binary_log_loss(cp, y, z):
    """Expose device log loss to the CUDA CV orchestrator."""

    return _binary_log_loss_from_logits(cp, y, z)
