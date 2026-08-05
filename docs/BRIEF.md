# CUTLASS Brief

## Purpose

CUTLASS (Critical-range rectified LASSO) is a lightweight Python package that implements the research workflow for
rectified L1-penalized logistic regression, optional adaptive-L1 reweighting, optional CUDA FISTA execution, and optional rule compression. It targets interpretable, sparse binary
classifiers by (1) rectifying numeric features into {-1, +1} indicators using class-conditional critical ranges,
(2) fitting an L1 logistic model with cross-validation, and (3) optionally "polishing" the model into a fixed-magnitude
logical rule optimized for Youden's J.

The base installation is intentionally minimal (NumPy + pandas). Matplotlib is
optional for plots and exactly one CuPy provider is optional for CUDA. The CPU
backend remains the numerical reference and the default execution mode.

## Core Workflow (High Level)

- Rectify: infer per-feature critical ranges from the positive class, then binarize features into {-1, +1}.
- Scale: optionally standardize features when not already binary.
- Fit: cross-validate an L1 logistic model across a C grid with warm starts and parallelized folds. The default is
  standard L1; version 0.5.0 adds `penalty="adaptive_l1"`, which fits an L2 pilot model, reweights the L1 design by
  `abs(beta_pilot) + adaptive_eps`, and maps final coefficients back to the original feature axis. Version 0.6.0 adds
  optional `cpu`, `cuda`, and `auto` execution backends for FISTA and hybrid fitting while retaining CPU as the default.
- Polish (optional): compress to a top-k rule with fixed magnitude K and intercept policies, adopting if Youden's J
  stays within a user-defined tolerance.

## Package Structure (src/cutlass)

- __init__.py: package exports, including estimators, preprocessing, metrics,
  backend discovery, progress/cancellation helpers, and execution exceptions.
- model.py: CutlassClassifier end-to-end estimator; handles rectification, scaling, fitting, and predictions.
- linear_model.py: CutlassLogisticCV (L1/adaptive-L1 logistic CV, parallel fold evaluation, optional logical polish).
- _solvers.py: low-level solvers (_CDLogistic coordinate descent, _FISTALogistic proximal gradient).
- preprocessing.py: Rectifier and a minimal StandardScaler; grouping heuristic by feature name prefix.
- metrics.py: Youden's J, ROC AUC, and precision-recall curve implementations.
- serialization.py: persistence helpers (rectifier limits JSON and fitted model
  NPZ with backend provenance).
- pipeline.py: minimal Pipeline used by experiment scripts (fit + predict_proba).
- _math.py: numerical helpers (sigmoid, softplus, log loss, soft-threshold).
- acceleration.py / exceptions.py: public backend health, progress, fallback, and error contracts.
- _backend.py / _cuda_backend.py / _cuda_solvers.py: lazy backend selection, CuPy device management, and FP64 CUDA FISTA/ridge solvers.

## Repository Guide

- `examples/quickstart.py`: minimal CPU usage example.
- `docs/vignettes/`: CPU reference, logical polish, batch, and GPU guides.
- `docs/GPU_implementation.md`: GPU architecture, implementation status, and
  validation record.
- `tests/`: CPU contract tests plus CUDA-marked parity, fallback, callback,
  cancellation, and serialization tests.

Generated experiment datasets, run outputs, and research papers are not part of
the distributable package.

## Backend and Solver Contract

| Solver | CPU | CUDA | Final-refit behaviour |
| --- | --- | --- | --- |
| `cd` | Yes | No | Coordinate descent on CPU. |
| `fista` | Yes | Yes | Selected backend for CV and final refit. |
| `hybrid` | Yes | Yes | CUDA FISTA for CV; CPU coordinate descent for final refit. |
| `saga`, `liblinear` | Yes | No | CPU compatibility aliases. |

- `backend="cpu"` is deterministic and never initializes CUDA.
- `backend="cuda"` uses the requested device or follows the explicit fallback
  policy.
- `backend="auto"` requires a usable device, a CUDA-capable solver, and at
  least 75,000,000 estimated work units by default. Work is
  `rows * features * folds * C values`, doubled for adaptive L1. The
  `CUTLASS_CUDA_AUTO_MIN_WORK` environment variable can override the threshold.
- CUDA folds run sequentially within one fit. `n_jobs` is normalized to one on
  CUDA and its effective value is reported.
- The current CUDA implementation is FP64. Public fitted attributes and
  predictions are NumPy arrays; CuPy arrays may be supplied to the lower-level
  estimator without a host-to-device input transfer.
- Logical polishing remains a visible CPU phase. Applications should own one
  persistent process per GPU and queue GPU jobs rather than spawning a process
  per fit.

Every fitted estimator exposes the requested and used backend, provider,
device, dtype, Auto decision, effective parallelism, phase timings, and a
JSON-safe `backend_report_`. Explicit CUDA failures distinguish unavailable,
configuration, execution, and cancellation errors.

## Architecture & Design Principles

- **Standalone Implementation**: CUTLASS avoids a scikit-learn dependency to maintain tight control over its specialized data transformations (critical range rectification) and optimization path (L1 coordinate descent, with an internal L2 pilot for adaptive L1). It closely mimics the `fit(X, y)` and `predict_proba(X)` API to stay intuitive.
- **Performance**: CPU math uses vectorized NumPy. CUDA FISTA and ridge kernels
  use CuPy with device-resident CV data, warm starts across the C path, explicit
  synchronization at observation boundaries, and observed memory diagnostics.
- **Dependencies**: Standard PEP 621 packaging (`pyproject.toml`). Base dependencies are strictly `numpy` and `pandas`, with `matplotlib`, CUDA 12, and CUDA 13 support available through explicit optional extras.

## Agent / Developer Modification Guide

This section provides a direct mapping of developer intentions to files and concepts, allowing an AI agent or human to modify the codebase exactly where necessary:

- **Modifying Optimization / Solvers**: CPU coordinate descent, FISTA, and ridge
  live in `src/cutlass/_solvers.py`; CUDA FISTA and ridge live in
  `src/cutlass/_cuda_solvers.py`.
- **Modifying GPU Execution**: Backend selection and admission live in `src/cutlass/_backend.py`; CuPy runtime/device behavior lives in `_cuda_backend.py`; CUDA FISTA and ridge math live in `_cuda_solvers.py`; estimator dispatch lives in `linear_model.py`.
- **Modifying Logical Rules & Compression**: To alter how models are "polished" into logical rules or rounded to top-k elements, modify `src/cutlass/linear_model.py` (`CutlassLogisticCV`).
- **Modifying Feature Binarization (Rectification)**: Changes to how critical ranges are computed from the positive class or how continuous variables are binarized must safely go in `src/cutlass/preprocessing.py` (`Rectifier`).
- **Modifying Overall Pipeline / Wrappers**: To change arguments exposed to users or how the rectifier and model are chained together, look at `src/cutlass/model.py` (`CutlassClassifier`). The adaptive mode is exposed as `CutlassClassifier(penalty="adaptive_l1")`; omitting `penalty` preserves the standard L1 default.
- **Modifying Metrics / Validation**: New evaluators or changes to Youden's J, AUC, or other metrics should be done in `src/cutlass/metrics.py`.
- **Validating Changes**: Run `python -m pytest -m "not cuda"` for the portable
  CPU contract. In a configured GPU environment, run `python -m pytest` for the
  CUDA parity and runtime suite. Build both wheel and source distribution with
  `python -m build` after packaging changes.
- **Build / Packaging**: The library uses standard tools (`python -m build`). Update `pyproject.toml` if modifying dependencies or metadata.
