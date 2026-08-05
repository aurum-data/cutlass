# CUTLASS

CUTLASS (Critical-range rectified LASSO) packages the workflow developed in the
project scripts into a reusable, publishable Python library.  It exposes a
scikit-learn inspired estimator that rectifies the input space into
\{-1, +1\} indicators, trains an L1-penalised logistic model with an efficient
coordinate-descent solver, and optionally compresses the model into a logical
rule without any dependence on scikit-learn itself. Version 0.6.0 adds an
optional CUDA backend for FISTA and adaptive-L1 fitting while preserving the
NumPy backend as the default and scientific reference.

This project is a statistical modelling package and is not NVIDIA's C++
CUTLASS linear-algebra library.

## Features

- **Rectifier transformer** that infers critical ranges from the positive class
  and binarises features into \{-1, +1\}.
- **Cross-validated L1 logistic model** with warm-started coordinate descent
  and optional FISTA solver.
- **Optional CUDA execution** through CuPy for FISTA cross-validation and final
  fitting, including a hybrid GPU-FISTA/CPU-coordinate-descent mode.
- **Adaptive-L1 mode** (`penalty="adaptive_l1"`) that fits an L2 logistic pilot,
  reweights the L1 penalty by `abs(beta_pilot) + adaptive_eps`, and maps
  coefficients back to the original feature scale.
- **Logical compression** step mirroring the research code (top-k votes with
  fixed magnitude `K` and several intercept policies).
- **Serialization helpers** to persist rectifier limits, fitted weights, and
  backend provenance.
- **Observable execution** through backend reports, synchronized phase timings,
  progress callbacks, cancellation, and GPU memory/transfer diagnostics.
- Lightweight CPU installation based on NumPy and pandas. Matplotlib and CuPy
  are optional extras for plots and CUDA execution respectively.

## Execution model

CPU remains the default so existing results and installations are unchanged.
The `backend` argument is available on both `CutlassLogisticCV` and
`CutlassClassifier`:

| Setting | Behaviour |
| --- | --- |
| `backend="cpu"` | Always use the NumPy reference implementation. |
| `backend="cuda"` | Require CUDA unless `allow_cpu_fallback=True`. |
| `backend="auto"` | Select CUDA only when it is usable, the solver supports it, and the estimated workload is large enough. |

Solver support is explicit:

| Solver | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `cd` | Yes | No | Coordinate descent; CUDA requests visibly fall back or raise. |
| `fista` | Yes | Yes | CV paths and final refit run on the selected backend. |
| `hybrid` | Yes | Yes | FISTA CV paths on CUDA, final sparse coordinate-descent refit on CPU. |
| `saga`, `liblinear` | Yes | No | Compatibility aliases implemented by the CPU path. |

Adaptive L1 is supported by `cd`, `fista`, and `hybrid`. Logical polishing is
always a CPU post-processing phase, including after a CUDA fit.

## Installation

```bash
pip install cutlass
```

The plotting utilities used by the logical compression step are optional.  To
enable them, install the `plots` extra:

```bash
pip install cutlass[plots]
```

CUDA is optional and requires a compatible NVIDIA driver. Install exactly one
CuPy provider matching the CUDA major version supported by the environment:

```bash
pip install "cutlass[cuda13]"
```

Use `cuda12` instead for a CUDA 12 environment. Do not install multiple CuPy
distributions in the same environment. CuPy is imported lazily, so the base
package remains usable on systems without CUDA.

## Quick start

```python
import pandas as pd
from cutlass import CutlassClassifier

# toy binary dataset
df = pd.DataFrame(
    {
        "feat_a": [0.1, 0.3, 0.7, 0.9, 0.2, 0.8],
        "feat_b": [10, 13, 8, 5, 11, 4],
        "INDC": [0, 0, 1, 1, 0, 1],
    }
)

X = df.drop(columns=["INDC"])
y = df["INDC"]

clf = CutlassClassifier(
    rectify=True,
    Cs=15,
    solver="cd",
    cv=3,
    logic_polish=True,
    logic_scale=10.0,
)
clf.fit(X, y)
print(clf.predict_proba(X))
print("limits:", clf.limits_)
```

The default penalty remains standard L1. To use the adaptive-L1 mode, pass the
optional penalty argument:

```python
adaptive_clf = CutlassClassifier(
    rectify=True,
    Cs=15,
    solver="cd",
    cv=3,
    penalty="adaptive_l1",
    adaptive_eps=1e-3,
)
adaptive_clf.fit(X, y)
```

To fit the FISTA CV paths on CUDA and retain coordinate descent for the final
sparse refit:

```python
from cutlass import CutlassLogisticCV, probe_backend

print(probe_backend("cuda", device=0).to_dict())

gpu_model = CutlassLogisticCV(
    Cs=15,
    cv=3,
    solver="hybrid",
    backend="cuda",
    device=0,
    dtype="float64",
    allow_cpu_fallback=True,
)
gpu_model.fit(X.to_numpy(), y)
print(gpu_model.backend_used_)
print(gpu_model.backend_report_)
```

`solver="fista"` runs both CV and final fitting on CUDA. The coordinate-descent
solver is CPU-only; requesting `backend="cuda"` with `solver="cd"` either falls
back visibly or raises when `allow_cpu_fallback=False`.

`backend="auto"` uses a deterministic policy. It currently selects CUDA for a
compatible solver when `n_rows * n_features * n_folds * n_C_values` is at least
75,000,000 work units (doubled for adaptive L1), unless
`CUTLASS_CUDA_AUTO_MIN_WORK` overrides that threshold. This prevents transfer and
startup overhead from slowing down small fits.

CUDA inputs may be NumPy arrays or CuPy device arrays. Fitted public attributes
and predictions are returned as NumPy arrays so serialization and downstream
code behave the same on every backend.

### Progress, cancellation, and diagnostics

Long-running fits can report phase progress and stop cooperatively:

```python
cancelled = False
gpu_model.fit(
    X.to_numpy(),
    y,
    progress_callback=lambda event: print(
        event["phase"], event["completed"], event["total"]
    ),
    cancel_callback=lambda: cancelled,
)
```

After fitting, inspect `backend_requested_`, `backend_used_`,
`backend_provider_`, `device_name_`, `dtype_`, `n_jobs_effective_`,
`auto_decision_`, `fit_timings_`, and `backend_report_`. The report also records
fallback reasons, runtime versions, transfers, synchronization points, and peak
observed GPU memory. Backend discovery is available through `list_devices()`
and `probe_backend()` without constructing an estimator.

## Vignettes

Additional step-by-step guides live under `docs/vignettes/`:

- [Basic rectified workflow](docs/vignettes/01_basic_rectified_workflow.md) - reproduce the CPU reference fit.
- [Logical polish](docs/vignettes/02_logical_polish.md) - enable logical compression and interpret diagnostics.
- [Batch experiments](docs/vignettes/03_batch_experiments.md) - run experiments and retain backend provenance.
- [GPU backend](docs/vignettes/04_gpu_backend.md) - configure CUDA, Auto mode, fallback, progress, and persistent services.
- [GPU implementation](docs/GPU_implementation.md) - architecture, delivered scope, and validation status.

## API highlights

- `cutlass.Rectifier`: transformer implementing the critical-range binarisation.
- `cutlass.CutlassLogisticCV`: lower-level L1 or adaptive-L1 logistic with
  cross-validation.
- `cutlass.CutlassClassifier`: full workflow composed of the rectifier,
  optional scaling, and the logistic path solver. Use `penalty="l1"` for the
  default behavior or `penalty="adaptive_l1"` for the adaptive mode.
- `cutlass.list_devices` and `cutlass.probe_backend`: runtime discovery and an
  allocation-based health check for applications and service startup.
- `cutlass.FitProgress`: the schema used to create JSON-safe progress
  dictionaries delivered to callbacks.
- `cutlass.BackendUnavailableError`, `cutlass.BackendConfigurationError`,
  `cutlass.BackendExecutionError`, and `cutlass.FitCancelledError`: actionable
  execution failures that applications can handle separately.
- `cutlass.serialization`: helpers for saving rectifier limits and fitted
  weights. Model artifacts include a JSON-safe backend provenance report.

Refer to the docstrings for detailed parameter descriptions; they mirror the
research scripts so existing experiment drivers can be migrated with minimal
changes.

## Development

To build the package locally:

```bash
python -m build
```

To update the project on PyPI, first bump `version` in `pyproject.toml`,
commit the release changes, and create a clean source/wheel build with
`python -m build`.  After confirming the files under `dist/` are correct,
upload them with `python -m twine upload dist/*` using an account or API token
that has permission to publish the `cutlass` package.

Run the CPU suite on any supported Python environment:

```bash
python -m pytest -m "not cuda"
```

In an environment with a usable NVIDIA GPU and CuPy provider, run the complete
suite (CUDA tests skip automatically when the runtime is unavailable):

```bash
python -m pytest
```

## License

MIT License.  See `LICENSE` for details.
