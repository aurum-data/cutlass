# Optional CUDA Backend

CUTLASS 0.6 provides an optional CuPy backend for FISTA and adaptive-L1 model
fitting. CPU remains the default, so existing scripts retain the NumPy reference
implementation unless they opt in.

## Install and probe

Install the extra matching the CUDA major version supported by the environment:

```powershell
python -m pip install "cutlass[cuda13]"
```

Then run a real allocation and calculation health check:

```python
from cutlass import list_devices, probe_backend

for device in list_devices("cuda"):
    print(device.to_dict())

status = probe_backend("cuda", device=0)
if not status.available:
    raise RuntimeError(status.error_message)
```

Importing CUTLASS and using the CPU backend do not import CuPy or initialize a
CUDA context.

Install only one CuPy provider in an environment. Use the `cuda12` extra instead
when the installed driver/toolkit combination requires CUDA 12.

## Supported combinations

| Solver | CPU | CUDA | Notes |
| --- | --- | --- | --- |
| `cd` | Yes | No | Coordinate descent. |
| `fista` | Yes | Yes | CV paths and final refit use the selected backend. |
| `hybrid` | Yes | Yes | CUDA CV paths followed by a CPU coordinate-descent refit. |
| `saga`, `liblinear` | Yes | No | CPU compatibility aliases. |

The CUDA implementation currently supports `dtype="float64"`. Standard and
adaptive L1 are available with `cd`, `fista`, and `hybrid`; logical polishing
is a separate CPU phase.

## Fit on CUDA

Use `solver="fista"` to run the CV paths and final refit on CUDA:

```python
from cutlass import CutlassLogisticCV

model = CutlassLogisticCV(
    Cs=15,
    cv=5,
    solver="fista",
    penalty="l1",
    backend="cuda",
    device=0,
    dtype="float64",
    allow_cpu_fallback=False,
    verbose=False,
)
model.fit(X, y)

print(model.C_)
print(model.backend_report_)
```

The public coefficients, intercept, CV diagnostics, adaptive-L1 attributes, and
predictions remain NumPy arrays. The lower-level estimator accepts either NumPy
or CuPy input arrays; device inputs avoid the initial host-to-device transfer.

The high-level `CutlassClassifier` accepts the same `backend`, `device`,
`dtype`, and `allow_cpu_fallback` arguments and forwards them to its internal
`CutlassLogisticCV` estimator.

## Hybrid sparse refit

`solver="hybrid"` runs FISTA cross-validation on CUDA and performs the selected
final refit with the existing CPU coordinate-descent solver:

```python
model = CutlassLogisticCV(
    Cs=15,
    cv=5,
    solver="hybrid",
    penalty="adaptive_l1",
    backend="cuda",
    allow_cpu_fallback=True,
)
model.fit(X, y)
```

The current `cd`, `saga`, and `liblinear` solver names are CPU-only. CUTLASS does
not silently reinterpret them as FISTA.

## Automatic selection

`backend="auto"` uses CUDA only for supported fits above a conservative work
threshold with sufficient device memory. The decision is available in
`auto_decision_` and `backend_report_`:

```python
model = CutlassLogisticCV(solver="fista", backend="auto", verbose=False)
model.fit(X, y)
print(model.backend_used_)
print(model.auto_decision_)
```

Policy version 1 uses a default threshold of 75,000,000 work units, where work
is approximately `rows * features * folds * C values` (doubled for adaptive
L1). The fitted diagnostics record the threshold used.

The threshold can be overridden for benchmarking with the
`CUTLASS_CUDA_AUTO_MIN_WORK` environment variable. Services that can see a
larger queue of compatible fits should make an application-level Auto decision
and submit an explicit backend to each fit.

## Progress and cancellation

Hooks are passed to `fit()` so they are not estimator parameters:

```python
cancelled = False


def on_progress(event):
    print(event["phase"], event["completed"], event["total"])


def should_cancel():
    return cancelled


model.fit(
    X,
    y,
    progress_callback=on_progress,
    cancel_callback=should_cancel,
)
```

Cancellation raises `cutlass.FitCancelledError` and never triggers CPU
fallback. Callback exceptions also propagate to the caller.

## Fallback and diagnostics

An explicit CUDA request with fallback enabled emits `CutlassBackendWarning`
when it restarts on CPU. Inspect:

```python
model.backend_requested_
model.backend_used_
model.fallback_reason_
model.fit_timings_
model.peak_device_memory_bytes_
model.runtime_versions_
model.backend_report_
```

Additional convenience attributes include `backend_provider_`, `device_id_`,
`device_name_`, `dtype_`, `n_jobs_effective_`, and `auto_decision_`. The
JSON-safe report includes solver/CV configuration, synchronized phase timings,
runtime versions, transfer bytes, synchronization count, and peak observed
device allocation/reservation.

Fallback restarts the entire fit. CUTLASS never combines partial CUDA and CPU CV
results. `allow_cpu_fallback=False` instead raises a typed backend exception.

## Persistent services

CUDA fitting stays in the calling process and does not start the CPU fold pool.
A service should use one long-lived worker process per GPU and submit CUTLASS
fits to that worker through a bounded application queue. CUTLASS deliberately
does not create a daemon, HTTP protocol, or application-specific scheduler.

## Current boundaries

- CUDA folds run sequentially and `n_jobs_effective_` is one.
- Coordinate descent, logical polishing, and hybrid final refits stay on CPU.
- Prediction is CPU-based because fitted public state is normalized to NumPy.
- There is no multi-model `fit_many` scheduler or concurrent-stream API.
- Reported phase timings are synchronized wall-clock measurements; they include
  the relevant orchestration and transfer boundaries rather than claiming to be
  isolated kernel benchmarks.
