# Optional GPU Backend for CUTLASS

## Purpose

This document defines a package-level GPU execution mode for CUTLASS. The design
supports SignalForge's Progressive Parameter Search while remaining suitable for
notebooks, batch research, services, command-line tools, and other applications
that use `CutlassLogisticCV` or `CutlassClassifier`.

The target is GPU-accelerated model fitting, not a GPU-only library. Dataframe
handling, rectification, duplicate consolidation, logical polishing,
serialization, orchestration, and most application-specific work remain on the
CPU unless profiling later justifies moving them. The expensive FISTA and
adaptive-L1 ridge computations move to CUDA through an optional CuPy provider.

The implementation must preserve:

- the existing NumPy path as the reference implementation;
- the current default behavior for callers that do not request a GPU;
- `CutlassLogisticCV` C-grid, fold, warm-start, CV-rule, adaptive-L1, and fitted
  attribute semantics;
- `CutlassClassifier` preprocessing and coefficient-expansion semantics;
- NumPy fitted attributes and predictions by default;
- reproducible model selection within documented floating-point tolerances;
- a structured, observable CPU fallback rather than a silent change of backend;
- a lightweight base installation with no mandatory CUDA dependency.

## Executive decision

Add `cpu`, `cuda`, and `auto` backends to the estimator APIs and implement CUDA
with CuPy. Keep `backend="cpu"` as the initial default so upgrading CUTLASS does
not change the numerical implementation used by existing research. Applications
that want automatic selection, including SignalForge, opt in with
`backend="auto"`.

Port FISTA and the adaptive-L1 ridge pilot first. Preserve the existing CPU
coordinate-descent implementation. The initial CUDA release supports:

| Solver | CPU backend | CUDA backend | Initial behavior |
|---|---:|---:|---|
| `fista` | Yes | Yes | CV and final refit run on the selected backend. |
| `hybrid` | Yes | Yes | FISTA CV runs on the selected backend; final coordinate-descent refit runs on CPU. |
| `cd` | Yes | No | `auto` selects CPU; explicit `cuda` falls back or raises according to policy. |
| `saga` / `liblinear` aliases | CPU `cd` | No | Same behavior as `cd`; aliases do not imply a CUDA implementation. |

This solver boundary is intentional. The current coordinate-descent algorithm
updates one feature at a time and refreshes logits between updates. Issuing CUDA
kernels from that Python loop is unlikely to be competitive, and changing to a
parallel approximation would change the optimizer. A fused or otherwise
semantically equivalent CUDA coordinate-descent solver can be evaluated later.

For SignalForge, the first model-fitting integration should request
`solver="hybrid"`, `backend="cuda"` or `"auto"`, and `dtype="float64"`. This
retains the existing hybrid contract already recognized by CUTLASS: FISTA for
the CV path and coordinate descent for the sparse final refit.

## Scope and responsibility boundary

CUTLASS provides reusable numerical capabilities. It must not embed a Flask
server, browser protocol, market-data cache, trading replay engine, or
application-specific multiprocessing topology.

| Concern | CUTLASS owns | Host application owns |
|---|---|---|
| Backend names and validation | Yes | Maps UI/config values into them |
| CUDA discovery and health probe | Yes | Exposes health through its own UI/API if needed |
| Host/device conversion | Yes | Supplies input arrays and retains source data |
| FISTA, ridge pilot, CV path, final fit | Yes | Chooses model configuration |
| Fold/C-path scheduling inside one fit | Yes | Schedules independent model fits or jobs |
| Progress and cancellation hooks | Defines and invokes | Translates hooks into queues, HTTP events, or UI updates |
| Device diagnostics and timing | Produces | Persists or displays them |
| CUDA worker process lifecycle | No | Owns process, queue, restart, and service lifetime |
| Candidate batching and feature cache | No | Groups application jobs by reusable data |
| SignalForge replay optimization | No | Implements domain-specific NumPy/CUDA replay |

This split lets SignalForge run one persistent CUDA-owning worker process while a
notebook can simply call `fit()`. CuPy reuses its CUDA context and memory pool for
subsequent fits in the same process, so CUTLASS does not need to create a daemon
or global job queue.

## Current CUTLASS architecture

The package is currently version 0.5.0 and depends only on NumPy and pandas.
Relevant files are:

- [`linear_model.py`](../src/cutlass/linear_model.py):
  `CutlassLogisticCV`, fold workers, adaptive weighting, CV aggregation, final
  refit, and logical polish;
- [`_solvers.py`](../src/cutlass/_solvers.py): CPU coordinate descent, FISTA,
  and the ridge pilot;
- [`_math.py`](../src/cutlass/_math.py): sigmoid, softplus, log loss, and soft
  thresholding;
- [`model.py`](../src/cutlass/model.py): `CutlassClassifier`, CPU preprocessing,
  estimator construction, coefficient expansion, and prediction;
- [`serialization.py`](../src/cutlass/serialization.py): backend-neutral NPZ
  persistence;
- [`pyproject.toml`](../pyproject.toml): required and optional dependencies.

`CutlassLogisticCV.fit()` currently:

1. converts `X` to a float64 Fortran-order NumPy array and `y` to a NumPy array;
2. creates deterministic folds with NumPy's seeded generator;
3. places `X` and `y` in CPU shared memory;
4. starts at most one process per fold;
5. walks each fold's ordered C path sequentially with warm starts;
6. aggregates validation losses and selects C with `min` or `1se`;
7. refits on all data and optionally performs logical polish;
8. returns NumPy coefficients, intercepts, and probabilities.

The CUDA path must dispatch before CPU shared memory or a process pool is
created. CUDA contexts must not be created independently in the existing fold
workers.

## Public API contract

### Estimator parameters

Add the same execution parameters to `CutlassLogisticCV` and
`CutlassClassifier`:

```python
CutlassLogisticCV(
    ...,
    backend="cpu",              # "cpu", "cuda", or "auto"
    device=None,                # None or a zero-based CUDA device id
    dtype="float64",            # FP64 in the first release
    allow_cpu_fallback=True,
)
```

The high-level classifier forwards these parameters unchanged to its fitted
`classifier_`. They must appear in `CutlassClassifier.get_params()` and work
with `set_params()`.

Parameter semantics:

- `backend="cpu"` always uses the existing NumPy implementation.
- `backend="cuda"` requests CUDA. Unsupported solver configurations and runtime
  failures either restart the whole fit on CPU or raise, according to
  `allow_cpu_fallback`.
- `backend="auto"` probes CUDA lazily and selects it only for a supported,
  sufficiently large workload. Choosing CPU during normal auto selection is not
  a fallback.
- `device=None` means device 0 for an explicit CUDA request and the first healthy
  eligible device for `auto`. An integer selects that exact device.
- `dtype="float64"` is the only parity-qualified CUDA precision in the first
  release. A later `float32` fast mode requires separate tests and must never be
  enabled implicitly.
- `allow_cpu_fallback=True` permits one clean restart on the CPU. CUTLASS never
  resumes halfway through a failed CUDA path or combines partial GPU and CPU CV
  losses.

Invalid backend names, devices, and dtypes fail validation before work begins.
The `n_jobs` parameter continues to control the CPU fold pool. It does not create
CUDA-owning processes; `n_jobs_effective_` is `1` for the CUDA portion of a fit.

Typical consumers use the same estimator surface:

```python
from cutlass import CutlassClassifier, CutlassLogisticCV

# Existing scripts remain on the NumPy reference path.
cpu_model = CutlassClassifier()

# A notebook can let CUTLASS choose for this one fit.
notebook_model = CutlassLogisticCV(
    Cs=15,
    cv=5,
    solver="fista",
    backend="auto",
)

# A service that has already reserved device 0 can require CUDA explicitly.
service_model = CutlassLogisticCV(
    Cs=15,
    cv=5,
    solver="hybrid",
    backend="cuda",
    device=0,
    allow_cpu_fallback=False,
)
```

### Why the default is `cpu`

Changing the default to `auto` would make installing an optional dependency
change the optimizer used by an otherwise unchanged research script. The first
GPU-capable release therefore defaults to `cpu`. Applications can deliberately
choose `auto`, and a future major release can reconsider the default after the
selection heuristic and parity suite have matured.

### Fitted diagnostics

Every completed fit, including a CPU-only fit, populates:

```python
model.backend_requested_              # "cpu", "cuda", or "auto"
model.backend_used_                   # "cpu" or "cuda"
model.backend_provider_               # "numpy" or "cupy"
model.device_id_                      # int or None
model.device_name_                    # str or None
model.dtype_                          # canonical dtype name
model.n_jobs_effective_               # actual CUTLASS execution parallelism
model.fallback_reason_                # structured dict or None
model.auto_decision_                  # structured dict or None
model.fit_timings_                    # JSON-compatible dict of seconds
model.peak_device_memory_bytes_       # int or None
model.runtime_versions_               # JSON-compatible version information
```

Also expose a single serialization-friendly summary:

```python
model.backend_report_
```

`backend_report_` contains the values above plus input shape, C count, fold
count, penalty, solver, transfer bytes, and synchronization count. It must not
contain CuPy objects, exception objects, local paths, or non-serializable device
handles.

An example report is:

```python
{
    "requested": "auto",
    "used": "cuda",
    "provider": "cupy",
    "decision_reason": "supported_batched_cv_work",
    "fallback": None,
    "device": {"id": 0, "name": "NVIDIA GeForce RTX 3090"},
    "dtype": "float64",
    "shape": {"rows": 12000, "features": 180},
    "cv": 3,
    "c_values": 15,
    "timings_seconds": {
        "backend_resolution": 0.004,
        "host_to_device": 0.012,
        "adaptive_pilot": 0.083,
        "cv_path": 0.214,
        "device_to_host": 0.002,
        "final_refit": 0.031,
        "logical_polish_cpu": 0.0,
        "total": 0.351,
    },
    "peak_device_memory_bytes": 241172480,
}
```

Exact keys should be stabilized in tests before release. New diagnostic keys may
be added in minor releases, but existing keys should not change meaning.

### Fallback visibility and exceptions

Fallback must not be silent. When an explicit CUDA request falls back, CUTLASS:

1. emits a `CutlassBackendWarning` once for the fit;
2. sets `backend_used_="cpu"`;
3. stores a machine-readable code and safe message in `fallback_reason_`; and
4. includes the failed phase and provider in `backend_report_`.

Define public exceptions in `cutlass.exceptions`:

- `BackendUnavailableError`: dependency, driver, or device is unavailable;
- `BackendConfigurationError`: requested solver/dtype/device combination is not
  supported;
- `BackendExecutionError`: CUDA work failed and fallback is disabled;
- `FitCancelledError`: cooperative cancellation was requested.

Do not treat cancellation as a backend failure and do not fall back to CPU after
cancellation.

### Progress and cancellation

Add optional fit-time hooks rather than estimator constructor parameters, so
callbacks do not affect estimator cloning or parameter serialization:

```python
model.fit(
    X,
    y,
    progress_callback=None,
    cancel_callback=None,
)
```

`progress_callback` receives a JSON-compatible `FitProgress` object or dict with
at least:

```text
phase, completed, total, backend, fold, c_index, message, elapsed_seconds
```

`cancel_callback` is a zero-argument callable returning a boolean. CUTLASS checks
it at safe boundaries: before transfer, between fold paths, between C values,
during long solver iterations at documented checkpoints, and before final
refit. Callbacks run synchronously in the process and thread that called `fit()`.
CUTLASS does not send HTTP events or inspect application queue objects.

The CUDA path can report between C values because it remains in the calling
process. The unchanged parallel CPU path may initially report only at fold
boundaries; finer CPU progress would require a new parent/worker message
protocol. A callback exception aborts the fit after cleanup and propagates to
the caller. It is not a CUDA failure and must not trigger CPU fallback.

For SignalForge, the persistent CUDA worker translates queue cancellation state
into `cancel_callback` and translates progress records into its NDJSON progress
stream.

### Input and output arrays

The compatibility contract remains:

- pandas and NumPy inputs are accepted;
- `coef_`, `intercept_`, `Cs_`, adaptive-L1 attributes, and default predictions
  are NumPy arrays;
- a serialized fitted model can be loaded and used on a CPU-only machine.

`CutlassLogisticCV` may additionally accept a CuPy or DLPack-compatible device
array when `backend="cuda"`. Same-device, compatible-dtype input should avoid a
host-to-device copy where practical, but zero-copy behavior is an optimization,
not part of the correctness contract. The
[CuPy interoperability guide](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)
documents DLPack exchange and stream-safety responsibilities.
`CutlassClassifier` continues to perform its pandas-oriented preprocessing on
CPU in the first release.

A future `output_type="backend"` option can expose device predictions to
GPU-native pipelines. It is not required by SignalForge and should not delay the
initial implementation.

## Public backend discovery

Add a lightweight public module:

```python
from cutlass.acceleration import list_devices, probe_backend

devices = list_devices("cuda")
status = probe_backend("cuda", device=0)
print(status.to_dict())
```

`list_devices()` performs lazy provider import and enumeration. `probe_backend()`
performs a real health check:

1. import the optional provider;
2. select the requested device;
3. allocate small float64 arrays;
4. run an elementwise operation and matrix multiplication;
5. synchronize;
6. verify the result on the host;
7. report free and total device memory; and
8. return a structured status without crashing CPU-only callers.

The result includes availability, provider, device id/name, compute capability,
free/total memory, driver/runtime/provider versions, supported dtypes, and a
stable error code. Safe messages must not include stack traces or local
environment paths.

Importing `cutlass` or constructing a CPU estimator must never import CuPy or
initialize CUDA. Only discovery, a CUDA/auto fit, or an explicit device-array
operation may do so.

## Internal architecture

### File layout

Add or refactor toward this layout:

```text
src/cutlass/
    acceleration.py       # public discovery/status dataclasses
    exceptions.py         # public backend exceptions and warning
    _backend.py            # private backend resolution and common protocol
    _cuda_backend.py       # lazy CuPy adapter, events, memory diagnostics
    _cuda_solvers.py       # _CuPyFISTALogistic and _CuPyRidgeLogistic
    _solvers.py            # existing NumPy solvers, initially unchanged
    linear_model.py        # CPU/CUDA CV dispatch and public estimator
    model.py               # high-level parameter forwarding
    serialization.py       # backend-neutral model plus provenance metadata
```

Keep CUDA-specific imports in `_cuda_backend.py` and `_cuda_solvers.py`. Do not
place `import cupy` at module scope in `__init__.py`, `model.py`,
`linear_model.py`, or `_solvers.py`.

### Backend protocol

The private backend adapter provides a small capability-oriented interface:

```text
name
provider
device_id
xp
asarray(...)
to_numpy(...)
synchronize()
event_timer()
memory_info()
pool_info()
supports(solver, penalty, dtype)
```

Do not attempt a mechanical global replacement of `np` with `xp`. NumPy scalar
conversion is cheap, while converting a CuPy scalar to Python forces device
synchronization. CUDA solvers need explicit control of convergence reductions,
backtracking decisions, transfers, and event timing.

### Preserve the CPU reference path

Refactor `CutlassLogisticCV.fit()` into a thin validator/dispatcher and two
private paths:

```text
fit()
  -> validate shared estimator semantics
  -> resolve backend and capabilities
  -> _fit_cpu(...)   # current implementation moved with minimal changes
  -> _fit_cuda(...)  # new implementation
  -> CPU logical polish when requested
  -> normalize public fitted attributes and diagnostics
```

The first refactor commit should move the current implementation without
changing numerical statements or process-pool behavior. Existing CPU tests and
new golden fixtures must pass before CUDA code is introduced.

### CUDA FISTA

Implement `_CuPyFISTALogistic` by matching `_FISTALogistic`'s mathematical
operations and state:

- `X`, `y`, weights, intercept, momentum state, logits, and gradients remain on
  device;
- matrix-vector products use CuPy operations;
- sigmoid clipping remains `[-40, 40]`;
- soft thresholding uses the same definition;
- the penalty remains `lambda = 1 / (C * n_train)`;
- the ordered C path and warm-start state remain on device within each fold;
- the power-iteration start vector is generated deterministically on CPU and
  transferred, avoiding provider-specific random-number differences;
- backtracking and restart conditions preserve the CPU inequalities;
- results return to host only after a fold path or final fit is complete.

The correctness implementation may synchronize on each convergence or
backtracking decision. Optimize synchronization only after parity is established.
If convergence checks are later performed in chunks, the changed stopping
behavior must be measured and documented.

### CUDA adaptive-L1 ridge pilot

Implement `_CuPyRidgeLogistic` after CUDA FISTA. For every CV fold:

1. fit the ridge pilot on device;
2. compute `abs(beta_pilot) + adaptive_eps` on device;
3. apply the feature scales on device;
4. run the warm-started FISTA C path on the weighted design;
5. calculate validation logits and log loss on device.

For the final full-data fit, retain the meanings and shapes of:

- `adaptive_feature_scales_`;
- `adaptive_penalty_weights_`;
- `adaptive_pilot_coef_`;
- `adaptive_pilot_intercept_`;
- `adaptive_weighted_coef_`.

These public attributes are copied to NumPy once the fit is complete.

### CV scheduling and memory

Each fold's C path is sequential because it relies on warm starts. Independent
folds may run concurrently, but concurrency must be bounded by memory and
benchmark results.

Implement in stages:

1. single CUDA stream, folds evaluated sequentially, for simplest parity;
2. bounded streams for independent folds when this improves measured throughput;
3. optional batched fold state for compatible shapes if streams remain
   underutilized;
4. a general `fit_many` API only if multiple applications demonstrate a need.

Transfer the full design matrix once per fit. Avoid materializing every fold's
training matrix simultaneously when indexed views or bounded staging suffice.
Before launching, estimate device memory for the design, fold indices, solver
state, validation state, adaptive weighted data, and working margin. `auto`
chooses CPU if the estimate is unsafe. Explicit CUDA either falls back or raises.

CuPy caches released allocations in device and pinned-memory pools. Report both
live bytes and reserved pool bytes. Do not call `free_all_blocks()` after every
successful fit because doing so defeats reuse. Clear unused blocks after an
out-of-memory failure, on an explicit application request, or during controlled
worker shutdown.

### Final refit and logical polish

For `solver="fista"`, the final refit stays on CUDA. For `solver="hybrid"`, copy
the selected full-data representation to CPU once and run the existing
`_CDLogistic` final refit. Logical polish remains on CPU for all backends because
it is relatively small, includes Python control flow, and may create matplotlib
objects.

The report separates `final_refit_gpu`, `final_refit_cpu`, and
`logical_polish_cpu` timing so applications can see hybrid costs.

### Prediction behavior

The initial public `predict_proba()` continues to return NumPy. A CUDA-fitted
model is backend-neutral after fit because its public coefficients are NumPy.
Prediction may therefore default to NumPy even when fitting used CUDA; this keeps
short inference calls from paying context and transfer overhead.

An explicit future inference backend can be added independently. Training
backend and inference backend must not be conflated in persisted artifacts.

## Auto backend selection

The CUTLASS heuristic evaluates only the work visible to one estimator:

```text
base_work = rows * features * folds * C_values
adaptive_multiplier = 2 when penalty == "adaptive_l1" else 1
estimated_work = base_work * adaptive_multiplier
```

This estimate is an input to a calibrated decision, not a universal performance
formula. The decision also considers:

- CUDA health and requested device;
- solver/backend capability;
- dtype support;
- estimated free device memory after a safety margin;
- host-to-device transfer bytes;
- cold-context and kernel-compilation state where observable;
- benchmark-derived crossover thresholds for the installed provider/device
  class.

`auto` selects CPU when CUDA is unavailable, the solver is unsupported, memory
is unsafe, or the fit is below the measured crossover. It selects CUDA only when
the configuration is supported and the expected work can amortize launch and
transfer overhead.

Applications can know more than a single estimator. SignalForge can force CUDA
for a queue of compatible weekly fits even when each fit would independently
fall below CUTLASS's auto threshold. Conversely, an interactive application can
force CPU to minimize latency. The library always records its local decision
inputs in `auto_decision_`.

SignalForge's `Auto` selection should therefore be resolved once at the search
planning level using the full candidate/week/direction workload and CUTLASS's
health probe. It then submits an explicit `cpu` or `cuda` backend to each model
job and persists both the user-requested value (`auto`) and the resolved value.
Passing `auto` independently to every small weekly fit would discard information
known by the search planner and can underuse the GPU.

Do not ship a hard-coded RTX 3090 threshold as a universal rule. Establish
conservative defaults from a benchmark matrix, allow thresholds to evolve by
package version, and record the threshold/version used for research provenance.

## Packaging and installation

Keep the base package unchanged:

```bash
pip install cutlass
```

Because CuPy wheel names depend on the CUDA major family, provide explicit
extras rather than an ambiguous `gpu` extra:

```toml
[project.optional-dependencies]
plots = ["matplotlib>=3.5"]
cuda12 = ["cupy-cuda12x[ctk]>=14,<15"]
cuda13 = ["cupy-cuda13x[ctk]>=14,<15"]
```

The exact bounds must be confirmed in the release test matrix. Do not install
more than one CuPy distribution in the same environment. CuPy publishes Windows
and Linux wheels for CUDA 12 and CUDA 13, and its CUDA component-wheel option can
provide the runtime pieces while still requiring a compatible NVIDIA driver.
See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

Example SignalForge environment setup for the observed CUDA 13-compatible
workstation:

```powershell
conda create --name sforge-gpu --clone sforge
conda activate sforge-gpu
python -m pip install "cutlass[cuda13]>=0.6,<0.7"
python -c "from cutlass.acceleration import probe_backend; print(probe_backend('cuda', device=0).to_dict())"
```

GPU support may require a newer minimum Python than the base package because the
optional provider has its own Python support window. CPU-only support can remain
broader. Publish and test the exact Python/CUDA/CuPy matrix for every feature
release.

## Timing and memory diagnostics

GPU operations are asynchronous. Use CUDA events around device work and
synchronize the ending event before reporting elapsed GPU time. Use
`perf_counter()` for caller-observed phases such as validation, import,
transfers, CPU fallback, CPU final refit, and total wall time. Do not report an
unsynchronized Python call duration as kernel time. CuPy's
[performance guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html)
describes the required event and warm-up behavior.

Record cold and warm timings separately in benchmarks because initial context
creation and kernel compilation can dominate small fits. Report at least:

- backend resolution and probe;
- input validation and conversion;
- host-to-device and device-to-host transfer;
- fold preparation;
- adaptive pilot;
- CV solver kernels;
- CV loss aggregation and C selection;
- final GPU and CPU refit;
- logical polish;
- total fit;
- live and reserved peak device bytes.

CuPy's default memory pools retain freed allocations for reuse, so reserved
memory can remain visible after arrays go out of scope. This is expected and
must not automatically be reported as a leak. Use pool statistics and repeated
fit tests to distinguish live growth from cached capacity. See the
[CuPy memory-management guide](https://docs.cupy.dev/en/stable/user_guide/memory.html).

## Failure recovery

Handle failures at the whole-fit boundary:

| Failure | `allow_cpu_fallback=True` | `allow_cpu_fallback=False` |
|---|---|---|
| CuPy not installed | Warn and restart CPU | `BackendUnavailableError` |
| No CUDA device / invalid id | Warn and restart CPU | `BackendUnavailableError` |
| Unsupported solver or dtype | Warn and restart CPU | `BackendConfigurationError` |
| Memory preflight fails | Warn and restart CPU | `BackendExecutionError` |
| CUDA out of memory | Synchronize, release references, clear unused pool blocks, restart CPU | `BackendExecutionError` |
| CUDA runtime/kernel error | Release safe state and restart CPU only if the process remains healthy | `BackendExecutionError` |
| Cancellation | Raise `FitCancelledError` | Raise `FitCancelledError` |

The package should not attempt to reset a corrupted CUDA context. It reports that
the provider process should be restarted. A service such as SignalForge owns
that restart. A notebook receives the typed error and can restart its kernel if
needed.

Never retain fold losses, coefficients, or a selected C from a failed CUDA
attempt when restarting on CPU. Reset fitted attributes and run the reference
path from the original validated input.

## Serialization and reproducibility

Fitted model state remains backend-neutral:

- coefficients, intercept, classes, C grid, selected C, scaler state, and
  adaptive-L1 attributes are NumPy values;
- loading and CPU inference never require CuPy;
- old NPZ files remain readable;
- new files may add backend provenance without making it required for inference.

Extend `save_classifier_npz()` with optional fields for requested/used backend,
provider, dtype, device description, provider/runtime versions, fallback code,
and timing JSON. The existing loader can ignore unknown fields. If a richer
model loader is later added, missing provenance in older artifacts means
`"unknown"`, not `"cpu"`.

Research artifacts should retain:

- CUTLASS version and source commit when available;
- requested and actual backend;
- solver, penalty, dtype, C grid, CV rule, and random seed;
- GPU model, compute capability, driver/runtime, and CuPy versions;
- auto-decision inputs and policy version;
- fallback code;
- phase timings and peak memory;
- parity mode or tolerance profile used by the consuming application.

## SignalForge integration contract

SignalForge can map its search configuration directly to the reusable estimator
API:

```python
model = cutlass.CutlassLogisticCV(
    Cs=config.Cs,
    cv=cv,
    max_iter=config.max_iter,
    n_jobs=config.n_jobs,
    tol=config.tol,
    random_state=config.random_state,
    cv_rule=config.cv_rule,
    zero_clamp=config.zero_clamp,
    penalty=config.penalty,
    adaptive_eps=config.adaptive_eps,
    adaptive_pilot_C=config.adaptive_pilot_C,
    solver="hybrid",
    backend=resolved_backend,              # "cpu" or "cuda" after search planning
    device=request.gpu_device,
    dtype=request.gpu_precision,
    allow_cpu_fallback=request.allow_cpu_fallback,
)
```

SignalForge-specific requirements map as follows:

| SignalForge requirement | CUTLASS facility | SignalForge adapter work |
|---|---|---|
| CPU / CUDA / Auto selector | `backend` and `probe_backend()` | Resolve Auto once from the full search workload, then submit explicit per-fit backends |
| RTX 3090/device display | `probe_backend()` | Expose through authenticated health endpoint |
| One CUDA owner | CUDA fit uses no fold process pool | Run calls in one long-lived spawned worker process |
| L1 and adaptive L1 | CUDA FISTA and ridge pilot | Pass existing penalty configuration |
| Sparse final refit | `solver="hybrid"` | Make solver explicit in run artifacts |
| Best-so-far progress | `progress_callback` | Forward structured events through worker/result queues |
| Cancellation | `cancel_callback` | Connect to per-search cancellation state |
| CPU fallback | policy, warning, diagnostics | Honor UI checkbox and show fallback reason |
| Artifact provenance | `backend_report_` | Persist report with candidate artifacts |
| Candidate/week batching | Outside estimator | Plan and queue compatible model jobs |
| Buy/Sell/cooldown replay | Outside CUTLASS | Keep CPU/browser fallback and add a separate backend CUDA kernel |

SignalForge currently creates `CutlassLogisticCV` without a solver argument, so
it receives CUTLASS's default `cd`. GPU integration must deliberately add
`solver="hybrid"`; CUTLASS must not reinterpret `cd` as FISTA merely because a
GPU was requested.

### Required SignalForge process rule

SignalForge's CUDA mode should have one long-lived process per selected GPU.
The worker initializes CUDA once and calls CUTLASS sequentially or through a
bounded scheduler. Outer CPU preparation may remain parallel, but an outer week
pool must not create one CUDA context per week while CUTLASS also schedules CV
work.

Recommended worker job types remain application-level concepts:

```text
HEALTH
FIT_MODEL
FIT_MODEL_BATCH
REPLAY_BATCH
CANCEL_SEARCH
CLEAR_CACHE
SHUTDOWN
```

CUTLASS implements only model fitting and health information behind these jobs.
This prevents trading-specific replay and queue behavior from leaking into the
library.

## Test plan

### CPU non-regression

Before adding CUDA tests:

1. capture golden C-path losses, selected C, coefficients, intercepts,
   probabilities, adaptive attributes, and logical-polish diagnostics for
   representative fixtures;
2. refactor CPU dispatch without numerical changes;
3. run the current adaptive-L1, coordinate-descent, duplicate consolidation,
   and logical-polish tests;
4. test that omitting backend arguments is identical to `backend="cpu"`;
5. test that importing and fitting CPU-only CUTLASS never imports CuPy.

No material CPU performance regression is allowed.

### Backend resolution without GPU hardware

Use an injected or monkeypatched private backend adapter to test:

- valid `cpu`, `cuda`, and `auto` resolution;
- unknown backend/dtype/device validation;
- provider import failure;
- no-device and unhealthy-device reports;
- solver capability checks;
- auto threshold and memory decisions;
- warning/exception behavior;
- full-fit restart and fitted-attribute cleanup;
- serializability of reports and progress events.

These tests run in normal CPU CI and must not require CuPy.

### CUDA parity

Mark hardware tests with `pytest.mark.cuda`. Compare CPU FISTA with CUDA FISTA,
and CPU hybrid with CUDA hybrid, for:

- standard L1 and adaptive L1;
- several row/feature aspect ratios;
- rectified `{-1, +1}` and continuous designs;
- imbalanced labels;
- integer and explicit C grids;
- `min` and `1se` rules;
- warm starts and deterministic fold indices;
- final coefficients, intercept, probabilities, and active support;
- all adaptive-L1 fitted attributes;
- high-level `CutlassClassifier` preprocessing and coefficient expansion;
- logical polish after a CUDA or hybrid fit;
- repeated fits in one process.

Initial FP64 gates for well-conditioned fixtures should target:

- finite values everywhere;
- CV mean losses within `rtol=1e-6`, `atol=1e-8`;
- maximum probability absolute difference no greater than `1e-6`;
- identical selected C when the best loss is separated from the runner-up by
  more than the loss tolerance;
- equivalent objective value and documented active-support agreement when
  correlated features make coefficients non-unique.

These values are provisional. Tighten or relax them only from measured evidence,
and version the accepted tolerance profile. Bitwise equality is not required
because GPU reductions may execute in a different order.

### Failure and lifecycle tests

On GPU hardware, test:

- invalid device;
- out-of-memory preflight and runtime OOM;
- cancellation before transfer, during CV, and before final refit;
- provider/kernel failure with and without fallback;
- callback exceptions;
- repeated CPU fallback without stale CUDA state;
- device memory live/reserved behavior over many fits;
- a fit after an OOM cleanup;
- model load and prediction in a separate CPU-only environment.

The SignalForge repository separately tests worker startup failure, browser
disconnect, Apply while search continues, multiple searches contending for one
GPU, server restart, replay parity, and stale queue cleanup.

### Performance tests

Benchmark identical CPU and CUDA configurations across:

- cold and warm process state;
- L1 and adaptive L1;
- FISTA and hybrid;
- small, medium, and large row/feature shapes;
- 2, 3, 5, and 10 folds;
- short and long C paths;
- NumPy host input and already-resident device input;
- one fit and repeated fits in one process.

Measure total wall time, transfer time, solver event time, synchronization count,
CPU utilization, GPU utilization, live/reserved peak device memory, and fits per
second. The release gate is measured benefit on representative broad workloads,
not a promised universal speedup. `auto` must choose CPU for benchmark cells
where CUDA is consistently slower.

## Implementation sequence

### Phase 1: Instrument and freeze CPU behavior

1. Add phase timings and expose current fold/C losses needed by parity tests.
2. Add deterministic golden fixtures for L1, adaptive L1, FISTA, hybrid, `min`,
   and `1se`.
3. Extract `_fit_cpu()` with the existing shared-memory/process-pool logic.
4. Confirm unchanged CPU results and performance.

### Phase 2: Public backend contract and probe

1. Add estimator parameters, fitted diagnostics, exceptions, and warning.
2. Add lazy backend resolution and capability checks.
3. Add `cutlass.acceleration` discovery and health APIs.
4. Add fake-backend CPU CI tests.
5. Forward parameters through `CutlassClassifier.get_params()` and
   `classifier_`.

At the end of this phase, `cuda` may report unavailable cleanly but must not yet
claim model-fitting support.

### Phase 3: CUDA FISTA

1. Add CuPy math helpers and `_CuPyFISTALogistic`.
2. Match the CPU FISTA update, backtracking, restart, and stopping logic.
3. Implement single-stream fold paths and warm starts.
4. Add CUDA event timing, transfer counts, and memory reporting.
5. Pass FP64 parity tests before optimizing streams or synchronization.

### Phase 4: Adaptive L1 and hybrid refit

1. Add `_CuPyRidgeLogistic`.
2. Keep fold and full-data feature scaling on device.
3. Populate all adaptive fitted attributes as NumPy.
4. Add the CPU coordinate-descent final stage for `hybrid`.
5. Run standard and adaptive parity matrices.

This phase supplies SignalForge's required CUTLASS model-fitting path.

### Phase 5: Auto selection, resilience, and callbacks

1. Build the representative CPU/CUDA benchmark matrix.
2. Implement a conservative, versioned auto heuristic.
3. Add memory preflight, whole-fit fallback, typed errors, and cleanup.
4. Add progress and cooperative cancellation checkpoints.
5. Test repeated fits in a persistent process.

### Phase 6: Packaging, documentation, and release

1. Add CUDA 12/13 optional extras without changing base dependencies.
2. Add a GPU usage vignette and API docstrings.
3. Extend serialization with optional backend provenance.
4. Add CPU CI and a separate NVIDIA GPU job where hardware is available.
5. Test source distribution and wheel in clean CPU-only and CUDA environments.
6. Release as a feature version, preferably CUTLASS 0.6.0.

### Phase 7: SignalForge integration and profiling

1. Update SignalForge's pin only after the CUTLASS parity gates pass.
2. Add backend fields to `SparseModelConfig` and result artifacts.
3. Make `solver="hybrid"` explicit for GPU-capable runs.
4. Connect health, progress, cancellation, diagnostics, and fallback.
5. Run the full Progressive Parameter Search benchmark and ranking-parity suite.
6. Decide from profiling whether CUTLASS needs a reusable `fit_many` API or
   whether SignalForge's bounded worker queue is sufficient.

## Release acceptance criteria

GPU mode is ready for a stable release only when:

- existing callers use the unchanged CPU path by default;
- base installation and import work without CuPy or an NVIDIA driver;
- explicit CUDA and auto decisions are observable and correctly reported;
- unsupported `cd` configurations never masquerade as CUDA;
- CPU versus CUDA FISTA/hybrid passes the approved FP64 parity profile;
- L1 and adaptive-L1 fitted attributes retain their definitions and shapes;
- model artifacts load and predict on a CPU-only machine;
- fallback restarts the entire fit and records the reason;
- cancellation does not trigger CPU fallback;
- repeated warm fits do not show unbounded live device-memory growth;
- broad representative workloads show a meaningful end-to-end benefit;
- `auto` avoids CUDA for workloads where measured overhead dominates;
- SignalForge preserves candidate ranking, best-so-far behavior, progress,
  cancellation, Apply semantics, and artifact provenance in its integration
  tests.

## Recommended first milestone

The first CUTLASS milestone should deliver a parity-qualified, optional CUDA
backend for `solver="fista"` and `solver="hybrid"`, including adaptive L1,
diagnostics, health probing, fallback, progress, and cancellation. It should not
attempt CUDA coordinate descent, application job queues, or trading replay.

That milestone is both useful beyond SignalForge and sufficient for SignalForge
to integrate GPU model fitting into a single persistent worker. Profiling the
integrated search then determines whether the next investment belongs in
multi-fit batching, a specialized coordinate-descent kernel, or the
application's replay optimizer.

## Final recommendation

Proceed with a CuPy-backed CUDA implementation in CUTLASS, but expose it through
hardware-oriented `cpu`, `cuda`, and `auto` contracts rather than
SignalForge-specific settings. Preserve CPU as the default and scientific
reference, implement FISTA and the adaptive ridge pilot natively on GPU, use the
existing CPU coordinate descent for the hybrid final refit, and make every
backend decision and fallback visible.

Keep the persistent worker and search batching in SignalForge. This gives
SignalForge safe single-owner GPU execution while leaving CUTLASS usable as a
normal estimator library in unrelated applications.
