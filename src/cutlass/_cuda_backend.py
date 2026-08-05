"""Lazy CuPy adapter used only when CUDA discovery or fitting is requested."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Optional, TypeVar

import numpy as np

from .acceleration import BackendStatus
from .exceptions import BackendUnavailableError

T = TypeVar("T")


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise BackendUnavailableError(
            f"CuPy is unavailable ({type(exc).__name__})."
        ) from exc
    return cp


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _versions(cp) -> dict[str, str]:
    return {
        "cupy": str(cp.__version__),
        "cuda_runtime": str(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": str(cp.cuda.runtime.driverGetVersion()),
        "numpy": str(np.__version__),
    }


def _device_status(cp, device_id: int, *, run_probe: bool) -> BackendStatus:
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if device_id < 0 or device_id >= count:
            return BackendStatus(
                backend="cuda",
                available=False,
                provider="cupy",
                device_id=device_id,
                supported_dtypes=("float64",),
                versions=_versions(cp),
                error_code="invalid_device",
                error_message=f"CUDA device {device_id} is not available.",
            )
        with cp.cuda.Device(device_id):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            capability = cp.cuda.Device(device_id).compute_capability
            if isinstance(capability, bytes):
                capability = capability.decode("ascii", errors="replace")
            if run_probe:
                a = cp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float64)
                b = (a @ a) + 1.0
                cp.cuda.get_current_stream().synchronize()
                expected = np.array([[8.0, 11.0], [16.0, 23.0]])
                if not np.allclose(cp.asnumpy(b), expected):
                    raise RuntimeError("CUDA arithmetic health check returned an invalid result.")
            return BackendStatus(
                backend="cuda",
                available=True,
                provider="cupy",
                device_id=device_id,
                device_name=_decode_name(props["name"]),
                compute_capability=str(capability),
                free_memory_bytes=int(free_bytes),
                total_memory_bytes=int(total_bytes),
                supported_dtypes=("float64",),
                versions=_versions(cp),
            )
    except Exception as exc:
        return BackendStatus(
            backend="cuda",
            available=False,
            provider="cupy",
            device_id=device_id,
            supported_dtypes=("float64",),
            versions=_versions(cp),
            error_code="device_probe_failed",
            error_message=f"CUDA device probe failed ({type(exc).__name__}).",
        )


def cuda_device_statuses(*, probe: bool = False) -> list[BackendStatus]:
    cp = _load_cupy()
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return [
            BackendStatus(
                backend="cuda",
                available=False,
                provider="cupy",
                supported_dtypes=("float64",),
                versions={"cupy": str(cp.__version__)},
                error_code="device_enumeration_failed",
                error_message=f"CUDA enumeration failed ({type(exc).__name__}).",
            )
        ]
    if count == 0:
        return [
            BackendStatus(
                backend="cuda",
                available=False,
                provider="cupy",
                supported_dtypes=("float64",),
                versions=_versions(cp),
                error_code="no_device",
                error_message="No CUDA device is available.",
            )
        ]
    return [_device_status(cp, idx, run_probe=probe) for idx in range(count)]


def probe_cuda(device: Optional[int] = None) -> BackendStatus:
    cp = _load_cupy()
    device_id = 0 if device is None else int(device)
    return _device_status(cp, device_id, run_probe=True)


class CudaBackend:
    """Small stateful adapter for one CUDA device in the calling process."""

    name = "cuda"
    provider = "cupy"

    def __init__(self, device: Optional[int] = None) -> None:
        self.cp = _load_cupy()
        self.device_id = 0 if device is None else int(device)
        status = _device_status(self.cp, self.device_id, run_probe=True)
        if not status.available:
            raise BackendUnavailableError(status.error_message or "CUDA is unavailable.")
        self.status = status
        self.device = self.cp.cuda.Device(self.device_id)
        self.synchronization_count = 0
        self.peak_used_bytes = 0
        self.peak_reserved_bytes = 0
        self.observe_memory()

    @property
    def device_name(self) -> str:
        return self.status.device_name or f"CUDA device {self.device_id}"

    @property
    def versions(self) -> dict[str, str]:
        return dict(self.status.versions)

    def asarray(self, value: Any, *, dtype=None, order: Optional[str] = None):
        with self.device:
            kwargs = {"dtype": dtype}
            if order is not None:
                kwargs["order"] = order
            result = self.cp.asarray(value, **kwargs)
            self.observe_memory()
            return result

    def to_numpy(self, value: Any) -> np.ndarray:
        with self.device:
            result = self.cp.asnumpy(value)
            self.observe_memory()
            return result

    def is_device_array(self, value: Any) -> bool:
        return isinstance(value, self.cp.ndarray)

    def synchronize(self) -> None:
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()
        self.synchronization_count += 1
        self.observe_memory()

    def timed(self, operation: Callable[[], T]) -> tuple[T, float]:
        """Time device work with CUDA events and return elapsed seconds."""

        with self.device:
            start = self.cp.cuda.Event()
            end = self.cp.cuda.Event()
            start.record()
            result = operation()
            end.record()
            end.synchronize()
            self.synchronization_count += 1
            elapsed = float(self.cp.cuda.get_elapsed_time(start, end)) / 1000.0
            self.observe_memory()
            return result, elapsed

    def timed_host(self, operation: Callable[[], T]) -> tuple[T, float]:
        start = perf_counter()
        result = operation()
        self.synchronize()
        return result, perf_counter() - start

    def memory_info(self) -> dict[str, int]:
        with self.device:
            free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
            pool = self.cp.get_default_memory_pool()
            return {
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "used_bytes": int(pool.used_bytes()),
                "reserved_bytes": int(pool.total_bytes()),
            }

    def observe_memory(self) -> dict[str, int]:
        info = self.memory_info()
        self.peak_used_bytes = max(self.peak_used_bytes, info["used_bytes"])
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, info["reserved_bytes"])
        return info

    def clear_unused(self) -> None:
        with self.device:
            self.cp.get_default_memory_pool().free_all_blocks()
            self.cp.get_default_pinned_memory_pool().free_all_blocks()

    def is_out_of_memory(self, exc: BaseException) -> bool:
        return isinstance(exc, self.cp.cuda.memory.OutOfMemoryError)
