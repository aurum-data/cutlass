"""Public discovery helpers for optional CUTLASS execution backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["BackendStatus", "FitProgress", "list_devices", "probe_backend"]


@dataclass(frozen=True)
class BackendStatus:
    """Serializable result from backend discovery or a health probe."""

    backend: str
    available: bool
    provider: str
    device_id: Optional[int] = None
    device_name: Optional[str] = None
    compute_capability: Optional[str] = None
    free_memory_bytes: Optional[int] = None
    total_memory_bytes: Optional[int] = None
    supported_dtypes: tuple[str, ...] = ()
    versions: Dict[str, str] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        result = asdict(self)
        result["supported_dtypes"] = list(self.supported_dtypes)
        return result


@dataclass(frozen=True)
class FitProgress:
    """Progress record emitted synchronously by estimator ``fit`` methods."""

    phase: str
    completed: int
    total: int
    backend: str
    fold: Optional[int] = None
    c_index: Optional[int] = None
    message: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def _normalise_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    aliases = {"numpy": "cpu", "cupy": "cuda"}
    value = aliases.get(value, value)
    if value not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'.")
    return value


def list_devices(backend: str = "cuda") -> List[BackendStatus]:
    """List devices for a backend without importing CuPy for CPU discovery."""

    name = _normalise_backend(backend)
    if name == "cpu":
        return [
            BackendStatus(
                backend="cpu",
                available=True,
                provider="numpy",
                supported_dtypes=("float64",),
            )
        ]

    try:
        from ._cuda_backend import cuda_device_statuses

        return cuda_device_statuses(probe=False)
    except Exception as exc:
        return [
            BackendStatus(
                backend="cuda",
                available=False,
                provider="cupy",
                supported_dtypes=("float64",),
                error_code="provider_unavailable",
                error_message=f"CUDA discovery failed ({type(exc).__name__}).",
            )
        ]


def probe_backend(backend: str = "cuda", device: Optional[int] = None) -> BackendStatus:
    """Run a small allocation and calculation to validate a backend."""

    name = _normalise_backend(backend)
    if name == "cpu":
        return list_devices("cpu")[0]

    try:
        from ._cuda_backend import probe_cuda

        return probe_cuda(device=device)
    except Exception as exc:
        return BackendStatus(
            backend="cuda",
            available=False,
            provider="cupy",
            device_id=device,
            supported_dtypes=("float64",),
            error_code="provider_unavailable",
            error_message=f"CUDA probe failed ({type(exc).__name__}).",
        )
