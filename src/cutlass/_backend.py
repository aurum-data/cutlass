"""Private backend validation and selection helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .exceptions import BackendConfigurationError

CUDA_SOLVERS = {"fista", "hybrid"}


def normalise_backend(value: str) -> str:
    """Return a canonical public backend name."""

    backend = str(value).strip().lower()
    aliases = {"numpy": "cpu", "cupy": "cuda"}
    backend = aliases.get(backend, backend)
    if backend not in {"cpu", "cuda", "auto"}:
        raise BackendConfigurationError(
            "backend must be 'cpu', 'cuda', or 'auto'."
        )
    return backend


def normalise_dtype(value: Any) -> str:
    """Validate the parity-qualified numerical precision."""

    try:
        dtype = np.dtype(value).name
    except TypeError as exc:
        raise BackendConfigurationError("dtype must be 'float64'.") from exc
    if dtype != "float64":
        raise BackendConfigurationError(
            "Only dtype='float64' is supported by the first GPU release."
        )
    return dtype


def normalise_device(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise BackendConfigurationError("device must be None or a non-negative integer.")
    try:
        device = int(value)
    except (TypeError, ValueError) as exc:
        raise BackendConfigurationError(
            "device must be None or a non-negative integer."
        ) from exc
    if device < 0:
        raise BackendConfigurationError("device must be non-negative.")
    return device


def cuda_supports(solver: str, dtype: str) -> tuple[bool, Optional[str]]:
    """Return CUDA capability and a stable failure code."""

    if dtype != "float64":
        return False, "unsupported_dtype"
    if str(solver).lower() not in CUDA_SOLVERS:
        return False, "unsupported_solver"
    return True, None


def estimate_work(rows: int, features: int, folds: int, c_values: int, penalty: str) -> int:
    multiplier = 2 if str(penalty).lower() == "adaptive_l1" else 1
    return int(rows) * int(features) * int(folds) * int(c_values) * multiplier


def estimate_cuda_bytes(rows: int, features: int, folds: int, penalty: str) -> int:
    """Conservative admission estimate, including working buffers and margin."""

    matrix = int(rows) * int(features) * 8
    vectors = (int(rows) * 8 * 8) + (int(features) * 8 * 16)
    adaptive = matrix if str(penalty).lower() == "adaptive_l1" else 0
    fold_staging = matrix if int(folds) > 1 else matrix // 2
    return int((matrix + vectors + adaptive + fold_staging) * 1.35)
