"""Exceptions and warnings for CUTLASS execution backends."""

from __future__ import annotations

__all__ = [
    "BackendConfigurationError",
    "BackendExecutionError",
    "BackendUnavailableError",
    "CutlassBackendWarning",
    "FitCancelledError",
]


class CutlassBackendError(RuntimeError):
    """Base class for execution-backend failures."""


class BackendUnavailableError(CutlassBackendError):
    """Raised when a requested execution backend or device is unavailable."""


class BackendConfigurationError(CutlassBackendError, ValueError):
    """Raised when a requested backend configuration is unsupported."""


class BackendExecutionError(CutlassBackendError):
    """Raised when backend execution fails and fallback is disabled."""


class FitCancelledError(CutlassBackendError):
    """Raised when cooperative cancellation is requested during a fit."""


class CutlassBackendWarning(RuntimeWarning):
    """Warns that a requested accelerator fell back to the CPU backend."""
