"""
CUTLASS package
---------------

Reusable implementation of the rectified L1 logistic regression workflow
developed in the accompanying research project.
"""

from __future__ import annotations

from ._version import __version__
from .acceleration import BackendStatus, FitProgress, list_devices, probe_backend
from .exceptions import (
    BackendConfigurationError,
    BackendExecutionError,
    BackendUnavailableError,
    CutlassBackendWarning,
    FitCancelledError,
)
from .metrics import calculate_youden_j, precision_recall_curve, roc_auc_score
from .model import CutlassClassifier
from .preprocessing import DuplicateColumnConsolidator, Rectifier, StandardScaler
from .linear_model import CutlassLogisticCV

__all__ = [
    "__version__",
    "BackendConfigurationError",
    "BackendExecutionError",
    "BackendStatus",
    "BackendUnavailableError",
    "CutlassClassifier",
    "CutlassBackendWarning",
    "CutlassLogisticCV",
    "DuplicateColumnConsolidator",
    "Rectifier",
    "StandardScaler",
    "FitCancelledError",
    "FitProgress",
    "calculate_youden_j",
    "list_devices",
    "precision_recall_curve",
    "probe_backend",
    "roc_auc_score",
]
