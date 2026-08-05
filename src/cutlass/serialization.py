"""Persistence helpers for CUTLASS models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .model import CutlassClassifier
from .preprocessing import Rectifier, StandardScaler

__all__ = [
    "save_limits_json",
    "load_limits_json",
    "save_classifier_npz",
    "load_classifier_npz",
]


def _nan_to_none(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, np.ndarray):
        return [_nan_to_none(v) for v in x.tolist()]
    return x


def _none_to_nan(x):
    if x is None:
        return float("nan")
    return x


def save_limits_json(limits: Mapping[str, Mapping[str, Tuple[float, float]]], path: str | Path) -> None:
    """Persist rectifier limits to a JSON file."""
    payload = {
        group: {feat: [_nan_to_none(lo), _nan_to_none(hi)] for feat, (lo, hi) in feats.items()}
        for group, feats in limits.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_limits_json(path: str | Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load rectifier limits stored by :func:`save_limits_json`."""
    raw = json.loads(Path(path).read_text())
    return {
        group: {feat: (float(_none_to_nan(lo)), float(_none_to_nan(hi))) for feat, (lo, hi) in feats.items()}
        for group, feats in raw.items()
    }


def save_classifier_npz(model: CutlassClassifier, feature_names: Sequence[str], path: str | Path) -> None:
    """
    Persist the fitted scaler + logistic model to a compressed npz file.
    Rectifier limits are not stored here; use :func:`save_limits_json`.
    """
    if not hasattr(model, "classifier_"):
        raise RuntimeError("Model must be fitted before calling save_classifier_npz.")

    state = {"feature_names": np.array(feature_names, dtype=object)}
    if model.scaler_ is not None:
        sc = model.scaler_
        state["scaler.with_mean"] = np.array([sc.with_mean], dtype=bool)
        state["scaler.with_std"] = np.array([sc.with_std], dtype=bool)
        state["scaler.mean_"] = sc.mean_.astype(np.float64)
        state["scaler.scale_"] = sc.scale_.astype(np.float64)

    lr = model.classifier_
    state["lr.coef_"] = lr.coef_.ravel().astype(np.float64)
    state["lr.intercept_"] = lr.intercept_.astype(np.float64)
    state["lr.classes_"] = lr.classes_.astype(int)
    if getattr(lr, "C_", None) is not None:
        state["lr.C_"] = np.array([float(lr.C_)], dtype=np.float64)
    if getattr(lr, "Cs_", None) is not None:
        state["lr.Cs_"] = np.array(lr.Cs_, dtype=np.float64)
    state["lr.solver"] = np.array([lr.solver], dtype=object)
    state["lr.penalty"] = np.array([lr.penalty], dtype=object)
    state["lr.cv_rule"] = np.array([lr.cv_rule], dtype=object)
    state["lr.zero_clamp"] = np.array([lr.zero_clamp], dtype=np.float64)
    if getattr(lr, "backend_report_", None):
        state["lr.backend_report_json"] = np.array(
            [json.dumps(lr.backend_report_, sort_keys=True)], dtype=object
        )
    for attribute in (
        "backend_requested_",
        "backend_used_",
        "backend_provider_",
        "device_name_",
        "dtype_",
    ):
        value = getattr(lr, attribute, None)
        if value is not None:
            state[f"lr.{attribute}"] = np.array([value], dtype=object)
    if getattr(lr, "device_id_", None) is not None:
        state["lr.device_id_"] = np.array([int(lr.device_id_)], dtype=np.int64)

    np.savez_compressed(path, **state)


def load_classifier_npz(path: str | Path) -> tuple[StandardScaler | None, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model coefficients and (optional) scaler parameters from disk.

    Returns
    -------
    scaler : StandardScaler or None
    coef : np.ndarray shape (p,)
    intercept : np.ndarray shape (1,)
    classes : np.ndarray shape (2,)
    """
    blob = np.load(path, allow_pickle=True)
    scaler = None
    if "scaler.mean_" in blob:
        scaler = StandardScaler(
            with_mean=bool(blob["scaler.with_mean"][0]),
            with_std=bool(blob["scaler.with_std"][0]),
        )
        scaler.mean_ = blob["scaler.mean_"].astype(np.float64)
        scaler.scale_ = blob["scaler.scale_"].astype(np.float64)
    coef = blob["lr.coef_"].astype(np.float64)
    intercept = blob["lr.intercept_"].astype(np.float64)
    classes = blob["lr.classes_"].astype(int)
    return scaler, coef, intercept, classes
