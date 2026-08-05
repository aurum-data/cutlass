from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cutlass import (
    BackendConfigurationError,
    BackendUnavailableError,
    CutlassBackendWarning,
    CutlassClassifier,
    CutlassLogisticCV,
    FitCancelledError,
)
from cutlass.serialization import save_classifier_npz


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] - 0.6 * X[:, 1] + 0.1 * rng.normal(size=60) > 0).astype(int)
    return X, y


def test_cpu_is_default_and_populates_backend_diagnostics() -> None:
    X, y = _data()
    events: list[dict[str, object]] = []
    model = CutlassLogisticCV(
        Cs=[0.1, 1.0],
        cv=2,
        n_jobs=1,
        solver="fista",
        max_iter=200,
        verbose=False,
    )
    model.fit(X, y, progress_callback=events.append)

    assert model.backend_requested_ == "cpu"
    assert model.backend_used_ == "cpu"
    assert model.backend_provider_ == "numpy"
    assert model.fallback_reason_ is None
    assert model.backend_report_["shape"] == {"rows": 60, "features": 5}
    assert model.backend_report_["timings_seconds"]["total"] >= 0.0
    assert events[0]["phase"] == "input_validation"
    assert events[-1]["phase"] == "complete"


def test_explicit_cuda_cd_falls_back_visibly_without_importing_provider() -> None:
    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[1.0],
        cv=2,
        n_jobs=1,
        solver="cd",
        backend="cuda",
        allow_cpu_fallback=True,
        max_iter=100,
        verbose=False,
    )
    with pytest.warns(CutlassBackendWarning, match="does not support"):
        model.fit(X, y)

    assert model.backend_requested_ == "cuda"
    assert model.backend_used_ == "cpu"
    assert model.fallback_reason_["code"] == "unsupported_solver"


def test_explicit_cuda_cd_raises_when_fallback_is_disabled() -> None:
    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[1.0],
        cv=2,
        solver="cd",
        backend="cuda",
        allow_cpu_fallback=False,
        verbose=False,
    )
    with pytest.raises(BackendConfigurationError, match="does not support"):
        model.fit(X, y)


def test_unavailable_cuda_provider_falls_back_with_structured_reason(monkeypatch) -> None:
    X, y = _data()

    def unavailable_backend(*_args, **_kwargs):
        raise BackendUnavailableError("provider unavailable for test")

    monkeypatch.setattr("cutlass._cuda_backend.CudaBackend", unavailable_backend)
    model = CutlassLogisticCV(
        Cs=[1.0],
        cv=2,
        n_jobs=1,
        solver="fista",
        backend="cuda",
        allow_cpu_fallback=True,
        max_iter=100,
        verbose=False,
    )
    with pytest.warns(CutlassBackendWarning, match="unavailable"):
        model.fit(X, y)

    assert model.backend_used_ == "cpu"
    assert model.fallback_reason_["code"] == "cuda_unavailable"


def test_auto_uses_cpu_normally_for_unsupported_solver() -> None:
    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[1.0],
        cv=2,
        n_jobs=1,
        solver="cd",
        backend="auto",
        max_iter=100,
        verbose=False,
    )
    model.fit(X, y)

    assert model.backend_used_ == "cpu"
    assert model.fallback_reason_ is None
    assert model.auto_decision_["reason"] == "unsupported_solver"


def test_auto_keeps_small_supported_fit_on_cpu() -> None:
    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[0.1, 1.0],
        cv=2,
        n_jobs=1,
        solver="fista",
        backend="auto",
        max_iter=100,
        verbose=False,
    ).fit(X, y)

    assert model.backend_used_ == "cpu"
    assert model.auto_decision_["reason"] == "below_crossover"
    assert model.auto_decision_["threshold"] == 75_000_000


def test_cancellation_never_falls_back() -> None:
    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[1.0],
        cv=2,
        solver="fista",
        backend="cpu",
        verbose=False,
    )
    with pytest.raises(FitCancelledError):
        model.fit(X, y, cancel_callback=lambda: True)

    assert model.backend_used_ is None
    assert model.fallback_reason_ is None


def test_classifier_forwards_backend_and_serializes_provenance(tmp_path) -> None:
    X, y = _data()
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    model = CutlassClassifier(
        rectify=False,
        duplicate_mode="none",
        use_scaler=False,
        Cs=[0.1, 1.0],
        cv=2,
        solver="fista",
        backend="cpu",
        max_iter=200,
        verbose=False,
    )
    model.fit(frame, y)

    assert model.get_params()["backend"] == "cpu"
    assert model.backend_used_ == "cpu"
    assert model.classifier_.backend_report_ == model.backend_report_

    output = tmp_path / "model.npz"
    save_classifier_npz(model, frame.columns, output)
    with np.load(output, allow_pickle=True) as blob:
        report = json.loads(str(blob["lr.backend_report_json"][0]))
        assert report["used"] == "cpu"
        assert str(blob["lr.backend_used_"][0]) == "cpu"
