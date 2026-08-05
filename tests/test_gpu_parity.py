from __future__ import annotations

import numpy as np
import pytest

from cutlass import CutlassLogisticCV, probe_backend


def _cuda_available() -> bool:
    return bool(probe_backend("cuda", device=0).available)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not _cuda_available(), reason="A healthy CUDA device is required."),
]


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(96, 7))
    logits = 1.4 * X[:, 0] - 0.8 * X[:, 1] + 0.35 * X[:, 2]
    y = (logits + 0.15 * rng.normal(size=len(X)) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("penalty", ["l1", "adaptive_l1"])
def test_cuda_fista_matches_cpu_fista(penalty: str) -> None:
    X, y = _data()
    kwargs = {
        "Cs": [0.05, 0.5, 5.0],
        "penalty": penalty,
        "cv": 3,
        "n_jobs": 1,
        "solver": "fista",
        "tol": 1e-5,
        "max_iter": 350,
        "random_state": 42,
        "verbose": False,
    }
    cpu = CutlassLogisticCV(**kwargs, backend="cpu").fit(X, y)
    gpu = CutlassLogisticCV(
        **kwargs,
        backend="cuda",
        device=0,
        allow_cpu_fallback=False,
    ).fit(X, y)

    assert gpu.backend_used_ == "cuda"
    assert gpu.C_ == cpu.C_
    assert np.allclose(gpu.cv_mean_losses_, cpu.cv_mean_losses_, rtol=1e-6, atol=1e-8)
    assert np.allclose(gpu.predict_proba(X), cpu.predict_proba(X), rtol=1e-6, atol=1e-8)
    assert np.allclose(gpu.coef_, cpu.coef_, rtol=1e-6, atol=1e-8)
    assert gpu.backend_report_["device"]["name"]
    assert gpu.backend_report_["synchronization_count"] > 0
    assert gpu.peak_device_memory_bytes_ > 0

    if penalty == "adaptive_l1":
        assert np.allclose(
            gpu.adaptive_feature_scales_,
            cpu.adaptive_feature_scales_,
            rtol=1e-6,
            atol=1e-8,
        )
        assert np.allclose(
            gpu.adaptive_weighted_coef_,
            cpu.adaptive_weighted_coef_,
            rtol=1e-6,
            atol=1e-8,
        )


def test_cuda_hybrid_matches_cpu_hybrid() -> None:
    X, y = _data()
    kwargs = {
        "Cs": [0.1, 1.0],
        "cv": 2,
        "n_jobs": 1,
        "solver": "hybrid",
        "tol": 1e-5,
        "max_iter": 300,
        "random_state": 7,
        "verbose": False,
    }
    cpu = CutlassLogisticCV(**kwargs, backend="cpu").fit(X, y)
    gpu = CutlassLogisticCV(
        **kwargs,
        backend="cuda",
        device=0,
        allow_cpu_fallback=False,
    ).fit(X, y)

    assert gpu.C_ == cpu.C_
    assert np.allclose(gpu.cv_mean_losses_, cpu.cv_mean_losses_, rtol=1e-6, atol=1e-8)
    assert np.allclose(gpu.coef_, cpu.coef_, rtol=1e-6, atol=1e-8)
    assert np.allclose(gpu.intercept_, cpu.intercept_, rtol=1e-6, atol=1e-8)
    assert gpu.fit_timings_["final_refit_cpu"] >= 0.0


def test_auto_can_select_cuda_with_calibrated_threshold(monkeypatch) -> None:
    X, y = _data()
    monkeypatch.setenv("CUTLASS_CUDA_AUTO_MIN_WORK", "1")
    model = CutlassLogisticCV(
        Cs=[0.1, 1.0],
        cv=2,
        solver="fista",
        backend="auto",
        device=0,
        allow_cpu_fallback=False,
        max_iter=250,
        verbose=False,
    ).fit(X, y)

    assert model.backend_used_ == "cuda"
    assert model.auto_decision_["selected"] == "cuda"


def test_gpu_progress_callback_error_propagates_without_fallback() -> None:
    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[1.0],
        cv=2,
        solver="fista",
        backend="cuda",
        device=0,
        allow_cpu_fallback=True,
        max_iter=100,
        verbose=False,
    )

    def fail_callback(_event):
        raise RuntimeError("progress consumer failed")

    with pytest.raises(RuntimeError, match="progress consumer failed"):
        model.fit(X, y, progress_callback=fail_callback)

    assert model.backend_used_ is None
    assert model.fallback_reason_ is None


def test_cuda_accepts_device_arrays_and_returns_numpy() -> None:
    import cupy as cp

    X, y = _data()
    model = CutlassLogisticCV(
        Cs=[0.1, 1.0],
        cv=2,
        solver="fista",
        backend="cuda",
        device=0,
        allow_cpu_fallback=False,
        max_iter=200,
        verbose=False,
    ).fit(cp.asarray(X), cp.asarray(y))

    assert model.backend_used_ == "cuda"
    assert model.backend_report_["transfer_bytes"] == 0
    assert isinstance(model.coef_, np.ndarray)
    assert isinstance(model.predict_proba(X), np.ndarray)
