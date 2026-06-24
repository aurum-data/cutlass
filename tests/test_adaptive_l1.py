from __future__ import annotations

import numpy as np
import pandas as pd

from cutlass import CutlassClassifier
from cutlass.linear_model import CutlassLogisticCV


def _toy_data():
    X = pd.DataFrame(
        {
            "signal": [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
            "weak": [-1.0, -1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            "noise": [0.3, -0.2, 0.1, -0.4, 0.2, -0.1, 0.4, -0.3],
        },
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    return X, y


def test_cutlass_logistic_cv_adaptive_l1_maps_coefficients_back() -> None:
    X, y = _toy_data()
    model = CutlassLogisticCV(
        Cs=[0.2, 2.0],
        penalty="adaptive_l1",
        cv=2,
        n_jobs=1,
        solver="cd",
        tol=1e-5,
        max_iter=1000,
        verbose=False,
    )

    model.fit(X.to_numpy(dtype=np.float64), y)

    assert model.C_ in {0.2, 2.0}
    assert model.coef_.shape == (1, X.shape[1])
    assert model.adaptive_feature_scales_.shape == (X.shape[1],)
    assert np.all(model.adaptive_feature_scales_ > 0.0)
    assert np.allclose(
        model.adaptive_weighted_coef_.ravel() * model.adaptive_feature_scales_,
        model.coef_.ravel(),
    )

    proba = model.predict_proba(X.to_numpy(dtype=np.float64))
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_cutlass_classifier_exposes_adaptive_l1_mode() -> None:
    X, y = _toy_data()
    clf = CutlassClassifier(
        rectify=False,
        duplicate_mode="none",
        use_scaler=False,
        Cs=[0.2, 2.0],
        cv=2,
        solver="cd",
        tol=1e-5,
        max_iter=1000,
        verbose=False,
        penalty="adaptive_l1",
    )

    clf.fit(X, y)

    assert clf.get_params()["penalty"] == "adaptive_l1"
    assert clf.classifier_.penalty == "adaptive_l1"
    assert clf.classifier_.adaptive_feature_scales_.shape == (X.shape[1],)
    assert clf.coef_.shape == (1, X.shape[1])

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()
