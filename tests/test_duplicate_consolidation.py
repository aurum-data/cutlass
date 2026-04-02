from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cutlass import CutlassClassifier, DuplicateColumnConsolidator
from cutlass.linear_model import CutlassLogisticCV


def test_duplicate_column_consolidator_modes_and_expansion() -> None:
    X = pd.DataFrame(
        {
            "P1_A_1": [1, -1, 1, -1],
            "P1_A_2": [1, -1, 1, -1],
            "P2_B_1": [1, -1, 1, -1],
            "P3_C_1": [-1, -1, 1, 1],
        }
    )

    within_group = DuplicateColumnConsolidator(
        mode="within_group",
        expansion="split_evenly",
    )
    X_within = within_group.fit_transform(X)
    assert X_within.shape == (4, 3)
    assert within_group.feature_names_ == ["P1_A_1", "P2_B_1", "P3_C_1"]
    assert within_group.duplicate_cols_removed_ == 1
    assert within_group.cross_group_alias_classes_ == 1

    expanded_within = within_group.expand_coefficients(np.array([6.0, 4.0, 2.0]))
    assert expanded_within["P1_A_1"] == pytest.approx(3.0)
    assert expanded_within["P1_A_2"] == pytest.approx(3.0)
    assert expanded_within["P2_B_1"] == pytest.approx(4.0)
    assert expanded_within["P3_C_1"] == pytest.approx(2.0)

    global_mode = DuplicateColumnConsolidator(
        mode="global",
        expansion="split_evenly",
    )
    X_global = global_mode.fit_transform(X)
    assert X_global.shape == (4, 2)
    assert global_mode.feature_names_ == ["P1_A_1", "P3_C_1"]
    assert global_mode.duplicate_cols_removed_ == 2

    expanded_global = global_mode.expand_coefficients(np.array([6.0, 2.0]))
    assert expanded_global["P1_A_1"] == pytest.approx(2.0)
    assert expanded_global["P1_A_2"] == pytest.approx(2.0)
    assert expanded_global["P2_B_1"] == pytest.approx(2.0)
    assert expanded_global["P3_C_1"] == pytest.approx(2.0)


def test_cutlass_classifier_deduplicates_and_expands_evenly() -> None:
    signal = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.float64)
    noise = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.float64)
    X = pd.DataFrame(
        {
            "P1_S_1": signal,
            "P1_S_2": signal,
            "P2_N_1": noise,
            "P2_N_2": noise,
        }
    )
    y = (signal == 1).astype(int)

    clf = CutlassClassifier(
        rectify=False,
        use_scaler=False,
        Cs=[10.0],
        cv=2,
        solver="cd",
        tol=1e-5,
        max_iter=2000,
        verbose=False,
    )
    clf.fit(X, y)

    assert clf.duplicate_report_["mode"] == "within_group"
    assert clf.duplicate_report_["duplicate_cols_removed"] == 2
    assert clf.fit_feature_names_ == ["P1_S_1", "P2_N_1"]
    assert clf.coef_fit_.shape == (1, 2)
    assert clf.coef_.shape == (1, 4)

    assert clf.coef_[0, 0] == pytest.approx(clf.coef_[0, 1])
    assert clf.coef_[0, 2] == pytest.approx(clf.coef_[0, 3])
    assert clf.coef_[0, 0] + clf.coef_[0, 1] == pytest.approx(clf.coef_fit_[0, 0])
    assert clf.coef_[0, 2] + clf.coef_[0, 3] == pytest.approx(clf.coef_fit_[0, 1])
    assert abs(clf.coef_fit_[0, 0]) > 1e-6

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.isfinite(proba).all()


def test_cutlass_classifier_logic_polish_defaults_to_expanded_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.float64)
    noise = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.float64)
    X = pd.DataFrame(
        {
            "P1_S_1": signal,
            "P1_S_2": signal,
            "P2_N_1": noise,
            "P2_N_2": noise,
        }
    )
    y = (signal == 1).astype(int)

    calls: list[int] = []

    def fake_logical_polish(self, X, y, w, b, **kwargs):
        calls.append(int(X.shape[1]))
        w_new = np.zeros_like(w, dtype=np.float64)
        b_new = 0.0
        return w_new, b_new, 1.0, True, [], {"called_on_p": int(X.shape[1])}

    monkeypatch.setattr(CutlassLogisticCV, "_logical_polish", fake_logical_polish)

    clf = CutlassClassifier(
        rectify=False,
        use_scaler=False,
        Cs=[10.0],
        cv=2,
        solver="cd",
        tol=1e-5,
        max_iter=2000,
        logic_polish=True,
        verbose=False,
    )
    clf.fit(X, y)

    assert clf.logic_polish_stage == "expanded"
    assert calls == [4]
    assert clf.coef_fit_.shape == (1, 2)
    assert clf.coef_.shape == (1, 4)
    assert np.allclose(clf.coef_, 0.0)
    assert clf.logic_diag_["stage"] == "expanded"

    proba = clf.predict_proba(X)
    assert np.allclose(proba[:, 1], 0.5)
    assert np.allclose(proba[:, 0], 0.5)


def test_cutlass_classifier_can_still_logic_polish_on_fit_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.float64)
    noise = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.float64)
    X = pd.DataFrame(
        {
            "P1_S_1": signal,
            "P1_S_2": signal,
            "P2_N_1": noise,
            "P2_N_2": noise,
        }
    )
    y = (signal == 1).astype(int)

    calls: list[int] = []

    def fake_logical_polish(self, X, y, w, b, **kwargs):
        calls.append(int(X.shape[1]))
        w_new = np.zeros_like(w, dtype=np.float64)
        b_new = 0.0
        return w_new, b_new, 1.0, True, [], {"called_on_p": int(X.shape[1])}

    monkeypatch.setattr(CutlassLogisticCV, "_logical_polish", fake_logical_polish)

    clf = CutlassClassifier(
        rectify=False,
        use_scaler=False,
        Cs=[10.0],
        cv=2,
        solver="cd",
        tol=1e-5,
        max_iter=2000,
        logic_polish=True,
        logic_polish_stage="fit",
        verbose=False,
    )
    clf.fit(X, y)

    assert calls == [2]
    assert clf.logic_diag_["stage"] == "fit"
    assert np.allclose(clf.coef_fit_, 0.0)
    assert np.allclose(clf.coef_, 0.0)
