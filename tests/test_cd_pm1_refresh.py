
import numpy as np

from cutlass._solvers import _CDLogistic


def logistic_obj(X, y, w, b, lam):
    z = X @ w + b
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    return float(loss + lam * np.abs(w).sum())


def test_cd_pm1_solver_updates_probabilities_and_curvature():
    X = np.array(
        [
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 0, 0, 1, 0], dtype=np.float64)
    lam = 0.12

    solver = _CDLogistic(lam=lam, tol=1e-7, max_iter=2000, verbose=False, all_pm1=True)
    solver.fit(X, y)

    base_rate = float(np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / np.clip(1 - y.mean(), 1e-6, 1 - 1e-6)))
    obj_zero = logistic_obj(X, y, np.zeros(X.shape[1], dtype=np.float64), base_rate, lam)
    obj_fit = logistic_obj(X, y, solver.w_, solver.b_, lam)

    assert np.isfinite(obj_fit)
    assert obj_fit < obj_zero - 1e-4
    assert np.isfinite(solver.predict_proba(X)).all()
