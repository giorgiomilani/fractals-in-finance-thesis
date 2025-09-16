import numpy as np

from fractalfinance.estimators import MFDFA
from fractalfinance.models import CascadeParams, calibrate, generate_cascade, mmar_simulate


def test_mmar_is_multifractal():
    theta_speed, X, r = mmar_simulate(
        2048, H=0.5, cascade=CascadeParams(depth=9, seed=0), seed=0
    )
    est = MFDFA(X).fit()
    width = np.ptp(est.result_["alpha"])
    assert width > 0.5  # FBM width≈0 → MMAR width should be wider
    assert np.isclose(theta_speed.mean(), 1.0, atol=1e-6)


def test_mmar_calibration_grid_search():
    true_params = CascadeParams(m_L=0.55, m_H=1.75, depth=7, seed=123)
    _, X, _ = mmar_simulate(1024, H=0.7, cascade=true_params, seed=123)
    fitted = calibrate(
        X,
        H=0.7,
        depth=7,
        search_mL=(0.4, 0.7, 4),
        search_mH=(1.4, 2.0, 4),
        n_trials=3,
        seed=321,
    )
    assert abs(fitted.m_L - true_params.m_L) < 0.2
    assert abs(fitted.m_H - true_params.m_H) < 0.2


def test_generate_cascade_mean_one():
    params = CascadeParams(m_L=0.5, m_H=1.8, depth=6, seed=42)
    weights = generate_cascade(params)
    assert np.isclose(weights.mean(), 1.0, atol=1e-6)
