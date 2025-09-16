import numpy as np

from fractalfinance.geometry.intermittency import (
    clustering_coefficient,
    estimate_mu,
)
from fractalfinance.models.mrw import MRWParams, simulate


def test_intermittency_matches_lambda_square():
    params = MRWParams(H=0.58, lambda_=0.22)
    omega, sigma_t, r = simulate(8192, params=params, seed=21)
    res = estimate_mu(r, p=2.0, min_scale=2, max_scale=512)
    assert abs(res.mu - params.lambda_**2) < 0.05


def test_clustering_coefficient_baseline():
    rng = np.random.default_rng(9)
    series = rng.normal(size=5000)
    coeffs = clustering_coefficient(series, threshold=series.std() * 1.5, max_lag=5)
    assert np.all(np.isfinite(coeffs.coefficient))
    assert np.allclose(coeffs.coefficient, 1.0, atol=0.2)
