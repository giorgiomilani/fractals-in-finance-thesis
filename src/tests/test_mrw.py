import numpy as np

from fractalfinance.estimators import StructureFunction
from fractalfinance.models.mrw import MRWParams, simulate


def test_mrw_scaling_matches_theory():
    params = MRWParams(H=0.62, lambda_=0.23, sigma=1.0, dt=1.0)
    omega, sigma_t, r = simulate(8192, params=params, seed=7)

    est = StructureFunction(
        r,
        q=np.array([1.0, 2.0, 3.0]),
        min_scale=2,
        max_scale=512,
        n_scales=12,
    )
    est.fit()
    res = est.result_
    assert res is not None

    assert abs(res.H - params.H) < 0.05
    assert abs(res.lambda_ - params.lambda_) < 0.05
