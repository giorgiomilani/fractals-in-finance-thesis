import numpy as np

from fractalfinance.models.msm_class import MSM, Params


def test_msm_forecast_positive():
    np.random.seed(0)
    p = Params(1.0, 0.7, 1.4, 0.05, 2.0, 4)
    # simulate 2 000 returns with older msm_simulate() helper
    from fractalfinance.models import msm_simulate

    _, r = msm_simulate(2000, p, seed=0)
    mdl = MSM.fit(r, p)
    v1 = mdl.forecast(h=1)
    v5 = mdl.forecast(h=5)
    assert v1 > 0 and v5 > 0 and abs(v5 - v1) < 5  # arbitrary sanity
