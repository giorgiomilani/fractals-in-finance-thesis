import numpy as np
from fractalfinance.estimators import MFDFA
from fractalfinance.models import CascadeParams, mmar_simulate


def test_mmar_is_multifractal():
    theta, X, r = mmar_simulate(
        2048, H=0.5, cascade=CascadeParams(depth=9, seed=0), seed=0
    )
    est = MFDFA(X).fit()
    width = np.ptp(est.result_["alpha"])
    assert width > 0.5  # FBM width≈0 → MMAR width should be wider
