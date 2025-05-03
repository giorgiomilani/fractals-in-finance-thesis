import numpy as np
import pandas as pd

from fractalfinance.estimators import RS


def test_rs_on_random_walk():
    np.random.seed(0)
    x = np.random.randn(4096).cumsum()
    est = RS(pd.Series(x)).fit()
    assert 0.4 < est.result_["H"] < 0.6  # Brownian H≈0.5
