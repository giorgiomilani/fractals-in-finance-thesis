import numpy as np

from fractalfinance.estimators.dfa import DFA
from fractalfinance.models.fbm import fbm


def test_fbm_hurst_accuracy():
    np.random.seed(123)
    H_true = 0.7
    n = 8192
    series = fbm(H_true, n, seed=123)
    est = DFA(series, from_levels=True).fit()
    assert abs(est.result_["H"] - H_true) < 0.05
