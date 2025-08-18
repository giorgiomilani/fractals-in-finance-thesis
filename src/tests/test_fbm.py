import numpy as np

from fractalfinance.estimators import DFA
from fractalfinance.models import fbm


def test_fbm_hurst_accuracy():
    np.random.seed(123)
    H_true = 0.7
    n = 8192
    series = fbm(H_true, n, seed=123)
    est = DFA(series).fit()
    assert abs(est.result_["H"] - H_true) < 0.05
