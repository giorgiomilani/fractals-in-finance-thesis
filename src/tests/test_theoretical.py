import numpy as np
import pandas as pd

from fractalfinance.preprocessing.core import wavelet_detrend
from fractalfinance.estimators.dfa import DFA
from fractalfinance.risk.var import var_evt


def test_wavelet_detrend_removes_linear_trend():
    t = np.arange(256)
    series = pd.Series(0.5 * t + np.random.randn(256))
    detrended = wavelet_detrend(series)
    a, b = np.polyfit(t, detrended.values, 1)
    assert abs(a) < 0.1


def test_dfa_returns_bootstrap_std():
    path = np.cumsum(np.random.randn(1024))
    dfa = DFA(path, auto_range=True, n_boot=5).fit()
    assert 0 < dfa.result_["H"] < 1
    assert "H_std" in dfa.result_


def test_var_evt_returns_diagnostics():
    x = np.random.lognormal(size=1000)
    var, ad, (emp, theo) = var_evt(x, diagnostics=True)
    assert var > 0
    assert ad > 0
    assert emp.shape == theo.shape
