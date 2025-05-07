import numpy as np, pandas as pd
from fractalfinance.models import GARCH, HAR

def test_garch_forecast_positive():
    np.random.seed(0)
    r = pd.Series(np.random.randn(2000) * 0.01)
    mdl = GARCH().fit(r)
    fc = mdl.forecast(h=5)
    assert (fc > 0).all() and fc.size == 5

def test_har_forecast_reasonable():
    # create synthetic RV with volatility clustering
    np.random.seed(1)
    rv = pd.Series(0.0001 + 0.00005 * np.random.randn(100).cumsum() ** 2)
    rv = rv.abs()
    mdl = HAR().fit(rv)
    fc = mdl.forecast(h=3)
    assert (fc > 0).all() and len(fc) == 3
