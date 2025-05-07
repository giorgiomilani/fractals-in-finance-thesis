import numpy as np
from fractalfinance.risk import (
    var_gaussian, es_gaussian, var_evt, es_evt,
    kupiec, christoffersen, acerbi_szekely
)

def test_parametric_var_monotone():
    sigma = np.array([0.01, 0.02])
    assert var_gaussian(sigma, 0.99)[1] > var_gaussian(sigma, 0.99)[0]

def test_evt_var_above_gaussian():
    np.random.seed(0)
    x = np.random.randn(5000)**2          # heavy-tailed losses
    assert var_evt(x, 0.99) > var_gaussian(x.std(), 0.99)

def test_kupiec_pvalue_reasonable():
    viol = np.array([0]*98 + [1]*2)       # 2 breaks in 100
    pval = kupiec(viol, 0.99)
    assert 0.01 < pval < 0.99

def test_acerbi_zscore_centered():
    loss = np.random.randn(1000)
    var = np.quantile(loss, 0.99) * np.ones_like(loss)
    es  = var * 1.2
    z   = acerbi_szekely(loss, var, es)
    assert abs(z) < 3                     # ≈ N(0,1)
