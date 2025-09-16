import numpy as np

from fractalfinance.risk import (
    acerbi_szekely,
    christoffersen,
    es_evt,
    es_evt_fractal,
    es_gaussian,
    es_stable,
    kupiec,
    multifractal_var,
    regime_dependent_risk,
    spectral_risk_measure,
    var_evt,
    var_evt_fractal,
    var_gaussian,
    var_stable,
)
from fractalfinance.risk.var import _fit_gpd_mle


def test_parametric_var_monotone():
    sigma = np.array([0.01, 0.02])
    assert var_gaussian(sigma, 0.99)[1] > var_gaussian(sigma, 0.99)[0]


def test_evt_var_above_gaussian():
    np.random.seed(0)
    x = np.random.randn(5000) ** 2  # heavy-tailed losses
    assert var_evt(x, 0.99) > var_gaussian(x.std(), 0.99)


def test_kupiec_pvalue_reasonable():
    viol = np.array([0] * 98 + [1] * 2)  # 2 breaks in 100
    pval = kupiec(viol, 0.99)
    assert 0.01 < pval < 0.99


def test_acerbi_zscore_centered():
    loss = np.random.randn(1000)
    var = np.quantile(loss, 0.99) * np.ones_like(loss)
    es = var * 1.2
    z = acerbi_szekely(loss, var, es)
    assert abs(z) < 3  # ≈ N(0,1)


def test_stable_var_heavier_tail():
    sigma = 0.02
    normal = var_gaussian(sigma, 0.99)
    stable = var_stable(0.0, sigma, alpha=1.7, beta=0.0, p=0.99)
    assert stable > normal


def test_stable_es_infinite_when_alpha_leq_one():
    es = es_stable(0.0, 1.0, alpha=0.8, beta=0.0, p=0.99)
    assert np.isinf(es)


def test_spectral_risk_tail_averse():
    losses = np.linspace(0.5, 5.0, 64)
    flat = spectral_risk_measure(losses, lambda u: np.ones_like(u))
    tail = spectral_risk_measure(losses, lambda u: u**4)
    assert tail > flat


def test_multifractal_var_monotone_in_delta_alpha():
    base = multifractal_var(0.02, delta_t=1.0, hurst=0.5, alpha=0.99, delta_alpha=0.0)
    stressed = multifractal_var(
        0.02, delta_t=1.0, hurst=0.5, alpha=0.99, delta_alpha=0.6
    )
    assert stressed > base


def test_var_evt_fractal_matches_evt_without_adjustment():
    rng = np.random.default_rng(0)
    losses = np.abs(rng.standard_t(df=4, size=5000))
    base = var_evt(losses, 0.99)
    fractal = var_evt_fractal(losses, 0.99)
    assert np.isclose(base, fractal, rtol=1e-6)


def test_var_evt_fractal_increases_with_positive_delta_alpha():
    rng = np.random.default_rng(1)
    losses = np.abs(rng.standard_t(df=4, size=6000))
    u = np.quantile(losses, 0.95)
    exc = losses[losses > u] - u
    xi0, _ = _fit_gpd_mle(exc)
    base = var_evt(losses, 0.99)
    adj = var_evt_fractal(losses, 0.99, delta_alpha=0.5, xi0=xi0, beta_coeff=0.4)
    assert adj > base


def test_es_evt_fractal_infinite_when_shape_at_least_one():
    rng = np.random.default_rng(2)
    losses = rng.lognormal(mean=0.0, sigma=0.6, size=4000)
    es = es_evt_fractal(losses, 0.99, delta_alpha=0.0, xi0=1.1, beta_coeff=0.0)
    assert np.isinf(es)


def test_regime_dependent_risk_increasing_components():
    delta_alpha = np.array([0.0, 0.3])
    hurst = np.array([0.4, 0.6])
    sigma = np.array([0.02, 0.02])
    risk = regime_dependent_risk(delta_alpha, hurst, sigma, delta_t=1.0)
    assert risk[1] > risk[0]
