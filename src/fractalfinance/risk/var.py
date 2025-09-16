"""Fractal and classical risk measures.

The module now complements the traditional Gaussian and EVT estimators
with heavy-tailed and multifractal counterparts inspired by the thesis
chapter on fractal risk management:

* :func:`var_gaussian` / :func:`es_gaussian` – parametric normal formulas
* :func:`var_evt`      / :func:`es_evt`      – POT-EVT tail via maximum-
  likelihood GPD
* :func:`var_stable`   / :func:`es_stable`   – α-stable (fractal) risk measures
* :func:`spectral_risk_measure` – spectral risk measure estimator
* :func:`multifractal_var` – VaR with multifractal scaling
* :func:`var_evt_fractal` / :func:`es_evt_fractal` – EVT with dynamic GPD
  shape linked to multifractal width
* :func:`regime_dependent_risk` – regime-sensitive risk proxy
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import genpareto, levy_stable, norm


# ------------------------------------------------------------------ #
def var_gaussian(sigma: float | np.ndarray, p: float = 0.99) -> np.ndarray:
    """One‑sided Gaussian VaR."""
    return norm.ppf(p) * np.asarray(sigma)


def es_gaussian(sigma: float | np.ndarray, p: float = 0.99) -> np.ndarray:
    """Gaussian Expected Shortfall."""
    z = norm.ppf(p)
    return (sigma / (1.0 - p)) * norm.pdf(z)


# ------------------------------------------------------------------ #
def var_stable(
    mu: float | np.ndarray,
    gamma: float | np.ndarray,
    alpha: float,
    beta: float = 0.0,
    p: float = 0.99,
) -> np.ndarray:
    """Stable (α-stable) Value-at-Risk."""

    if not 0.0 < alpha <= 2.0:
        raise ValueError("alpha must be in (0, 2]")
    if not -1.0 <= beta <= 1.0:
        raise ValueError("beta must be within [-1, 1]")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")

    q = levy_stable.ppf(p, alpha, beta)
    return np.asarray(mu, dtype=float) + np.asarray(gamma, dtype=float) * q


def es_stable(
    mu: float | np.ndarray,
    gamma: float | np.ndarray,
    alpha: float,
    beta: float = 0.0,
    p: float = 0.99,
) -> np.ndarray:
    """Stable (α-stable) Expected Shortfall."""

    if alpha <= 1.0:
        return np.full_like(np.asarray(mu, dtype=float), np.inf, dtype=float)
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")

    q = levy_stable.ppf(p, alpha, beta)
    cond_mean = levy_stable.expect(
        lambda x: x,
        args=(alpha, beta),
        lb=float(q),
        conditional=True,
    )
    return np.asarray(mu, dtype=float) + np.asarray(gamma, dtype=float) * cond_mean


def spectral_risk_measure(
    losses: ArrayLike,
    phi: Callable[[np.ndarray], ArrayLike] | ArrayLike,
) -> float:
    """Spectral risk measure :math:`ρ_φ(X)` using a discrete approximation."""

    loss = np.sort(np.asarray(losses, dtype=float))
    if loss.size == 0:
        raise ValueError("losses must contain at least one observation")

    u = (np.arange(1, loss.size + 1) - 0.5) / loss.size
    weights = phi(u) if callable(phi) else phi
    weights = np.asarray(weights, dtype=float)

    if weights.shape != loss.shape:
        raise ValueError("phi must match the length of losses")
    if np.any(weights < 0.0):
        raise ValueError("phi must be non-negative")

    mean_weight = weights.mean()
    if mean_weight <= 0.0:
        raise ValueError("phi must integrate to a positive value")

    normalized = weights / mean_weight  # ensures discrete integral equals one
    return float(np.mean(normalized * loss))


def _default_multifractal_quantile(alpha: float, delta_alpha: float) -> float:
    base = norm.ppf(alpha)
    scale = np.exp(0.5 * delta_alpha)
    return float(base * scale)


def multifractal_var(
    sigma0: float | np.ndarray,
    delta_t: float,
    hurst: Callable[[float], float] | float,
    alpha: float,
    delta_alpha: float = 0.0,
    q_alpha: float | None = None,
    quantile_func: Callable[[float, float], float] | None = None,
) -> np.ndarray:
    """Multifractal VaR :math:`σ_0 (Δt)^{h(q_α)} F^{-1}`."""

    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    q = alpha if q_alpha is None else q_alpha
    h_val = hurst(q) if callable(hurst) else float(hurst)
    quantile = (
        _default_multifractal_quantile(alpha, delta_alpha)
        if quantile_func is None
        else float(quantile_func(alpha, delta_alpha))
    )
    sigma0_arr = np.asarray(sigma0, dtype=float)
    return sigma0_arr * (delta_t**h_val) * quantile


# ------------------------------------------------------------------ #
def _fit_gpd_mle(exc: np.ndarray) -> tuple[float, float]:
    """
    Maximum‑likelihood fit of the Generalised Pareto Distribution to tail
    exceedances (floc=0 ensures peaks‑over‑threshold parameterisation).
    Returns shape ξ (k) and scale β.
    """
    k, loc, beta = genpareto.fit(exc, floc=0.0)
    return float(k), float(beta)


def _anderson_darling_gpd(exc: np.ndarray, k: float, beta: float) -> float:
    """Anderson–Darling statistic for GPD fit."""
    exc = np.sort(exc)
    n = exc.size
    cdf = genpareto.cdf(exc, c=k, scale=beta)
    i = np.arange(1, n + 1)
    S = np.sum((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])), dtype=float)
    return -n - S / n


def var_evt(
    x: ArrayLike,
    p: float = 0.99,
    threshold_q: float = 0.95,
    diagnostics: bool = False,
):
    """
    POT‑EVT VaR for *positive* losses (heavy right tail).

    When ``diagnostics`` is ``True`` the function also returns the
    Anderson–Darling statistic and QQ‑plot theoretical quantiles.
    """
    x = np.asarray(x, dtype=float).ravel()
    u = np.quantile(x, threshold_q)
    exc = x[x > u] - u

    # if too few exceedances, lower the threshold to 0.90
    if exc.size < 50:
        u = np.quantile(x, 0.90)
        exc = x[x > u] - u

    k, beta = _fit_gpd_mle(exc)
    ad = _anderson_darling_gpd(exc, k, beta)
    n, n_exc = x.size, exc.size
    tail_prob = (1.0 - p) / (n_exc / n)  # conditional exceed. prob

    if k != 0:
        var = u + (beta / k) * (tail_prob ** (-k) - 1.0)
    else:  # k → 0 (exp tail)
        var = u - beta * np.log(tail_prob)

    if diagnostics:
        i = (np.arange(1, exc.size + 1) - 0.5) / exc.size
        theo = genpareto.ppf(i, c=k, scale=beta)
        return float(var), float(ad), (np.sort(exc), theo)

    return float(var)


def es_evt(
    x: ArrayLike,
    p: float = 0.99,
    threshold_q: float = 0.95,
) -> float:
    """
    EVT Expected Shortfall corresponding to var_evt.
    """
    x = np.asarray(x, dtype=float).ravel()
    u = np.quantile(x, threshold_q)
    exc = x[x > u] - u
    k, beta = _fit_gpd_mle(exc)

    var = var_evt(x, p, threshold_q)
    if k >= 1:
        return np.inf  # ES diverges
    return (var + (beta - k * u)) / (1.0 - k)


def var_evt_fractal(
    x: ArrayLike,
    p: float = 0.99,
    threshold_q: float = 0.95,
    delta_alpha: float | None = None,
    xi0: float | None = None,
    beta_coeff: float | None = None,
    diagnostics: bool = False,
):
    """EVT VaR with optional fractal adjustment of the shape parameter."""

    x = np.asarray(x, dtype=float).ravel()
    u = np.quantile(x, threshold_q)
    exc = x[x > u] - u

    if exc.size < 50:
        u = np.quantile(x, 0.90)
        exc = x[x > u] - u

    k_mle, beta = _fit_gpd_mle(exc)
    ad = _anderson_darling_gpd(exc, k_mle, beta)

    if None not in (delta_alpha, xi0, beta_coeff):
        k = xi0 + beta_coeff * float(delta_alpha)
    else:
        k = k_mle

    n, n_exc = x.size, exc.size
    tail_prob = (1.0 - p) / (n_exc / n)

    if k != 0:
        var = u + (beta / k) * (tail_prob ** (-k) - 1.0)
    else:
        var = u - beta * np.log(tail_prob)

    if diagnostics:
        i = (np.arange(1, exc.size + 1) - 0.5) / exc.size
        theo = genpareto.ppf(i, c=k, scale=beta)
        return float(var), float(ad), (np.sort(exc), theo)

    return float(var)


def es_evt_fractal(
    x: ArrayLike,
    p: float = 0.99,
    threshold_q: float = 0.95,
    delta_alpha: float | None = None,
    xi0: float | None = None,
    beta_coeff: float | None = None,
) -> float:
    """Fractal Expected Shortfall corresponding to :func:`var_evt_fractal`."""

    x = np.asarray(x, dtype=float).ravel()
    u = np.quantile(x, threshold_q)
    exc = x[x > u] - u

    if exc.size < 50:
        u = np.quantile(x, 0.90)
        exc = x[x > u] - u

    k_mle, beta = _fit_gpd_mle(exc)
    if None not in (delta_alpha, xi0, beta_coeff):
        k = xi0 + beta_coeff * float(delta_alpha)
    else:
        k = k_mle

    var = var_evt_fractal(
        x,
        p=p,
        threshold_q=threshold_q,
        delta_alpha=delta_alpha,
        xi0=xi0,
        beta_coeff=beta_coeff,
    )

    if k >= 1:
        return float("inf")

    return float((var + (beta - k * u)) / (1.0 - k))


def regime_dependent_risk(
    delta_alpha: ArrayLike,
    hurst: ArrayLike,
    sigma: ArrayLike,
    delta_t: float = 1.0,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Simple proxy :math:`f(Δα_t, H_t, σ_t)` for regime-dependent risk."""

    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive")
    if len(weights) != 3:
        raise ValueError("weights must contain three elements")

    w_delta, w_hurst, w_sigma = (float(w) for w in weights)
    delta_alpha = np.asarray(delta_alpha, dtype=float)
    hurst = np.asarray(hurst, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    scaling = 1.0 + w_delta * delta_alpha
    hurst_scale = delta_t ** (w_hurst * hurst)
    sigma_term = w_sigma * sigma
    return sigma_term * scaling * hurst_scale
