"""
Value‑at‑Risk (VaR) and Expected Shortfall (ES)
===============================================
* var_gaussian / es_gaussian – parametric normal formulas
* var_evt      / es_evt      – POT‑EVT tail using maximum‑likelihood GPD
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm, genpareto
from numpy.typing import ArrayLike


# ------------------------------------------------------------------ #
def var_gaussian(sigma: float | np.ndarray, p: float = 0.99) -> np.ndarray:
    """One‑sided Gaussian VaR."""
    return norm.ppf(p) * np.asarray(sigma)


def es_gaussian(sigma: float | np.ndarray, p: float = 0.99) -> np.ndarray:
    """Gaussian Expected Shortfall."""
    z = norm.ppf(p)
    return (sigma / (1.0 - p)) * norm.pdf(z)


# ------------------------------------------------------------------ #
def _fit_gpd_mle(exc: np.ndarray) -> tuple[float, float]:
    """
    Maximum‑likelihood fit of the Generalised Pareto Distribution to tail
    exceedances (floc=0 ensures peaks‑over‑threshold parameterisation).
    Returns shape ξ (k) and scale β.
    """
    k, loc, beta = genpareto.fit(exc, floc=0.0)
    return float(k), float(beta)


def var_evt(
    x: ArrayLike,
    p: float = 0.99,
    threshold_q: float = 0.95,
) -> float:
    """
    POT‑EVT VaR for *positive* losses (heavy right tail).

    Parameters
    ----------
    x : 1‑D array‑like
    p : confidence level (e.g. 0.99)
    threshold_q : quantile for threshold u (default 0.95)

    Returns
    -------
    float  VaRₚ
    """
    x = np.asarray(x, dtype=float).ravel()
    u = np.quantile(x, threshold_q)
    exc = x[x > u] - u

    # if too few exceedances, lower the threshold to 0.90
    if exc.size < 50:
        u = np.quantile(x, 0.90)
        exc = x[x > u] - u

    k, beta = _fit_gpd_mle(exc)
    n, n_exc = x.size, exc.size
    tail_prob = (1.0 - p) / (n_exc / n)           # conditional exceed. prob

    if k != 0:
        var = u + (beta / k) * (tail_prob ** (-k) - 1.0)
    else:                                         # k → 0 (exp tail)
        var = u - beta * np.log(tail_prob)

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
        return np.inf          # ES diverges
    return (var + (beta - k * u)) / (1.0 - k)
