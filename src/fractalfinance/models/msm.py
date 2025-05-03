"""
Markov–Switching Multifractal (MSM) Variance Model
==================================================
Implementation follows Calvet & Fisher (2004).

simulate(...)  → σ_t, r_t   (r_t = σ_t ε_t)
loglik(...)    → Gaussian log‑likelihood via Hamilton filter
fit(...)       → Pedagogical grid‑search MLE
"""

from __future__ import annotations

from itertools import product
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

__all__ = ["MSMParams", "simulate", "loglik", "fit"]


# ------------------------------------------------------------------ #
class MSMParams(NamedTuple):
    sigma2: float
    m_L: float
    m_H: float
    gamma_1: float
    b: float
    K: int


# ------------------------------------------------------------------ #
def _gamma_k(g1: float, b: float, k: int) -> float:
    """Scale‑dependent switching probability γ_k."""
    return 1.0 - (1.0 - g1) ** (b ** (k - 1))


# ------------------------------------------------------------------ #
def _grid(lo: float, hi: float, num: int) -> np.ndarray:
    """Inclusive linspace so both boundaries are evaluated."""
    if num <= 1:
        return np.array([lo])
    return np.linspace(lo, hi, num, endpoint=True)


# ------------------------------------------------------------------ #
def simulate(
    n: int,
    params: MSMParams,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n observations of (σ_t, r_t) from an MSM(K) model.

    Returns
    -------
    sigma_t : ndarray, shape (n,)
    r_t     : ndarray, shape (n,)
    """
    if seed is not None:
        np.random.seed(seed)

    K = params.K
    m_vals = np.array([params.m_L, params.m_H])

    # initialise states S_k(0) ~ Bernoulli(0.5)
    states = np.random.randint(0, 2, size=K)
    sigma_t = np.empty(n)
    for t in range(n):
        # update each multiplier independently
        for k in range(K):
            if np.random.rand() < _gamma_k(params.gamma_1, params.b, k + 1):
                states[k] ^= 1  # switch 0↔1

        M_t = m_vals[states].prod()
        sigma_t[t] = np.sqrt(params.sigma2 * M_t)

    r_t = sigma_t * np.random.randn(n)
    return sigma_t, r_t


# ------------------------------------------------------------------ #
def _all_states(K: int) -> np.ndarray:
    """Enumerate the 2^K possible multiplier state vectors as 0/1 arrays."""
    return np.array(list(product([0, 1], repeat=K)), dtype=int)


def _transition_matrix(params: MSMParams) -> np.ndarray:
    """
    Transition matrix P(s′ | s) factorises across scales:
        P = ⨂_{k=1}^K P_k
    """
    states = _all_states(params.K)
    P = np.ones((2 ** params.K, 2 ** params.K))
    for k in range(params.K):
        gk = _gamma_k(params.gamma_1, params.b, k + 1)
        same = states[:, k, None] == states[None, :, k]
        P *= np.where(same, 1.0 - gk, gk)
    return P


# ------------------------------------------------------------------ #
def loglik(r: ArrayLike, params: MSMParams) -> float:
    """
    Gaussian log‑likelihood via forward recursion (Hamilton filter).
    """
    r = np.asarray(r)
    states = _all_states(params.K)
    m_vals = np.where(states == 0, params.m_L, params.m_H)
    sigma_state = np.sqrt(params.sigma2 * m_vals.prod(axis=1))

    logP = np.log(_transition_matrix(params))
    log_alpha = np.full(states.shape[0], -np.log(states.shape[0]))  # flat prior
    const = -0.5 * np.log(2 * np.pi)

    ll = 0.0
    for rt in r:
        log_emit = const - np.log(sigma_state) - 0.5 * (rt / sigma_state) ** 2
        log_alpha = log_emit + logsumexp(logP + log_alpha[:, None], axis=0)
        ll += logsumexp(log_alpha)
        log_alpha -= logsumexp(log_alpha)  # normalise
    return ll


# ------------------------------------------------------------------ #
def fit(
    r: ArrayLike,
    K: int = 6,
    grid_m_H: tuple[float, float, int] = (1.4, 2.0, 4),
    grid_gamma1: tuple[float, float, int] = (0.01, 0.1, 4),
    fixed: dict | None = None,
) -> MSMParams:
    """
    Pedagogical grid‑search MLE (slow but simple).

    Returns
    -------
    MSMParams
        The parameter set with highest Gaussian log‑likelihood on the grid.
    """
    r = np.asarray(r)
    fixed = fixed or {}
    best_ll, best_par = -np.inf, None
    sigma2 = np.var(r)

    for mH in _grid(*grid_m_H):
        for g1 in _grid(*grid_gamma1):
            par = MSMParams(
                sigma2=sigma2,
                m_L=1 / mH,  # symmetry constraint
                m_H=mH,
                gamma_1=g1,
                b=fixed.get("b", 2.0),
                K=K,
            )
            ll = loglik(r, par)
            if ll > best_ll:
                best_ll, best_par = ll, par

    return best_par
