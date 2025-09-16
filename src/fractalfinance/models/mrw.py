"""Multifractal Random Walk (MRW) simulator.

The implementation follows Bacry, Delour & Muzy (2001) and reproduces the
log‑normal cascade described in Chapter 5.  The Gaussian log‑volatility field
``ω_t`` has covariance

.. math:: \operatorname{Cov}(ω_t, ω_{t+τ}) = λ^2 \log^+ \frac{T}{|τ|},

regularised at the sampling interval ``dt``.  Returns are generated as

.. math:: r_t = σ \,(\Delta t)^{H} ε_t \exp(ω_t),

where ``ε_t`` are i.i.d. standard normal variables.  By enforcing
``E[exp(ω_t)] = 1`` the simulated series exhibits the structure‑function
exponents

.. math:: ζ(q) = qH - \tfrac12 λ^2 q(q-1)

used throughout the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .fbm import fbm

__all__ = ["MRWParams", "simulate", "structure_exponent"]


# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MRWParams:
    """Parameter bundle for MRW simulation."""

    H: float = 0.5
    lambda_: float = 0.2
    sigma: float = 1.0
    T: float | None = None
    dt: float = 1.0


# ---------------------------------------------------------------------------
def _log_covariance(n: int, lambda_: float, T: float, dt: float) -> np.ndarray:
    """Covariance vector :math:`λ^2 log^+(T/|τ|)` sampled at ``dt`` intervals."""

    lags = np.arange(n, dtype=float) * dt
    cov = np.empty(n, dtype=float)
    cov[0] = lambda_**2 * np.log((T + dt) / dt)
    for k in range(1, n):
        tau = lags[k]
        if tau >= T:
            cov[k] = 0.0
        else:
            cov[k] = lambda_**2 * np.log((T + dt) / (tau + dt))
    return cov


# ---------------------------------------------------------------------------
def _log_field(n: int, lambda_: float, T: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Approximate log-correlated Gaussian field via convolution kernel."""

    pad_n = 1 << int(np.ceil(np.log2(max(n, 2))))
    levels = int(np.ceil(np.log2(pad_n)))
    omega = np.zeros(pad_n, dtype=float)
    scale = lambda_ * np.sqrt(np.log(2.0))
    for k in range(levels):
        block = 2**k
        num_blocks = int(np.ceil(pad_n / block))
        vals = rng.normal(scale=scale, size=num_blocks)
        for j in range(num_blocks):
            start = j * block
            end = min((j + 1) * block, pad_n)
            omega[start:end] += vals[j]
    omega = omega[:n]
    target_var = lambda_**2 * np.log((T + dt) / dt)
    var = np.var(omega)
    if var > 0 and target_var > 0:
        omega *= np.sqrt(target_var / var)
    else:
        omega = np.zeros(n)
    return omega


# ---------------------------------------------------------------------------
def structure_exponent(q: float, H: float, lambda_: float) -> float:
    """Closed-form MRW scaling exponent ``ζ(q)``."""

    return q * H - 0.5 * lambda_**2 * q * (q - 1.0)


# ---------------------------------------------------------------------------
def simulate(
    n: int,
    params: MRWParams | None = None,
    *,
    H: float | None = None,
    lambda_: float | None = None,
    sigma: float | None = None,
    T: float | None = None,
    dt: float | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an MRW return series.

    Parameters
    ----------
    n : int
        Number of observations.
    params : :class:`Params`, optional
        Bundle of parameters to use.  Individual keyword arguments override the
        values stored in ``params``.
    H, lambda_, sigma, T, dt : float, optional
        Model parameters matching the notation in Chapter 5.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    ω_t : ndarray, shape (n,)
        Log-volatility field with mean :math:`-\tfrac12\mathrm{Var}(ω)`.
    sigma_t : ndarray, shape (n,)
        Instantaneous volatility ``exp(ω_t)``.
    r_t : ndarray, shape (n,)
        MRW returns ``σ (Δt)^H ε_t exp(ω_t)``.
    """

    if params is None:
        params = MRWParams()

    H = params.H if H is None else H
    lambda_ = params.lambda_ if lambda_ is None else lambda_
    sigma = params.sigma if sigma is None else sigma
    dt = params.dt if dt is None else dt

    if T is None:
        T = params.T if params.T is not None else n * dt
    T = float(T)

    rng = np.random.default_rng(seed)

    cov = _log_covariance(n, lambda_=lambda_, T=T, dt=dt)
    omega = _log_field(n, lambda_, T, dt, rng)

    var_omega = cov[0]
    omega = omega - 0.5 * var_omega  # ensure E[exp(ω_t)] = 1
    sigma_t = np.exp(omega)

    if abs(H - 0.5) < 1e-12:
        base = rng.normal(scale=sigma * (dt ** H), size=n)
    else:
        state = np.random.get_state()
        try:
            base = np.asarray(
                fbm(
                    H=H,
                    n=n,
                    length=n * dt,
                    kind="increment",
                    seed=None if seed is None else seed + 137,
                )
            )
        finally:
            np.random.set_state(state)
        base = base * sigma

    r_t = base * sigma_t

    return omega, sigma_t, r_t
