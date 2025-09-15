from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from .fbm import fbm

@dataclass
class CascadeParams:
    m_L: float = 0.5
    m_H: float = 2.0
    depth: int = 8
    seed: int | None = None

def simulate(
    n: int,
    H: float,
    *,
    cascade: CascadeParams | None = None,
    m_L: float = 0.5,
    m_H: float = 2.0,
    depth: int = 8,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a simple MMAR path.

    The Hurst exponent *H* controls the roughness of the underlying
    fractional Brownian motion increments which are then modulated by a
    lognormal multiplicative cascade.
    """
    if cascade is not None:
        m_L, m_H, depth, seed = (
            cascade.m_L,
            cascade.m_H,
            cascade.depth,
            cascade.seed,
        )

    # Base fractional Gaussian noise reflecting the Hurst exponent
    fgn = fbm(H=H, n=n, kind="increment", seed=seed)

    # Multiplicative cascade of volatilities
    rng = np.random.default_rng(None if seed is None else seed + 1)
    sigma = m_H - m_L
    multipliers = rng.lognormal(mean=np.log(m_L), sigma=sigma, size=n)
    r = fgn * multipliers
    X = np.cumsum(r)
    theta = multipliers
    return theta, X, r

# Legacy alias used in tests
mmar_simulate = simulate

def calibrate(series: np.ndarray, H: float, search_mH: Tuple[float, float, int]) -> CascadeParams:
    """Calibrate ``m_H`` by matching return variances over a search grid."""
    start, stop, steps = search_mH
    grid = np.linspace(start, stop, int(steps))
    target = np.var(np.diff(series))
    best_mH = grid[0]
    best_err = float("inf")
    for mH in grid:
        _, _, r = simulate(len(series), H, m_L=0.5, m_H=mH, depth=8, seed=0)
        err = abs(np.var(r) - target)
        if err < best_err:
            best_err = err
            best_mH = mH
    return CascadeParams(m_L=0.5, m_H=float(best_mH), depth=8, seed=None)
