from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

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
    if cascade is not None:
        m_L, m_H, depth, seed = cascade.m_L, cascade.m_H, cascade.depth, cascade.seed
    rng = np.random.default_rng(seed)
    sigma = m_H - m_L
    multipliers = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    r = rng.normal(scale=multipliers, size=n)
    X = np.cumsum(r)
    theta = multipliers
    return theta, X, r

# Legacy alias used in tests
mmar_simulate = simulate

def calibrate(series: np.ndarray, H: float, search_mH: Tuple[float, float, int]) -> CascadeParams:
    start, stop, steps = search_mH
    r = np.diff(series)
    target = np.var(r)
    sigma_est = np.sqrt(0.5 * np.log(target))
    m_H_est = 0.5 + sigma_est
    grid = np.linspace(start, stop, int(steps))
    best_mH = grid[np.argmin(np.abs(grid - m_H_est))]
    return CascadeParams(m_L=0.5, m_H=float(best_mH), depth=8, seed=None)
