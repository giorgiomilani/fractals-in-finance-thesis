"""Level-crossing diagnostics for roughness assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fractalfinance.estimators._base import BaseEstimator

__all__ = ["count_crossings", "crossing_scaling", "LevelCrossingResult"]


def count_crossings(series: Sequence[float], level: float = 0.0) -> int:
    """Count upcrossings of ``level`` in a one-dimensional series."""

    x = np.asarray(series, dtype=float) - level
    signs = np.sign(x)
    up = (signs[:-1] < 0) & (signs[1:] >= 0)
    return int(np.count_nonzero(up))


def _coarse_grain(series: np.ndarray, scale: int) -> np.ndarray | None:
    if scale <= 0:
        raise ValueError("scale must be positive")
    coarse = series[::scale]
    if coarse.size < 2:
        return None
    return coarse


@dataclass(slots=True)
class LevelCrossingResult:
    scales: np.ndarray
    counts: np.ndarray
    slope: float
    beta: float
    H: float
    slice_used: slice


def crossing_scaling(
    series: Sequence[float],
    *,
    scales: np.ndarray | None = None,
    level: float = 0.0,
    from_levels: bool = False,
) -> LevelCrossingResult:
    """Estimate crossing exponent ``β`` such that ``N(Δt) ∝ Δt^{-β}``."""

    data = np.asarray(series, dtype=float)
    level_path = data if from_levels else np.cumsum(data)

    N = data.size
    if scales is None:
        max_scale = max(2, N // 32)
        scales = np.unique(
            np.floor(np.logspace(0, np.log10(max_scale), num=8)).astype(int)
        )
    scales = scales[scales >= 1]

    counts = []
    valid_scales = []
    for s in scales:
        coarse = _coarse_grain(level_path, int(s))
        if coarse is None:
            continue
        c = count_crossings(coarse, level=level)
        if c > 0:
            counts.append(c)
            valid_scales.append(s)

    counts = np.asarray(counts, dtype=float)
    valid_scales = np.asarray(valid_scales, dtype=float)
    if counts.size < 2:
        raise RuntimeError("Not enough scales with valid crossing counts")

    log_scales = np.log(valid_scales)
    log_counts = np.log(counts)
    sl = BaseEstimator._best_range(
        log_scales, log_counts, BaseEstimator.min_points, BaseEstimator.r2_thresh
    )
    xs = log_scales[sl]
    ys = log_counts[sl]
    slope, intercept = np.polyfit(xs, ys, 1)
    beta = -float(slope)
    H = 1.0 - beta

    return LevelCrossingResult(
        scales=valid_scales,
        counts=counts,
        slope=float(slope),
        beta=beta,
        H=H,
        slice_used=sl,
    )
