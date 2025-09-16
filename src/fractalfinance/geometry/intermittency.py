"""Intermittency and volatility-clustering diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fractalfinance.estimators.structure import StructureFunction

__all__ = [
    "estimate_mu",
    "clustering_coefficient",
    "IntermittencyResult",
    "ClusteringResult",
]


@dataclass(slots=True)
class IntermittencyResult:
    mu: float
    H: float
    zeta_p: float
    p: float
    delta_alpha: float


def estimate_mu(
    series: Sequence[float],
    *,
    p: float = 2.0,
    min_scale: int = 1,
    max_scale: int | None = None,
    n_scales: int = 12,
    from_levels: bool = False,
) -> IntermittencyResult:
    """Estimate the intermittency exponent ``μ`` from structure functions."""

    if p <= 1:
        raise ValueError("p must exceed 1 to identify μ")
    est = StructureFunction(
        series,
        q=np.array([1.0, float(p)]),
        min_scale=min_scale,
        max_scale=max_scale,
        n_scales=n_scales,
        from_levels=from_levels,
    )
    est.fit()
    res = est.result_
    assert res is not None
    H = float(res.zeta[1.0])
    zeta_p = float(res.zeta[float(p)])
    mu = 2.0 * (p * H - zeta_p) / (p * (p - 1.0))
    return IntermittencyResult(mu=mu, H=H, zeta_p=zeta_p, p=p, delta_alpha=res.delta_alpha)


@dataclass(slots=True)
class ClusteringResult:
    lags: np.ndarray
    coefficient: np.ndarray
    threshold: float


def clustering_coefficient(
    series: Sequence[float],
    *,
    threshold: float,
    max_lag: int = 10,
    from_levels: bool = False,
) -> ClusteringResult:
    """Compute volatility-clustering coefficient ``C(τ)`` for ``τ=1..max_lag``."""

    data = np.asarray(series, dtype=float)
    if from_levels:
        data = np.diff(data)

    if data.size <= max_lag:
        raise ValueError("Series too short for requested lags")

    indicator = (np.abs(data) > threshold).astype(float)
    base_prob = indicator.mean()
    if base_prob == 0:
        raise RuntimeError("Threshold too high: no exceedances observed")

    coeffs = []
    lags = np.arange(1, max_lag + 1)
    for tau in lags:
        lead = indicator[tau:]
        lagged = indicator[:-tau]
        prob_cond = np.mean(lead * lagged)
        prob_lag = lagged.mean()
        if prob_lag == 0:
            coeffs.append(np.nan)
            continue
        coeffs.append(prob_cond / prob_lag / base_prob)

    return ClusteringResult(
        lags=lags,
        coefficient=np.asarray(coeffs, dtype=float),
        threshold=float(threshold),
    )
