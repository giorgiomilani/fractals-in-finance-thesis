"""Wavelet variance, spectrum and Hurst estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pywt

from fractalfinance.estimators._base import BaseEstimator

__all__ = [
    "detail_coeffs",
    "wavelet_variance",
    "logscale_diagram",
    "estimate_hurst",
    "WaveletHurstResult",
]


def detail_coeffs(
    series: Sequence[float],
    *,
    wavelet: str = "db2",
    level: int | None = None,
    mode: str = "periodization",
) -> list[np.ndarray]:
    """Return detail coefficients ordered from fine to coarse scales."""

    data = np.asarray(series, dtype=float)
    coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)
    details = coeffs[1:][::-1]
    return [np.asarray(c, dtype=float) for c in details]


def wavelet_variance(details: Sequence[np.ndarray]) -> np.ndarray:
    """Compute sample variance of detail coefficients at each level."""

    return np.array([np.mean(d**2) for d in details], dtype=float)


def logscale_diagram(details: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return dyadic scale indices and log2 variances."""

    variances = wavelet_variance(details)
    levels = np.arange(1, len(details) + 1)
    return levels, np.log2(variances)


def _newey_west(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    T, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    S = np.zeros((p, p), dtype=float)
    for k in range(lags + 1):
        weight = 1.0 if k == 0 else 1.0 - k / (lags + 1)
        if k == 0:
            Z = X * resid[:, None]
            S += Z.T @ Z
        else:
            Z1 = X[k:] * resid[k:, None]
            Z0 = X[:-k] * resid[:-k, None]
            S += weight * (Z1.T @ Z0 + Z0.T @ Z1)
    S /= T
    return XtX_inv @ S @ XtX_inv


@dataclass(slots=True)
class WaveletHurstResult:
    H: float
    slope: float
    intercept: float
    stderr: float
    levels: np.ndarray
    log2_variance: np.ndarray
    slice_used: slice


def estimate_hurst(
    series: Sequence[float],
    *,
    wavelet: str = "db2",
    level: int | None = None,
    min_level: int = 1,
    max_level: int | None = None,
    kind: str = "fgn",
    from_levels: bool = False,
    hac_lags: int | None = None,
) -> WaveletHurstResult:
    """Estimate the Hurst exponent via wavelet-logscale regression."""

    data = np.asarray(series, dtype=float)
    if from_levels:
        data = np.diff(data)
    details = detail_coeffs(data, wavelet=wavelet, level=level)
    levels = np.arange(1, len(details) + 1)

    if max_level is None:
        max_level = len(details)
    sel = slice(min_level - 1, max_level)
    sel_levels = levels[sel]
    sel_details = details[sel]

    var2 = np.log2(wavelet_variance(sel_details))
    mask = np.isfinite(var2)
    sel_levels = sel_levels[mask]
    var2 = var2[mask]
    if sel_levels.size < 2:
        raise RuntimeError("Not enough valid wavelet scales for regression")

    xs = sel_levels.astype(float)
    ys = var2
    sl = BaseEstimator._best_range(
        xs, ys, BaseEstimator.min_points, BaseEstimator.r2_thresh
    )
    xs_reg = xs[sl]
    ys_reg = ys[sl]

    X = np.column_stack((np.ones(xs_reg.size), xs_reg))
    beta, *_ = np.linalg.lstsq(X, ys_reg, rcond=None)
    intercept, slope = beta

    resid = ys_reg - (intercept + slope * xs_reg)
    if hac_lags is None:
        hac_lags = min(2, xs_reg.size - 1)
    hac_lags = max(hac_lags, 0)
    var_beta = _newey_west(X, resid, hac_lags)
    stderr = float(np.sqrt(max(var_beta[1, 1], 0.0)))

    if kind.lower() == "fgn":
        H = 0.5 * (slope + 1.0)
    elif kind.lower() == "fbm":
        H = 0.5 * (slope - 1.0)
    else:
        raise ValueError("kind must be 'fgn' or 'fbm'")

    return WaveletHurstResult(
        H=float(H),
        slope=float(slope),
        intercept=float(intercept),
        stderr=stderr,
        levels=sel_levels,
        log2_variance=var2,
        slice_used=sl,
    )
