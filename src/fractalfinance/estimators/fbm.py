"""Parameter estimation utilities for fractional Brownian motion."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, optimize
import pywt

__all__ = [
    "fbm_covariance",
    "fbm_mle",
    "fbm_wavelet_whittle",
]


def fbm_covariance(H: float, n: int, length: float = 1.0) -> np.ndarray:
    """Toeplitz covariance matrix for fractional Gaussian noise."""

    if not (0 < H < 1):
        raise ValueError("H must lie in (0, 1).")
    if n <= 0:
        raise ValueError("n must be positive.")

    dt = float(length) / float(n)
    k = np.arange(n, dtype=float)
    gamma = 0.5 * ((k + 1) ** (2 * H) - 2 * (k ** (2 * H)) + np.abs(k - 1) ** (2 * H))
    gamma *= dt ** (2 * H)
    prefactor = 1.0 / (4.0 * n**2)
    cov = prefactor * gamma[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    return cov


def _prepare_series(series: ArrayLike, from_levels: bool) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if from_levels:
        x = np.diff(x, n=1)
    x = x[np.isfinite(x)]
    if x.size < 4:
        raise ValueError("Series too short for estimation.")
    return x - x.mean()


def fbm_mle(
    series: ArrayLike,
    *,
    from_levels: bool = True,
    length: float = 1.0,
    bounds: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, float]:
    """Maximum-likelihood estimates of (H, sigma) for fBm increments."""

    x = _prepare_series(series, from_levels)
    n = x.size

    def negloglik(H: float) -> float:
        if not (bounds[0] < H < bounds[1]):
            return float("inf")
        try:
            cov = fbm_covariance(H, n, length=length)
            cho = linalg.cho_factor(cov, lower=True, check_finite=False)
            y = linalg.cho_solve(cho, x, check_finite=False)
            rss = float(x @ y)
            logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
        except linalg.LinAlgError:
            return float("inf")
        return 0.5 * (n * np.log(rss / n) + logdet + n * np.log(2 * np.pi) + n)

    res = optimize.minimize_scalar(
        negloglik,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError("MLE optimisation for fBm failed to converge.")

    H_hat = float(res.x)
    cov_hat = fbm_covariance(H_hat, n, length=length)
    cho = linalg.cho_factor(cov_hat, lower=True, check_finite=False)
    y = linalg.cho_solve(cho, x, check_finite=False)
    sigma2 = float((x @ y) / n)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    loglik = -float(res.fun)
    return {"H": H_hat, "sigma": sigma, "loglik": loglik}


def fbm_wavelet_whittle(
    series: ArrayLike,
    *,
    from_levels: bool = True,
    wavelet: str = "db2",
    levels: int | None = None,
    length: float = 1.0,
) -> Dict[str, float]:
    """Wavelet Whittle estimator for the Hurst exponent."""

    x = _prepare_series(series, from_levels)
    wave = pywt.Wavelet(wavelet)
    if levels is None:
        levels = pywt.dwt_max_level(len(x), wave.dec_len)
    if levels < 1:
        raise ValueError("levels must be at least 1.")

    coeffs = pywt.wavedec(x, wave, mode="periodization", level=levels)
    details = coeffs[1:][::-1]
    scales = 2.0 ** np.arange(1, len(details) + 1, dtype=float)
    vars_ = np.array([np.var(c, ddof=1) for c in details])
    weights = np.array([c.size for c in details], dtype=float)
    mask = (vars_ > 0) & np.isfinite(vars_)
    if mask.sum() < 2:
        raise RuntimeError("Insufficient valid wavelet scales for Whittle estimation.")

    log_scales = np.log2(scales[mask])
    log_vars = np.log2(vars_[mask])
    weights = weights[mask]
    slope, intercept = np.polyfit(log_scales, log_vars, 1, w=weights)
    H = float((slope + 1.0) / 2.0)
    n = x.size
    bias = 0.5 / np.log2(n) if n > 2 else 0.0
    H = float(np.clip(H - bias, 0.0, 1.0))
    cov = fbm_covariance(max(H, 0.01), n, length=length)
    cho = linalg.cho_factor(cov, lower=True, check_finite=False)
    y = linalg.cho_solve(cho, x, check_finite=False)
    sigma2 = float((x @ y) / n)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    return {
        "H": H,
        "sigma": sigma,
        "intercept": float(intercept),
        "levels": int(len(details)),
    }

