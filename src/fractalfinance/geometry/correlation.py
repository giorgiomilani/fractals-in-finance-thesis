"""Correlation-dimension diagnostics with surrogate validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "time_delay_embedding",
    "correlation_integral",
    "correlation_dimension",
    "iaaft_surrogates",
    "estimate",
    "CorrelationDimensionResult",
]


def time_delay_embedding(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Embed a scalar time series into ``m`` dimensions with delay ``tau``."""

    x = np.asarray(series, dtype=float)
    if m <= 0 or tau <= 0:
        raise ValueError("Embedding dimension and delay must be positive")
    N = x.size - (m - 1) * tau
    if N <= 0:
        raise ValueError("Series too short for the requested embedding")
    emb = np.empty((N, m), dtype=float)
    for i in range(m):
        emb[:, i] = x[i * tau : i * tau + N]
    return emb


def correlation_integral(
    embedded: np.ndarray,
    radii: np.ndarray,
    *,
    theiler: int = 0,
) -> np.ndarray:
    """Compute Grassberger–Procaccia correlation integral ``C(r)``."""

    radii = np.asarray(radii, dtype=float)
    if np.any(radii <= 0):
        raise ValueError("Radii must be positive")

    n = embedded.shape[0]
    if n < 2:
        raise ValueError("At least two points required for correlation integral")

    dists = squareform(pdist(embedded, metric="euclidean"))
    if theiler > 0:
        idx = np.arange(n)
        mask = np.abs(idx[:, None] - idx[None, :]) <= theiler
        dists[mask] = np.inf

    upper = dists[np.triu_indices(n, k=1)]
    counts = np.array([(upper < r).sum() for r in radii], dtype=float)
    norm = n * (n - 1) / 2.0
    return counts / norm


def correlation_dimension(
    radii: np.ndarray,
    C_r: np.ndarray,
    *,
    min_points: int = 6,
    r2_thresh: float = 0.97,
) -> Tuple[float, slice]:
    """Estimate ``D_2`` via linear regression on the log–log correlation plot."""

    log_r = np.log(radii)
    log_C = np.log(C_r)
    mask = np.isfinite(log_C)
    log_r = log_r[mask]
    log_C = log_C[mask]
    if log_r.size < 2:
        raise RuntimeError("Insufficient finite values for correlation dimension")

    n = log_r.size
    best_slice = slice(0, n)
    best_score: Tuple[int, float] | None = None

    for start in range(0, n - min_points + 1):
        for stop in range(start + min_points, n + 1):
            xs = log_r[start:stop]
            ys = log_C[start:stop]
            slope, intercept = np.polyfit(xs, ys, 1)
            fitted = slope * xs + intercept
            resid = ys - fitted
            ss_res = float(np.sum(resid**2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            length = stop - start
            score = (1 if r2 >= r2_thresh else 0, length)
            if best_score is None or score > best_score:
                best_score = score
                best_slice = slice(start, stop)

    sl = best_slice
    slope, _ = np.polyfit(log_r[sl], log_C[sl], 1)
    return float(slope), sl


def iaaft_surrogates(
    series: np.ndarray,
    n_surrogates: int,
    *,
    seed: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    """Generate IAAFT surrogates preserving the spectrum and marginal CDF."""

    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError("IAAFT expects a one-dimensional series")

    rng = np.random.default_rng(seed)
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)

    surrogates = np.empty((n_surrogates, x.size), dtype=float)
    for s in range(n_surrogates):
        y = rng.permutation(x)
        prev_spec = None
        for _ in range(max_iter):
            fft_y = np.fft.rfft(y)
            magnitude = np.abs(fft_y)
            phase = np.ones_like(fft_y)
            nonzero = magnitude > 0
            phase[nonzero] = fft_y[nonzero] / magnitude[nonzero]
            y = np.fft.irfft(target_amp * phase, n=x.size)

            ranks = np.argsort(np.argsort(y))
            y = sorted_x[ranks]

            spec = np.abs(np.fft.rfft(y))
            if prev_spec is not None and np.linalg.norm(spec - prev_spec) < tol:
                break
            prev_spec = spec
        surrogates[s] = y
    return surrogates


@dataclass(slots=True)
class CorrelationDimensionResult:
    radii: np.ndarray
    C_r: np.ndarray
    D2: float
    slope_range: slice
    surrogates_D2: np.ndarray | None
    embedding: np.ndarray


def estimate(
    series: np.ndarray,
    *,
    m: int,
    tau: int,
    radii: np.ndarray,
    theiler: int = 0,
    n_surrogates: int = 0,
    seed: int | None = None,
) -> CorrelationDimensionResult:
    """Full correlation-dimension workflow with optional surrogate tests."""

    emb = time_delay_embedding(series, m=m, tau=tau)
    C_r = correlation_integral(emb, radii, theiler=theiler)
    D2, sl = correlation_dimension(radii, C_r)

    sur_D2 = None
    if n_surrogates > 0:
        sur = iaaft_surrogates(series, n_surrogates, seed=seed)
        sur_D2 = np.empty(n_surrogates, dtype=float)
        for i in range(n_surrogates):
            emb_s = time_delay_embedding(sur[i], m=m, tau=tau)
            C_s = correlation_integral(emb_s, radii, theiler=theiler)
            D2_s, _ = correlation_dimension(radii, C_s)
            sur_D2[i] = D2_s

    return CorrelationDimensionResult(
        radii=np.asarray(radii, dtype=float),
        C_r=C_r,
        D2=D2,
        slope_range=sl,
        surrogates_D2=sur_D2,
        embedding=emb,
    )
