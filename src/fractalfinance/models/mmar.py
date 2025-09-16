from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import numpy as np

from .fbm import fbm

__all__ = [
    "CascadeParams",
    "generate_cascade",
    "simulate",
    "mmar_simulate",
    "calibrate",
]


@dataclass(frozen=True)
class CascadeParams:
    """Parameters controlling the dyadic multiplicative cascade."""

    m_L: float = 0.6
    m_H: float = 1.6
    depth: int = 8
    seed: int | None = None

    def __post_init__(self) -> None:  # noqa: D401 - validation helper
        if self.m_L <= 0 or self.m_H <= 0:
            raise ValueError("Cascade multipliers must be strictly positive.")
        if not np.isfinite(self.m_L) or not np.isfinite(self.m_H):
            raise ValueError("Cascade multipliers must be finite numbers.")
        if self.m_L >= self.m_H:
            raise ValueError("Require m_L < m_H for a meaningful cascade.")
        if self.depth < 0:
            raise ValueError("Cascade depth must be non-negative.")


def _binary_cascade(params: CascadeParams) -> np.ndarray:
    """Generate a dyadic cascade with unit-mean weights."""

    n_bins = 1 << max(int(params.depth), 0)
    if n_bins == 0:
        return np.ones(1, dtype=float)

    weights = np.ones(n_bins, dtype=float)
    rng = np.random.default_rng(params.seed)
    segment = n_bins
    for _ in range(int(params.depth)):
        segment //= 2
        if segment == 0:
            break
        for start in range(0, n_bins, 2 * segment):
            left = slice(start, start + segment)
            right = slice(start + segment, start + 2 * segment)
            m_left = rng.choice((params.m_L, params.m_H))
            m_right = rng.choice((params.m_L, params.m_H))
            weights[left] *= m_left
            weights[right] *= m_right
    weights /= weights.mean()
    return weights


def generate_cascade(params: CascadeParams) -> np.ndarray:
    """Public helper returning the raw cascade weights (mean-one)."""

    return _binary_cascade(params)


def _trading_clock(weights: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample cascade weights into *n* time steps and their increments."""

    n_bins = len(weights)
    cdf = np.concatenate([[0.0], np.cumsum(weights, dtype=float)])
    cdf /= cdf[-1]
    grid = np.linspace(0.0, 1.0, n + 1)
    x_src = np.linspace(0.0, 1.0, n_bins + 1)
    theta = np.interp(grid, x_src, cdf)
    delta = np.diff(theta)
    delta = np.clip(delta, 1e-12, None)
    return theta[1:], delta


def _resolve_params(
    cascade: CascadeParams | None,
    m_L: float,
    m_H: float,
    depth: int,
    seed: int | None,
) -> CascadeParams:
    if cascade is not None:
        if seed is not None and seed != cascade.seed:
            cascade = replace(cascade, seed=seed)
        return cascade
    return CascadeParams(m_L=m_L, m_H=m_H, depth=depth, seed=seed)


def simulate(
    n: int,
    H: float,
    *,
    cascade: CascadeParams | None = None,
    m_L: float = 0.6,
    m_H: float = 1.6,
    depth: int = 8,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an MMAR path with a multiplicative trading clock.

    Parameters
    ----------
    n:
        Number of increments to generate.
    H:
        Hurst exponent of the underlying fractional Brownian motion.
    cascade:
        Optional :class:`CascadeParams` instance controlling the
        multiplicative cascade.  When omitted, ``m_L``, ``m_H`` and
        ``depth`` are used.
    seed:
        Seed applied to both the cascade generator and the Davies–Harte
        FGN routine for reproducibility.

    Returns
    -------
    theta_speed, X, r:
        ``theta_speed`` is the local trading speed (unit mean) obtained
        from the cascade, ``X`` the integrated MMAR path and ``r`` the
        subordinated returns.
    """

    params = _resolve_params(cascade, m_L, m_H, depth, seed)
    weights = _binary_cascade(params)
    _, delta_theta = _trading_clock(weights, n)

    fgn = fbm(H=H, n=n, kind="increment", seed=seed)
    base_dt = 1.0 / n
    local_scale = (delta_theta / base_dt) ** H
    r = fgn * local_scale
    X = np.cumsum(r)
    theta_speed = delta_theta / base_dt
    return theta_speed, X, r


# Legacy alias used in tests
mmar_simulate = simulate


def _feature_vector(returns: np.ndarray) -> np.ndarray:
    """Feature vector capturing tail thickness and volatility clustering."""

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        raise ValueError("Series too short for MMAR calibration.")
    r = r - r.mean()
    std = r.std()
    if std == 0:
        return np.array([3.0, 0.0, 0.0], dtype=float)
    r /= std
    kurt = float(np.mean(r**4))
    abs_r = np.abs(r)
    if abs_r.size < 2:
        acf = 0.0
    else:
        acf = float(np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1])
        if not np.isfinite(acf):
            acf = 0.0
    log_std = float(np.std(np.log(abs_r + 1e-8)))
    return np.array([kurt, acf, log_std], dtype=float)


def calibrate(
    series: np.ndarray,
    H: float,
    *,
    depth: int = 8,
    search_mL: Tuple[float, float, int] = (0.4, 0.9, 6),
    search_mH: Tuple[float, float, int] = (1.2, 2.5, 6),
    n_trials: int = 4,
    seed: int | None = None,
    from_levels: bool = True,
) -> CascadeParams:
    """Grid-search calibration for cascade multipliers.

    The routine matches a small feature vector (kurtosis, absolute-return
    autocorrelation and log-amplitude dispersion) between the data and
    simulated MMAR paths.
    """

    if from_levels:
        returns = np.diff(series, n=1)
    else:
        returns = np.asarray(series, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 8:
        raise ValueError("Need at least 8 increments for calibration.")

    target = _feature_vector(returns)
    grid_mL = np.linspace(*search_mL)
    grid_mH = np.linspace(*search_mH)
    best_params: CascadeParams | None = None
    best_error = float("inf")

    if seed is None:
        trial_seeds: Sequence[int | None] = [None] * int(max(n_trials, 1))
    else:
        base = np.random.SeedSequence(seed)
        trial_seeds = [int(s.generate_state(1)[0]) for s in base.spawn(int(max(n_trials, 1)))]

    for mL in grid_mL:
        if mL <= 0:
            continue
        for mH in grid_mH:
            if mH <= 0 or mL >= mH:
                continue
            feats: list[np.ndarray] = []
            for trial_seed in trial_seeds:
                params = CascadeParams(m_L=float(mL), m_H=float(mH), depth=int(depth), seed=trial_seed)
                _, _, r_sim = simulate(
                    len(returns),
                    H,
                    cascade=params,
                    seed=trial_seed,
                )
                feats.append(_feature_vector(r_sim))
            mean_feat = np.mean(feats, axis=0)
            # relative weighting of components
            denom = np.where(target != 0, np.abs(target), 1.0)
            error = float(np.mean(((mean_feat - target) / denom) ** 2))
            if error < best_error:
                best_error = error
                best_params = CascadeParams(
                    m_L=float(mL),
                    m_H=float(mH),
                    depth=int(depth),
                    seed=seed,
                )

    if best_params is None:
        raise RuntimeError("Calibration failed to locate feasible parameters.")
    return best_params

