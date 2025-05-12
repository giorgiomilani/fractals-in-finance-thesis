"""
Multifractal Model of Asset Returns (MMAR)
==========================================

theta, X, r = mmar_simulate(n, H=0.7, …)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from dataclasses import replace           # add near the top of the file …

import numpy as np

# ════════════════════════════════════════════════════════════════════════
# 1.  BINOMIAL MULTIPLICATIVE CASCADE
# ════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class CascadeParams:
    m_L: float = 0.6
    m_H: float = 1.4
    depth: int = 9            # 2**depth finest‑scale bins
    seed: int | None = None


def _multiplicative_cascade(p: CascadeParams) -> np.ndarray:
    """Return normalised μ‑measure on 2**depth bins."""
    rng = np.random.default_rng(p.seed)
    mu = np.ones(1, float)
    for _ in range(p.depth):
        mu = np.repeat(mu, 2)
        mu *= rng.choice((p.m_L, p.m_H), size=mu.size)
    return mu / mu.sum()


# ════════════════════════════════════════════════════════════════════════
# 2.  DAVIES–HARTE fractional‑Gaussian noise (unit variance)
# ════════════════════════════════════════════════════════════════════════
def _fgn_davies_harte(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    k = np.arange(n)
    gamma = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))
    g = np.concatenate([gamma, [0.0], gamma[1:][::-1]])
    eigs = np.fft.fft(g).real
    if (eigs < 0).any():
        raise ValueError("n too small – covariance embedding not PD")

    Z = np.empty(2 * n, np.complex128)
    Z[0] = rng.normal() * np.sqrt(eigs[0] / (2 * n))
    Z[n] = rng.normal() * np.sqrt(eigs[n] / (2 * n))
    W = rng.normal(size=n - 1) + 1j * rng.normal(size=n - 1)
    Z[1:n] = W * np.sqrt(eigs[1:n] / (4 * n))
    Z[n + 1 :] = np.conj(Z[1:n][::-1])

    fgn = np.fft.ifft(Z).real[:n]
    return fgn / fgn.std(ddof=0)


# ════════════════════════════════════════════════════════════════════════
# 3.  PUBLIC SIMULATOR
# ════════════════════════════════════════════════════════════════════════
def mmar_simulate(
    n: int,
    H: float = 0.7,
    *,
    cascade: CascadeParams | None = None,
    seed: int | None = None,
    **loose,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accept either loose kwargs (m_L=…, m_H=…, depth=…, seed=…)
    **or** an explicit `cascade=CascadeParams(...)`.
    """
    # ── resolve cascade params ────────────────────────────────────────
    if cascade is None:
        if seed is not None:
            loose.setdefault("seed", seed)
        cascade = CascadeParams(**loose)
    else:
        if loose:
            raise TypeError("Pass EITHER loose kwargs OR cascade=, not both.")
        if (seed is not None) and (seed != cascade.seed):
            cascade = replace(cascade, seed=seed)

    rng = np.random.default_rng(seed)

    # ── multifractal clock θ(t) ───────────────────────────────────────
    mu = _multiplicative_cascade(cascade)
    k = n // mu.size
    if k == 0:
        raise ValueError("n must be at least 2**depth")
    theta = np.repeat(mu, k).cumsum()
    theta = theta[:n] / theta.sum()        # θ ∈ [0,1]

    # ── FBM on irregular clock ───────────────────────────────────────
    fgn = _fgn_davies_harte(n, H, rng)
    X = np.cumsum(np.interp(np.linspace(0, 1, n), theta, fgn))
    r = np.diff(np.insert(X, 0, X[0])) - np.diff(np.insert(X, 0, X[0])).mean()
    return theta, X, r


# legacy alias used by notebooks / tests
simulate = mmar_simulate

# ════════════════════════════════════════════════════════════════════════
# 4.  CALIBRATION UTILITIES
# ════════════════════════════════════════════════════════════════════════
def tau_from_path(
    x: np.ndarray,
    q: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Quick τ(q) estimator with log‑moment safeguard."""
    s = np.logspace(3, np.log10(len(x) // 4), 10).astype(int)
    tau = []
    for qv in q:
        moments = []
        for w in s:
            rs = np.abs(np.diff(x, w))
            rs = np.clip(rs, eps, None)
            log_rs = np.log(rs)
            moments.append(np.exp(qv * log_rs.mean()))
        moments = np.asarray(moments)
        mask = np.isfinite(moments) & (moments > 0)
        if mask.sum() < 2:
            tau.append(np.nan)
            continue
        tau.append(np.polyfit(np.log(s[mask]), np.log(moments[mask]), 1)[0])
    return np.asarray(tau)


def calibrate(
    empirical_path: np.ndarray,
    H: float,
    *,
    search_mH: tuple[float, float, int] = (1.2, 2.0, 5),
    depth: int = 8,
    q: np.ndarray = np.linspace(-4, 4, 9),
    seed: int = 0,
    n_rep: int = 5,                    # ← average over a few replicas
) -> CascadeParams:
    """
    Grid‑search (m_L, m_H) to minimise RMSE between empirical and
    simulated τ(q), averaging over *n_rep* independent simulations.
    """
    tau_emp = tau_from_path(empirical_path, q)

    best: CascadeParams | None = None
    best_err = np.inf

    mH_grid = np.linspace(*search_mH)
    for mH in mH_grid:
        cand = CascadeParams(m_L=1.0 / mH, m_H=mH, depth=depth, seed=seed)

        # ------- average τ(q) across replicas ------------------------
        tau_sims = []
        for rep in range(n_rep):
            _, Xsim, _ = simulate(
                len(empirical_path),
                H,
                cascade=cand,
                seed=seed + rep,           # independent RNG stream
            )
            tau_sims.append(tau_from_path(Xsim, q))
        tau_sim = np.nanmean(tau_sims, axis=0)

        # ------- RMSE on common, finite points -----------------------
        mask = np.isfinite(tau_emp) & np.isfinite(tau_sim)
        if mask.sum() < 2:
            continue
        err = np.sqrt(np.mean((tau_emp[mask] - tau_sim[mask]) ** 2))

        if err < best_err:
            best_err, best = err, cand

    if best is None:
        raise RuntimeError(
            "Calibration failed: no candidate produced ≥2 finite τ(q). "
            "Try widening `search_mH`."
        )
    return best
