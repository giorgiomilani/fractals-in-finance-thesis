"""
Public API re‑exports for ``fractalfinance.models``.
Keeps backwards‑compatibility with the test‑suite while avoiding
circular imports.
"""

from __future__ import annotations

# ── volatility benchmarks ──────────────────────────────────────────────
try:  # optional volatility benchmarks (statsmodels dependency)
    from .benchmarks import GARCH, HAR
except Exception:  # pragma: no cover - dependency missing
    GARCH = HAR = None

# ── fractional Brownian motion ─────────────────────────────────────────
from .fbm import fbm

# ── Markov–switching multifractal (functional + OO) ────────────────────
from .msm import MSMParams, loglik as msm_loglik, simulate as msm_simulate, fit as msm_fit
from .msm_class import MSM, Params as MSMParamsClass

# ── MMAR (multifractal measure & FBM)  ─────────────────────────────────
from .mmar import CascadeParams
from .mmar import simulate as _mmar_simulate  # real implementation

# ── Multifractal random walk (MRW) ──────────────────────────────────────
from .mrw import MRWParams
from .mrw import simulate as mrw_simulate
from .mrw import structure_exponent as mrw_structure_exponent

# ─── UNIVERSAL MMAR SIMULATION WRAPPER ────────────────────────────────
def _wrapper_mmar_simulate(
    n: int,
    H: float,
    *,
    cascade: CascadeParams | None = None,
    seed: int | None = None,
    **loose_kwargs,
):
    """
    Accepts either…

        • mmar_simulate(N, H, cascade=CascadeParams(...))
        • mmar_simulate(N, H, m_L=…, m_H=…, depth=…, seed=…)
        • mmar_simulate(N, H, cascade=…, seed=99)  # overrides seed

    Raises if both `cascade=` *and* loose multipliers are supplied.
    """
    # ── CASE A – user passed a CascadeParams object ───────────────────
    if cascade is not None:
        if loose_kwargs:
            raise TypeError(
                "Provide EITHER `cascade=` OR loose parameters (m_L, m_H, …), "
                "not both."
            )
        if seed is not None and seed != cascade.seed:
            cascade = CascadeParams(
                m_L=cascade.m_L,
                m_H=cascade.m_H,
                depth=cascade.depth,
                seed=seed,
            )
        return _mmar_simulate(n=n, H=H, cascade=cascade)

    # ── CASE B – flat keyword list ------------------------------------
    if seed is not None:
        loose_kwargs.setdefault("seed", seed)

    cascade = CascadeParams(**loose_kwargs)
    return _mmar_simulate(n=n, H=H, cascade=cascade)


# public API (new + legacy)
mmar_simulate = _wrapper_mmar_simulate
simulate = _wrapper_mmar_simulate  # legacy alias

# ---------------------------------------------------------------------
__all__ = [
    # FBM
    "fbm",
    # MSM functional interface
    "MSMParams",
    "msm_simulate",
    "msm_loglik",
    "msm_fit",
    # MSM OO class
    "MSM",
    "MSMParamsClass",
    # MMAR
    "CascadeParams",
    "mmar_simulate",
    "simulate",
    # MRW
    "MRWParams",
    "mrw_simulate",
    "mrw_structure_exponent",
    # Bench‑mark volatility models
    "GARCH",
    "HAR",
]
