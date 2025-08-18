"""
Back‑testing of VaR / ES
========================
kupiec          – unconditional coverage (binomial tail p‑value)
christoffersen  – conditional coverage LR‑cc
acerbi_szekely  – ES back‑test Z‑score
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom, chi2


# ------------------------------------------------------------------ #
def kupiec(viol: np.ndarray, p: float) -> float:
    """
    Kupiec unconditional coverage test – p‑value via binomial tail.
    Works well for small samples where LR‑uc can be extreme.
    """
    viol = np.asarray(viol, dtype=int)
    T = len(viol)
    N = viol.sum()
    # two‑sided tail probability under H₀ (expected prob = 1‑p)
    prob = (
        binom.cdf(N, T, 1 - p) if N < (T * (1 - p)) else 1 - binom.cdf(N - 1, T, 1 - p)
    )
    return float(prob)


# ------------------------------------------------------------------ #
def christoffersen(viol: np.ndarray, p: float) -> float:
    """
    Christoffersen independence + coverage test (LR‑cc) – p‑value.
    """
    viol = np.asarray(viol, dtype=int)
    T00 = np.sum((viol[:-1] == 0) & (viol[1:] == 0))
    T01 = np.sum((viol[:-1] == 0) & (viol[1:] == 1))
    T10 = np.sum((viol[:-1] == 1) & (viol[1:] == 0))
    T11 = np.sum((viol[:-1] == 1) & (viol[1:] == 1))

    pi0 = T01 / max(T00 + T01, 1)
    pi1 = T11 / max(T10 + T11, 1)
    pi = (T01 + T11) / max(T00 + T01 + T10 + T11, 1)

    LR = -2.0 * (
        T00 * np.log(1 - pi0 + 1e-12)
        + T01 * np.log(pi0 + 1e-12)
        + T10 * np.log(1 - pi1 + 1e-12)
        + T11 * np.log(pi1 + 1e-12)
        - ((T00 + T01) * np.log(1 - pi + 1e-12) + (T10 + T11) * np.log(pi + 1e-12))
    )
    return float(1.0 - chi2.cdf(LR, df=2))


# ------------------------------------------------------------------ #
def acerbi_szekely(loss: np.ndarray, var: np.ndarray, es: np.ndarray) -> float:
    """
    Acerbi‑Székely (2014) ES back‑test under i.i.d. assumption.
    Returns Z‑score (approx N(0,1) under H₀).
    """
    loss = np.asarray(loss, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)

    hits = (loss > var).astype(float)
    w = hits / (1.0 - 0.99)  # weight for 99 % tail
    z = (loss - es) * w
    zbar = z.mean()
    s = z.std(ddof=1)
    return float(zbar / (s / np.sqrt(len(loss))))
