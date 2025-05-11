"""
Multifractal Detrended Fluctuation Analysis (MFDFA)
===================================================

Implements the Kantelhardt et al. (2002) DFA‑1 algorithm and returns the
singularity spectrum (α, f(α)).

Key points
----------
* Analyse **increments** of the series to avoid FBM bias.
* Skip scales with fewer than two windows.
* Ignore (nan) zero variances to prevent log/zero warnings.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from ._base import BaseEstimator


# ─── compatibility shim ────────────────────────────────────────────────
# NumPy 2.0 removed ndarray.ptp; re‑add it so legacy code & tests work.
if not hasattr(np.ndarray, "ptp"):
    def _ptp(self, axis=None, out=None, keepdims=False):
        return np.ptp(self, axis=axis, out=out, keepdims=keepdims)
    setattr(np.ndarray, "ptp", _ptp)

import pandas as _pd  # local alias to avoid confusion with main pd
if not hasattr(_pd.Series, "ptp"):
    def _series_ptp(self):
        """Peak‑to‑peak (max‑min) – NumPy compatible helper."""
        return float(self.max() - self.min())
    _pd.Series.ptp = _series_ptp           # type: ignore[attr-defined]
# ───────────────────────────────────────────────────────────────────────


class MFDFA(BaseEstimator):
    def __init__(
        self,
        series,
        *,
        q: np.ndarray | None = None,
        min_scale: int = 8,
        max_scale: int | None = None,
        n_scales: int = 20,
    ):
        super().__init__(series)
        self.q = np.asarray(q if q is not None else np.arange(-4, 5), dtype=float)
        self.min_scale = int(min_scale)
        self.max_scale = None if max_scale is None else int(max_scale)
        self.n_scales = int(n_scales)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _F2(profile: np.ndarray, s: int) -> np.ndarray:
        """Variance of linear‑detrended windows at scale *s* (length k)."""
        N = len(profile)
        k = N // s
        if k < 2:
            return np.empty(0)

        windows = profile[: k * s].reshape(k, s)
        t = np.arange(s)
        F2 = np.empty(k)
        for i, w in enumerate(windows):
            a, b = np.polyfit(t, w, 1)
            F2[i] = np.mean((w - (a * t + b)) ** 2)
        return F2

    # ------------------------------------------------------------------ #
    def fit(self):
        # 1. Convert to ndarray and take **increments**
        x_raw = self.series.to_numpy(dtype=float) if isinstance(self.series, pd.Series) else np.asarray(self.series, dtype=float)
        x = np.diff(x_raw)
        N = len(x)

        # 2. Profile
        profile = np.cumsum(x - x.mean())

        # 3. Log‑spaced scale grid
        max_scale = self.max_scale or N // 4
        scales = np.unique(
            np.floor(
                np.logspace(
                    np.log10(self.min_scale),
                    np.log10(max_scale),
                    num=self.n_scales,
                )
            ).astype(int)
        )

        # 4. Fluctuation functions F(q,s) with safe numerics
        Hq: Dict[float, float] = {}
        for q in self.q:
            logF: List[float] = []
            logS: List[float] = []
            for s in scales:
                F2 = self._F2(profile, s)
                if F2.size == 0:
                    continue
                F2 = F2[F2 > 0]  # drop zero variances
                if F2.size == 0:
                    continue

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    if q == 0:  # special‑case q→0 limit
                        F = np.exp(0.5 * np.nanmean(np.log(F2)))
                    else:
                        F = np.nanmean(F2 ** (q / 2.0)) ** (1.0 / q)

                if np.isfinite(F) and F > 0:
                    logF.append(np.log(F))
                    logS.append(np.log(s))

            if len(logF) < 2:  # not enough scales for regression
                continue
            h, _ = np.polyfit(logS, logF, 1)
            Hq[q] = h

        # 5. Singularity spectrum τ(q) → α,f(α)
        qs = np.array(sorted(Hq.keys()))
        hq = np.array([Hq[q] for q in qs])
        tau_q = qs * hq - 1
        alpha = np.gradient(tau_q, qs)
        f_alpha = qs * alpha - tau_q

        # store as Series so .ptp() works with NumPy 2.x
        self.result_ = {
            "q": qs,
            "h": pd.Series(hq, index=qs, name="h(q)"),
            "tau": pd.Series(tau_q, index=qs, name="tau(q)"),
            "alpha": pd.Series(alpha, index=qs, name="alpha"),
            "f_alpha": pd.Series(f_alpha, index=qs, name="f(alpha)"),
        }
        return self
