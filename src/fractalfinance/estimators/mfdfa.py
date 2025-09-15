"""
Multifractal Detrended Fluctuation Analysis (MFDFA)
===================================================

Implements the Kantelhardt et al. (2002) procedure (DFA‑1 detrending)
and returns the singularity spectrum (α, f(α)).

Key implementation notes
------------------------
* The analysis is performed on the **increments** of the input series,
  not on the raw levels.  For an FBM path this prevents bias that would
  otherwise inflate the spectrum width.
* Scales that yield fewer than two non‑overlapping windows are skipped.
* Only finite fluctuation moments are used in the regression.
"""

from __future__ import annotations
import warnings
from typing import Dict

import numpy as np
import pandas as pd

from ._base import BaseEstimator


# ── compatibility shims ─────────────────────────────────────────────────
# NumPy 2.0 removed ndarray.ptp – add a trivial wrapper so legacy tests
# that expect `array.ptp()` continue to work.
if not hasattr(np.ndarray, "ptp"):

    def _ptp(self, axis=None, out=None, keepdims=False):
        return np.ptp(self, axis=axis, out=out, keepdims=keepdims)

    setattr(np.ndarray, "ptp", _ptp)

# pandas.Series also gets a tiny helper for `.ptp()`
if not hasattr(pd.Series, "ptp"):

    def _series_ptp(self: pd.Series) -> float:
        """Peak‑to‑peak helper compatible with NumPy’s API."""
        return float(self.max() - self.min())

    pd.Series.ptp = _series_ptp  # type: ignore[attr-defined]
# ────────────────────────────────────────────────────────────────────────

class MFDFA(BaseEstimator):
    """
    Parameters
    ----------
    series : array‑like
        Level series (the estimator will internally difference it).
    q : ndarray, optional
        Range of moments (default −4…+4).
    min_scale, max_scale, n_scales : int
        Log‑spaced scale grid configuration.
    """

    def __init__(
        self,
        series,
        *,
        q: np.ndarray | None = None,
        min_scale: int = 8,
        max_scale: int | None = None,
        n_scales: int = 20,
        auto_range: bool = False,
        r2_thresh: float = 0.98,
        min_points: int = 5,
    ):
        super().__init__(series)
        self.q = q if q is not None else np.arange(-4, 5)  # −4 … 4
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.auto_range = auto_range
        self.r2_thresh = r2_thresh
        self.min_points = min_points

    # ------------------------------------------------------------------ #
    @staticmethod
    def _best_range(log_s: list[float], Fq: list[float], min_points: int, r2: float):
        n = len(log_s)
        best = slice(0, n)
        best_r2 = -np.inf
        for i in range(n - min_points + 1):
            for j in range(i + min_points, n + 1):
                r_val = np.corrcoef(log_s[i:j], Fq[i:j])[0, 1] ** 2
                if r_val > best_r2 and r_val >= r2:
                    best_r2 = r_val
                    best = slice(i, j)
        return best

    # ------------------------------------------------------------------ #
    @staticmethod
    def _F2(profile: np.ndarray, s: int) -> np.ndarray:
        """Return variance of detrended windows at scale *s* (shape: k,)."""
        N = len(profile)
        k = N // s
        if k < 2:
            return np.array([])

        windows = profile[: k * s].reshape(k, s)
        t = np.arange(s)
        F2 = np.empty(k)
        for i, w in enumerate(windows):
            a, b = np.polyfit(t, w, 1)
            F2[i] = np.mean((w - (a * t + b)) ** 2)
        return F2

    # ------------------------------------------------------------------ #
    def fit(self):
        # 1 ▸ increments
        x_raw = (
            self.series.to_numpy(dtype=float)
            if isinstance(self.series, pd.Series)
            else np.asarray(self.series, dtype=float)
        )
        x = np.diff(x_raw)
        N = len(x)

        # 2 ▸ profile
        profile = np.cumsum(x - x.mean())

        # 3 ▸ log‑spaced scales
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

        # 4 ▸ fluctuation functions
        Hq: Dict[float, float] = {}
        for q in self.q:
            Fq = []
            log_s_valid = []
            for s in scales:
                F2 = self._F2(profile, s)
                if F2.size == 0:
                    continue
                F2 = F2[F2 > 0]  # drop zero variances

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    if q == 0:
                        F = np.exp(0.5 * np.nanmean(np.log(F2)))
                    else:
                        F = np.nanmean(F2 ** (q / 2.0)) ** (1.0 / q)
                if np.isfinite(F) and F > 0:
                    Fq.append(np.log(F))
                    log_s_valid.append(np.log(s))
            if len(Fq) < 2:
                continue
            sl = (
                self._best_range(log_s_valid, Fq, self.min_points, self.r2_thresh)
                if self.auto_range
                else slice(0, len(Fq))
            )
            h, _ = np.polyfit(np.array(log_s_valid)[sl], np.array(Fq)[sl], 1)
            Hq[q] = h
        # 5 ▸ singularity spectrum
        qs = np.array(sorted(Hq.keys()))
        hq = np.array([Hq[q] for q in qs])
        tq = qs * hq - 1
        alpha = np.gradient(tq, qs)
        f_alpha = qs * alpha - tq
        self.result_ = {
            "q": qs,
            "h": hq,
            "tau": tq,
            "alpha": alpha,
            "f_alpha": f_alpha,
        }
        return self
