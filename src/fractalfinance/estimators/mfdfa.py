"""
Multifractal Detrended Fluctuation Analysis (MFDFA)
===================================================

Implements the Kantelhardt et al. (2002) procedure (DFA‑1 detrending)
and returns the singularity spectrum (α, f(α)).

Key implementation notes
------------------------
* The analysis operates on series of **increments** (returns).  When
  passing level data, specify ``from_levels=True`` to difference the
  series prior to profiling; this matches the thesis formulation where
  the profile integrates mean‑centered increments.
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
        Input increments.  Set ``from_levels=True`` when supplying
        level observations so the estimator differences them.
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
        from_levels: bool = False,
        auto_range: bool | None = None,
        min_points: int | None = None,
        r2_thresh: float | None = None,

    ):
        super().__init__(series)
        self.q = q if q is not None else np.arange(-4, 5)  # −4 … 4
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.from_levels = from_levels
        if auto_range is not None:
            self.auto_range = bool(auto_range)
        if min_points is not None:
            self.min_points = int(min_points)
        if r2_thresh is not None:
            self.r2_thresh = float(r2_thresh)



    # ------------------------------------------------------------------ #
    @staticmethod
    def _F2(profile: np.ndarray, s: int) -> np.ndarray:
        """Variance of detrended windows at scale *s* (``2k`` values).

        The thesis formulates MFDFA with ``2N_s`` segments obtained by
        scanning the profile from both the start and the end.  This helper
        mirrors that approach by detrending forward and reversed windows and
        returning the variances of all ``2k`` segments.
        """

        N = len(profile)
        k = N // s
        if k < 2:
            return np.array([])

        t = np.arange(s)
        F2 = []

        # forward windows
        for w in profile[: k * s].reshape(k, s):
            a, b = np.polyfit(t, w, 1)
            F2.append(np.mean((w - (a * t + b)) ** 2))

        # reversed windows (account for residual segment)
        for w in profile[::-1][: k * s].reshape(k, s):
            a, b = np.polyfit(t, w, 1)
            F2.append(np.mean((w - (a * t + b)) ** 2))

        return np.asarray(F2)

    # ------------------------------------------------------------------ #
    def fit(self):
        # 1 ▸ increments (optionally difference level data)
        x = np.asarray(self.series, dtype=float)
        x = np.diff(x) if self.from_levels else x
        N = len(x)

        # 2 ▸ profile
        profile = np.cumsum(x - x.mean())

        # 3 ▸ log‑spaced scales
        max_scale = self.max_scale or N // 4
        if max_scale < self.min_scale:
            raise ValueError("max_scale must be >= min_scale")
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
            log_s_arr = np.asarray(log_s_valid, dtype=float)
            Fq_arr = np.asarray(Fq, dtype=float)
            sl = (
                self._best_range(log_s_arr, Fq_arr, self.min_points, self.r2_thresh)
                if self.auto_range
                else slice(0, len(Fq_arr))
            )
            if log_s_arr[sl].size < 2:
                continue
            h, _ = np.polyfit(log_s_arr[sl], Fq_arr[sl], 1)
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
