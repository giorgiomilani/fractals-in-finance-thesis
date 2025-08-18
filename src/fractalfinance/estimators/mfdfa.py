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
import numpy as np
import pandas as pd
from ._base import BaseEstimator

<<<<<<< Updated upstream
=======
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

>>>>>>> Stashed changes

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
    ):
        super().__init__(series)
        self.q = q if q is not None else np.arange(-4, 5)  # −4 … 4
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales

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
<<<<<<< Updated upstream
        # 1.  Convert to ndarray and take INCREMENTS
        x_raw = self.series
        if isinstance(x_raw, pd.Series):
            x_raw = x_raw.to_numpy(dtype=float)
        else:
            x_raw = np.asarray(x_raw, dtype=float)
        x = np.diff(x_raw, n=1)
        N = len(x)

        # 2.  Build profile
        profile = np.cumsum(x - x.mean())

        # 3.  Scale grid
=======
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
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
        # 4.  Fluctuation function for each q
        Hq = {}
=======
        # 4 ▸ fluctuation functions
        Hq: Dict[float, float] = {}
>>>>>>> Stashed changes
        for q in self.q:
            Fq = []
            log_s_valid = []
            for s in scales:
                F2 = self._F2(profile, s)
                if F2.size == 0:
                    continue
<<<<<<< Updated upstream
=======
                F2 = F2[F2 > 0]  # drop zero variances

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    if q == 0:
                        F = np.exp(0.5 * np.nanmean(np.log(F2)))
                    else:
                        F = np.nanmean(F2 ** (q / 2.0)) ** (1.0 / q)
>>>>>>> Stashed changes

                if q == 0:
                    F = np.exp(0.5 * np.mean(np.log(F2)))
                else:
                    F = (np.mean(F2 ** (q / 2))) ** (1 / q)
                if np.isfinite(F) and F > 0:
                    Fq.append(np.log(F))
                    log_s_valid.append(np.log(s))

<<<<<<< Updated upstream
            if len(Fq) < 2:
=======
            if len(logF) < 2:
>>>>>>> Stashed changes
                continue
            h, _ = np.polyfit(log_s_valid, Fq, 1)
            Hq[q] = h

<<<<<<< Updated upstream
        # 5.  Singularity spectrum
=======
        # 5 ▸ singularity spectrum
>>>>>>> Stashed changes
        qs = np.array(sorted(Hq.keys()))
        hq = np.array([Hq[q] for q in qs])
        tq = qs * hq - 1
        alpha = np.gradient(tq, qs)
        f_alpha = qs * alpha - tq

<<<<<<< Updated upstream
=======
        # wrap arrays in Series so the `.ptp()` helper is available
>>>>>>> Stashed changes
        self.result_ = {
            "q": qs,
            "h": hq,
            "tau": tq,
            "alpha": alpha,
            "f_alpha": f_alpha,
        }
        return self
