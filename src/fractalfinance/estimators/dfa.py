"""Detrended‑Fluctuation Analysis (DFA‑1).

This implementation follows the standard formulation used in the
thesis, where the *profile* is built by cumulatively summing the
mean‑centered input series :math:`x_k`.  When the supplied data are
*level* observations of a process such as fractional Brownian motion,
setting ``from_levels=True`` first differences the series so that the
estimated slope maps to the Hurst exponent :math:`H` rather than
``H+1``.  By default the estimator assumes the input already consists of
increments (returns).

* Skips scales where fewer than two windows fit.
* Requires at least two finite fluctuation points for regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import BaseEstimator


class DFA(BaseEstimator):
    def __init__(
        self,
        series,
        *,
        min_scale: int = 8,
        max_scale: int | None = None,
        n_scales: int = 20,
        from_levels: bool = False,

    ):
        super().__init__(series)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.from_levels = from_levels


    # ------------------------------------------------------------------ #
    @staticmethod
    def _detrended_var(profile: np.ndarray, s: int) -> float:
        """Mean squared residual of linear detrend at scale *s*."""
        N = len(profile)
        k = N // s
        if k < 2:
            return np.nan

        windows = profile[: k * s].reshape(k, s)
        t = np.arange(s)
        res = []
        for w in windows:
            a, b = np.polyfit(t, w, 1)
            res.append(np.mean((w - (a * t + b)) ** 2))
        return float(np.mean(res))

    # ------------------------------------------------------------------ #
    def fit(self):
        # 1. Safe ndarray
        x_raw = self.series
        if isinstance(x_raw, pd.Series):
            x_raw = x_raw.to_numpy(dtype=float)
        else:
            x_raw = np.asarray(x_raw, dtype=float)

        # 2. Optionally convert level series to increments (fGn)
        x = np.diff(x_raw, n=1) if self.from_levels else x_raw
        x = np.asarray(x, dtype=float)
        N = len(x)
        if N < 2:
            raise ValueError("Series too short for DFA.")

        # 3. Build profile of mean‑centered data
        profile = np.cumsum(x - x.mean())

        # 4. Log‑spaced scales
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

        # 5. Fluctuation function
        F2 = np.array([self._detrended_var(profile, s) for s in scales])
        mask = np.isfinite(F2) & (F2 > 0)
        if mask.sum() < 2:
            raise RuntimeError("DFA: not enough valid scales for regression.")

        log_s = np.log(scales[mask])
        log_F = 0.5 * np.log(F2[mask])  # F = sqrt(F²)

        sl = (
            self._best_range(log_s, log_F, self.min_points, self.r2_thresh)
            if self.auto_range
            else slice(0, len(log_s))
        )
        H, _ = np.polyfit(log_s[sl], log_F[sl], 1)
        result = {"H": float(H), "scales": scales[mask][sl]}

        if self.n_boot > 0:
            boot = []
            for _ in range(self.n_boot):
                resample = np.random.choice(x, size=N, replace=True)
                prof_b = np.cumsum(resample - resample.mean())
                F2_b = np.array([self._detrended_var(prof_b, s) for s in scales])
                mask_b = np.isfinite(F2_b) & (F2_b > 0)
                log_s_b = np.log(scales[mask_b])
                log_F_b = 0.5 * np.log(F2_b[mask_b])
                sl_b = (
                    self._best_range(log_s_b, log_F_b, self.min_points, self.r2_thresh)
                    if self.auto_range
                    else slice(0, len(log_s_b))
                )
                H_b, _ = np.polyfit(log_s_b[sl_b], log_F_b[sl_b], 1)
                boot.append(H_b)
            result["H_std"] = float(np.std(boot, ddof=1))
        self.result_ = result
        return self
