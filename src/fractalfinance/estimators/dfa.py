"""Detrended‑Fluctuation Analysis (DFA‑1).

This implementation follows the standard formulation used in the
thesis, where the *profile* is built by cumulatively summing the
mean‑centered input series :math:`x_k`.  By default the estimator treats
the provided data as *level* observations of a process such as
fractional Brownian motion and differences them (``from_levels=True``)
so that the estimated slope maps to the Hurst exponent :math:`H` rather
than ``H+1``.  Set ``from_levels=False`` when the series already
represents increments (returns).

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
        from_levels: bool = True,
        auto_range: bool | None = None,
        min_points: int | None = None,
        r2_thresh: float | None = None,
        n_boot: int | None = None,


    ):
        super().__init__(series)
        self.min_scale = int(min_scale)
        self.max_scale = max_scale
        self.n_scales = int(n_scales)
        self.from_levels = from_levels
        if auto_range is not None:
            self.auto_range = bool(auto_range)
        if min_points is not None:
            self.min_points = int(min_points)
        if r2_thresh is not None:
            self.r2_thresh = float(r2_thresh)
        if n_boot is not None:
            self.n_boot = int(n_boot)



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
        slope, intercept = np.polyfit(log_s[sl], log_F[sl], 1)
        fit_start = int(sl.start or 0)
        fit_stop = int(sl.stop or len(log_s))
        fluct = np.sqrt(F2[mask])
        result = {
            "H": float(slope),
            "intercept": float(intercept),
            "scales": scales[mask],
            "fluctuation": fluct,
            "log_scales": log_s,
            "log_fluct": log_F,
            "fit_start": fit_start,
            "fit_stop": fit_stop,
            "fit_scales": scales[mask][sl],
        }

        if self.n_boot > 0:
            boot = []
            for _ in range(self.n_boot):
                resample = np.random.choice(x, size=N, replace=True)
                prof_b = np.cumsum(resample - resample.mean())
                F2_b = np.array([self._detrended_var(prof_b, s) for s in scales])
                mask_b = np.isfinite(F2_b) & (F2_b > 0)
                if mask_b.sum() < 2:
                    continue
                log_s_b = np.log(scales[mask_b])
                log_F_b = 0.5 * np.log(F2_b[mask_b])
                sl_b = (
                    self._best_range(log_s_b, log_F_b, self.min_points, self.r2_thresh)
                    if self.auto_range
                    else slice(0, len(log_s_b))
                )
                if log_s_b[sl_b].size < 2:
                    continue
                H_b, _ = np.polyfit(log_s_b[sl_b], log_F_b[sl_b], 1)
                boot.append(H_b)
            if len(boot) == 1:
                result["H_std"] = 0.0
            elif len(boot) > 1:
                result["H_std"] = float(np.std(boot, ddof=1))
        self.result_ = result
        return self
