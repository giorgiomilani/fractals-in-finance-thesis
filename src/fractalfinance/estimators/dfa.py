"""
Detrended‑Fluctuation Analysis (DFA‑1) estimator
===============================================

* Works on the **increments** of the input series so that
  FBM levels with H produce slope ≈ H (not H+1).
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
        auto_range: bool = False,
        r2_thresh: float = 0.98,
        min_points: int = 5,
        n_boot: int = 0,
    ):
        super().__init__(series)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        self.auto_range = auto_range
        self.r2_thresh = r2_thresh
        self.min_points = min_points
        self.n_boot = n_boot

    # ------------------------------------------------------------------ #
    @staticmethod
    def _best_range(log_s: np.ndarray, log_F: np.ndarray, min_points: int, r2: float):
        """Return slice of scales with highest R² above threshold."""
        n = len(log_s)
        best = slice(0, n)
        best_r2 = -np.inf
        for i in range(n - min_points + 1):
            for j in range(i + min_points, n + 1):
                r = np.corrcoef(log_s[i:j], log_F[i:j])[0, 1] ** 2
                if r > best_r2 and r >= r2:
                    best_r2 = r
                    best = slice(i, j)
        return best

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

        # 2. Convert to increments (fGn) to target slope = H
        x = np.diff(x_raw, n=1)
        N = len(x)
        if N < 2:
            raise ValueError("Series too short for DFA.")

        # 3. Build profile of increments
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
