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
    ):
        super().__init__(series)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales

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

        H, _ = np.polyfit(log_s, log_F, 1)
        self.result_ = {"H": float(H)}
        return self
