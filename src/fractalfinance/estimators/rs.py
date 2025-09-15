"""
Rescaled‑Range (R/S) Hurst‑exponent estimator
=============================================
Implements classic Hurst (1951) analysis:

    1.  Divide the series into windows of length n
    2.  Within each window compute  R/S  where
        R = max(cumdev) − min(cumdev)
        S = sample std‑dev of the window
    3.  Average R/S over windows, repeat for several n
    4.  Slope of  log〈R/S〉  vs  log n  gives H

By default the estimator assumes the input already represents
increments (returns).  When passing *level* observations, set
``from_levels=True`` so the series is first differenced.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEstimator


class RS(BaseEstimator):
    """Rescaled‑range estimator of the Hurst exponent."""

    def fit(
        self,
        min_chunk: int = 16,
        max_chunk: int | None = None,
        *,
        from_levels: bool = False,
    ):
        # ------------------------------------------------------------------ #
        # 1. Work with increments; optionally difference a level path
        x = np.asarray(self.series, dtype=float)
        x = np.diff(x, n=1) if from_levels else x
        N = len(x)
        if N < 2:
            raise ValueError("Need at least two observations")

        # ------------------------------------------------------------------ #
        # 2. Choose window sizes on a log grid
        max_chunk = max_chunk or N // 4
        ns = np.unique(
            np.floor(np.logspace(np.log10(min_chunk), np.log10(max_chunk), 9)).astype(
                int
            )
        )

        RS_vals, logn = [], []
        for n in ns:
            k = N // n  # number of full windows
            if k < 2:
                continue

            Z = x[: k * n].reshape(k, n)
            rs_seg = []
            for row in Z:
                cumdev = np.cumsum(row - row.mean())
                R = np.ptp(cumdev)  # range
                S = row.std(ddof=1)
                if S > 0:
                    rs_seg.append(R / S)

            if rs_seg:
                RS_vals.append(float(np.nanmean(rs_seg)))
                logn.append(np.log(n))

        # ------------------------------------------------------------------ #
        # 3. Linear fit in log–log space: slope = H
        H, _ = np.polyfit(logn, np.log(RS_vals), 1)
        self.result_ = {"H": float(H)}
        return self
