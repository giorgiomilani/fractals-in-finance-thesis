"""
Common base class for all Hurst / multifractal estimators
========================================================
* Accepts a pandas Series **or** a NumPy 1‑D array.
* Internally stores `self.series` as a 1‑D float ndarray.
* Provides `.result_` for fit outputs.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import pandas as pd


class BaseEstimator(abc.ABC):
    """Minimal parent class; concrete estimators implement `.fit()`."""

    def __init__(self, series: pd.Series | np.ndarray | list[float]):
        # ------------------------------------------------------------------
        # Convert input to a 1‑D float ndarray, regardless of original type
        if isinstance(series, pd.Series):
            self.series = series.astype(float).to_numpy()
        else:
            self.series = np.asarray(series, dtype=float)

        if self.series.ndim != 1:
            raise ValueError("Input series must be one‑dimensional")

        self.result_: dict[str, Any] | None = None
        # Defaults for scaling-range refinement used by some estimators.
        self.auto_range: bool = False
        self.min_points: int = 5
        self.r2_thresh: float = 0.98
        self.n_boot: int = 0

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, **kwargs) -> "BaseEstimator":
        """Run the estimator and populate `self.result_`."""
        ...

    # ------------------------------------------------------------------
    @staticmethod
    def _best_range(
        x: np.ndarray,
        y: np.ndarray,
        min_points: int,
        r2_thresh: float,
    ) -> slice:
        """Return contiguous slice with the highest linear *R*² score.

        Parameters
        ----------
        x, y : array-like
            Log-scale abscissa and ordinate of the scaling regression.
        min_points : int
            Minimum number of points to keep in the fit.
        r2_thresh : float
            Target coefficient of determination.  When no window reaches
            this threshold the function returns the contiguous window with
            the highest *R*².
        """

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(x_arr)
        min_points = max(2, int(min_points))
        if n <= min_points:
            return slice(0, n)

        best = slice(0, n)
        best_len = 0
        best_r2 = -np.inf
        fallback = slice(0, n)
        fallback_r2 = -np.inf

        for start in range(0, n - min_points + 1):
            for stop in range(start + min_points, n + 1):
                xs = x_arr[start:stop]
                ys = y_arr[start:stop]
                if xs.size < 2:
                    continue
                slope, intercept = np.polyfit(xs, ys, 1)
                fitted = slope * xs + intercept
                ss_tot = np.sum((ys - ys.mean()) ** 2)
                ss_res = np.sum((ys - fitted) ** 2)
                r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
                length = stop - start

                if r2 >= r2_thresh:
                    if length > best_len or (length == best_len and r2 > best_r2):
                        best = slice(start, stop)
                        best_len = length
                        best_r2 = r2

                if r2 > fallback_r2:
                    fallback = slice(start, stop)
                    fallback_r2 = r2

        if best_len >= min_points and best_r2 >= r2_thresh:
            return best
        return fallback
