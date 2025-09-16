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
    """Minimal parent class; concrete estimators implement `.fit()`.

    Besides coercing the input to a one‑dimensional float array, the base
    class centralises a couple of convenience attributes used throughout the
    thesis' estimators:

    ``auto_range``
        When ``True`` the estimator automatically selects the most linear
        region of a log–log diagram.  The helper :meth:`_best_range` mirrors
        the manual diagnostic procedure used in the empirical chapters by
        scanning all contiguous windows and picking the longest stretch whose
        :math:`R^2` exceeds ``r2_thresh``.  If no window satisfies the
        criterion the whole range is used.

    ``min_points`` and ``r2_thresh``
        Respectively the minimum number of scales required for the automatic
        selection and the minimum acceptable :math:`R^2` of the linear fit.

    These defaults keep the estimators aligned with the methodology described
    in Chapter 5 and can still be overridden on a per‑instance basis.
    """

    auto_range: bool = False
    min_points: int = 6
    r2_thresh: float = 0.98
    n_boot: int = 0

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
        """Return the slice corresponding to the most linear log–log region.

        Parameters
        ----------
        x, y : ndarray
            Abscissa/ordinate coordinates of the log–log diagram (already in
            logarithmic units).
        min_points : int
            Minimum number of points required for the regression window.
        r2_thresh : float
            Minimum :math:`R^2` accepted for the window.  If no contiguous
            window attains the threshold the function falls back to the whole
            range.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        if n == 0:
            return slice(0, 0)
        if n <= min_points:
            return slice(0, n)

        best_slice = slice(0, n)
        best_score: tuple[int, float, float] | None = None

        def _ols_stats(start: int, stop: int) -> tuple[float, float]:
            xs = x[start:stop]
            ys = y[start:stop]
            slope, intercept = np.polyfit(xs, ys, 1)
            fitted = slope * xs + intercept
            resid = ys - fitted
            ss_res = float(np.sum(resid**2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            return r2, slope

        for start in range(0, n - min_points + 1):
            for stop in range(start + min_points, n + 1):
                r2, slope = _ols_stats(start, stop)
                length = stop - start
                priority = 1 if r2 >= r2_thresh else 0
                score = (priority, length, r2)

                if best_score is None or score > best_score:
                    best_score = score
                    best_slice = slice(start, stop)

        return best_slice

