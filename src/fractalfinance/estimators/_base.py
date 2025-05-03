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

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, **kwargs) -> "BaseEstimator":
        """Run the estimator and populate `self.result_`."""
        ...
