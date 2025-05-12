"""
Benchmark Volatility Models
===========================
1. AR(1)–GARCH(1, 1) – standard conditional variance model
2. HAR‑RV            – Heterogeneous‑Autoregressive realised variance model

Each exposes a simple API:

    garch = GARCH().fit(returns)
    sig2  = garch.forecast(h=5)       # np.ndarray

    har   = HAR().fit(realised_var)
    rv_fc = har.forecast(h=3)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm

__all__ = ["GARCH", "HAR"]


# --------------------------------------------------------------------- #
class GARCH:
    """
    AR(1) mean – GARCH(1, 1) variance (Gaussian dist.) via **arch**.
    """

    def __init__(self) -> None:
        self._res = None

    # ................................................................. #
    def fit(self, r: pd.Series | np.ndarray) -> "GARCH":
        r = pd.Series(r, dtype=float)
        am = arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
        self._res = am.fit(disp="off")
        return self

    # ................................................................. #
    def forecast(self, h: int = 1) -> np.ndarray:
        if self._res is None:
            raise RuntimeError("Call .fit() first.")
        f = self._res.forecast(horizon=h, reindex=False)
        return f.variance.values.flatten()          # shape (h,)


# --------------------------------------------------------------------- #
class HAR:
    """
    Heterogeneous‑Autoregressive model (Corsi, 2009) for realised variance.

    Y_{t+1} = β₀ + β_d Y_{t,d} + β_w Y_{t,w} + β_m Y_{t,m} + ε_{t+1}

    with daily, weekly (5‑day) and monthly (22‑day) realised‑variance means.
    """

    def __init__(self, lags: Sequence[int] = (1, 5, 22)) -> None:
        self.lags = tuple(sorted(lags))
        self.params_: pd.Series | None = None
        self._last_row_: pd.Series | None = None

    # ................................................................. #
    def _design_matrix(self, rv: pd.Series) -> pd.DataFrame:
        """Lagged‑mean regressors shifted by one day (so known at t)."""
        X = pd.DataFrame(index=rv.index)
        X["const"] = 1.0
        for L in self.lags:
            X[f"rv_{L}"] = rv.rolling(L).mean().shift(1)
        return X.dropna()

    # ................................................................. #
    def fit(self, rv: pd.Series) -> "HAR":
        """OLS fit; rv must be positive daily realised variance."""
        rv = rv.astype(float)
        X = self._design_matrix(rv)
        y = rv.loc[X.index]                       # target = RV_{t}
        model = sm.OLS(y, X).fit()
        self.params_ = model.params
        self._last_row_ = X.iloc[-1]
        return self

    # ................................................................. #
    def forecast(self, h: int = 1) -> np.ndarray:
        if self.params_ is None or self._last_row_ is None:
            raise RuntimeError("Call .fit() first.")

        preds = []
        history = [self._last_row_[f"rv_{self.lags[0]}"]]  # start with last daily RV

        for _ in range(h):
            regressors = [1.0]  # const
            for L in self.lags:
                reg = np.mean(history[-L:]) if len(history) >= L else history[-1]
                regressors.append(reg)

            pred = float(np.dot(self.params_.values, regressors))
            pred = max(pred, 1e-12)              # ensure strictly positive
            preds.append(pred)
            history.append(pred)                 # update rolling window

        return np.array(preds)
