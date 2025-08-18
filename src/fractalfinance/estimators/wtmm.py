"""
Wavelet‑Transform Modulus‑Maxima (WTMM) Multifractal Estimator
-------------------------------------------------------------
Implements 1‑D WTMM with a Mexican‑Hat (Ricker) continuous wavelet.

Steps
1. Continuous wavelet transform at dyadic scales a_j = a0·2^j.
2. Track modulus maxima across scales.
3. Build partition function Z(q,a).
4. Regress log Z vs log a → τ(q) → singularity spectrum (α,f(α)).

References
* Muzy, Bacry, Arneodo (1993)  *Phys. Rev. Lett.*
* Mallat, “A Wavelet Tour of Signal Processing”, ch. 15
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pywt

from ._base import BaseEstimator


class WTMM(BaseEstimator):
    def __init__(
        self,
        series,
        q: Sequence[float] = (-4, -2, -1, -0.5, 0.5, 1, 2, 3, 4),
        a0: float = 2.0,
        n_scales: int = 16,
    ):
        # linear interpolation if user passes a Series with NaNs
        if isinstance(series, pd.Series):
            series = series.interpolate("linear")
        super().__init__(series)

        self.q = np.asarray(q, dtype=float)
        self.a0 = float(a0)
        self.n_scales = int(n_scales)

        # dyadic scales a_j = a0 * 2^j
        self.scales = self.a0 * 2.0 ** np.arange(self.n_scales)
        self._wavelet = pywt.ContinuousWavelet("mexh")

    # ------------------------------------------------------------------ #
    def _cwt(self, x: np.ndarray) -> np.ndarray:
        """Return CWT coefficients of shape (len(scales), len(x))."""
        coeffs, _freqs = pywt.cwt(
            data=x,
            scales=self.scales,
            wavelet=self._wavelet,
            method="fft",
        )
        return coeffs

    # ------------------------------------------------------------------ #
    @staticmethod
    def _modulus_maxima(coeff_row: np.ndarray) -> np.ndarray:
        """Indices where |coeff| has a strict local maximum."""
        mag = np.abs(coeff_row)
        peaks = (mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])
        return np.where(peaks)[0] + 1  # +1 because of slice offset

    # ------------------------------------------------------------------ #
    def fit(self) -> "WTMM":
        # ---------- 1. Input handling & increments --------------------- #
        x_raw = self.series
        if isinstance(x_raw, pd.Series):
            x_raw = x_raw.to_numpy(dtype=float)
        x = np.asarray(x_raw, dtype=float)
        x = np.diff(x, n=1)  # analyse increments (fGn)

        # ---------- 2. Continuous wavelet transform -------------------- #
        coeffs = self._cwt(x)  # shape (S, N)

        # ---------- 3. Partition function Z(q,a) ----------------------- #
        Z = np.empty((self.q.size, self.scales.size))
        for j, a in enumerate(self.scales):
            maxima_idx = self._modulus_maxima(coeffs[j])
            if maxima_idx.size == 0:
                Z[:, j] = np.nan
                continue

            mod_vals = np.abs(coeffs[j, maxima_idx])
            for iq, qv in enumerate(self.q):
                if qv == 0:
                    Z[iq, j] = np.exp(np.nanmean(np.log(mod_vals)))
                else:
                    Z[iq, j] = (np.nanmean(mod_vals**qv)) ** (1.0 / qv)

        # ---------- 4. τ(q) via log–log regression --------------------- #
        tau_q = np.empty_like(self.q)
        for i in range(self.q.size):
            mask = np.isfinite(Z[i])
            if mask.sum() < 2:
                tau_q[i] = np.nan
                continue
            slope, _ = np.polyfit(np.log(self.scales[mask]), np.log(Z[i, mask]), 1)
            tau_q[i] = slope

        # ---------- 5. Singularity spectrum --------------------------- #
        alpha = np.gradient(tau_q, self.q)
        f_alpha = self.q * alpha - tau_q

        self.result_ = {
            "tau(q)": tau_q,
            "alpha": alpha,
            "f(alpha)": f_alpha,
            "q": self.q,
            "scales": self.scales,
        }
        return self
