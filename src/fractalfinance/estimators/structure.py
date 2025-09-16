"""Structure-function based scaling and intermittency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ._base import BaseEstimator

__all__ = ["StructureFunction", "ScalingResult"]


@dataclass(slots=True)
class ScalingResult:
    q: np.ndarray
    scales: np.ndarray
    structure: Dict[float, np.ndarray]
    zeta: Dict[float, float]
    H: float
    lambda_: float
    tau: np.ndarray
    alpha: np.ndarray
    f_alpha: np.ndarray
    delta_alpha: float


class StructureFunction(BaseEstimator):
    """Estimate scaling exponents ``ζ(q)`` from empirical structure functions."""

    def __init__(
        self,
        series,
        *,
        q: np.ndarray | None = None,
        min_scale: int = 1,
        max_scale: int | None = None,
        n_scales: int = 12,
        from_levels: bool = False,
        auto_range: bool | None = None,
        min_points: int | None = None,
        r2_thresh: float | None = None,
    ) -> None:
        super().__init__(series)
        if q is None:
            q = np.linspace(0.5, 4.0, 8)
        self.q = np.asarray(q, dtype=float)
        if np.any(self.q <= 0):
            raise ValueError("StructureFunction expects strictly positive q values")
        self.min_scale = int(min_scale)
        self.max_scale = max_scale
        self.n_scales = int(n_scales)
        self.from_levels = bool(from_levels)
        if auto_range is not None:
            self.auto_range = bool(auto_range)
        else:
            self.auto_range = True
        if min_points is not None:
            self.min_points = int(min_points)
        if r2_thresh is not None:
            self.r2_thresh = float(r2_thresh)

    @staticmethod
    def _structure_function(x: np.ndarray, scale: int, q: float) -> float:
        if scale <= 0:
            raise ValueError("scale must be positive")
        if x.size <= scale:
            return np.nan
        increments = np.convolve(x, np.ones(scale), mode="valid")
        return float(np.mean(np.abs(increments) ** q))

    def fit(self) -> "StructureFunction":
        x = np.asarray(self.series, dtype=float)
        x = np.diff(x) if self.from_levels else x
        N = x.size
        if N < 2:
            raise ValueError("Series too short for structure-function analysis")

        max_scale = self.max_scale or max(int(N // 8), self.min_scale + 1)
        scales = np.unique(
            np.floor(
                np.logspace(
                    np.log10(self.min_scale),
                    np.log10(max_scale),
                    num=self.n_scales,
                )
            ).astype(int)
        )
        scales = scales[scales > 0]
        structure: Dict[float, np.ndarray] = {}
        zeta: Dict[float, float] = {}

        log_scales = np.log(scales)

        for qv in self.q:
            key = float(qv)
            moments = np.array([self._structure_function(x, s, qv) for s in scales])
            mask = np.isfinite(moments) & (moments > 0)
            if mask.sum() < 2:
                continue
            ys = np.log(moments[mask])
            xs = log_scales[mask]
            sl = (
                self._best_range(xs, ys, self.min_points, self.r2_thresh)
                if self.auto_range
                else slice(0, xs.size)
            )
            slope, intercept = np.polyfit(xs[sl], ys[sl], 1)
            zeta[key] = float(slope)
            structure[key] = moments

        if not zeta:
            raise RuntimeError("Failed to compute any structure-function slope")

        qs = np.array(sorted(zeta.keys()))
        zeta_vals = np.array([zeta[q] for q in qs])
        design = np.column_stack((qs, -0.5 * qs * (qs - 1.0)))
        theta, *_ = np.linalg.lstsq(design, zeta_vals, rcond=None)
        H_est = float(theta[0])
        lambda2 = float(theta[1])
        lambda_est = float(np.sqrt(max(lambda2, 0.0)))

        tau = zeta_vals.copy()
        alpha = np.gradient(tau, qs)
        f_alpha = qs * alpha - tau
        delta_alpha = float(alpha.max() - alpha.min())

        self.result_ = ScalingResult(
            q=qs,
            scales=scales,
            structure=structure,
            zeta=zeta,
            H=H_est,
            lambda_=lambda_est,
            tau=tau,
            alpha=alpha,
            f_alpha=f_alpha,
            delta_alpha=delta_alpha,
        )
        return self
