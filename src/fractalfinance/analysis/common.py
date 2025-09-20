"""Shared utilities for running fractal and volatility analyses.

The thesis repository ships a handful of end-to-end workflows that all follow
the same pattern: fetch prices, derive log-returns, fit volatility models,
compute fractal diagnostics, and persist a standard collection of plots.  The
original scripts duplicated a fair amount of bookkeeping (figure handling,
annualisation helpers, etc.).  This module centralises that logic so more
advanced runners – e.g. multi-scale comparisons – can reuse the building blocks
without re-implementing the heavy lifting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from fractalfinance.estimators import (
    DFA,
    MFDFA,
    RS,
    ScalingResult,
    StructureFunction,
    WTMM,
)
from fractalfinance.models import HAR, msm_fit


TRADING_DAYS = 252.0


# ──────────────────────────────────────────────────────────────────────────────
# generic helpers
# ──────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (including parents) when missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> str:
    """Persist ``fig`` under ``out_dir``/``filename`` and close the figure."""

    ensure_dir(out_dir)
    target = out_dir / filename
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def to_naive(index: pd.Index) -> pd.DatetimeIndex:
    """Return a timezone-naive copy of ``index`` always expressed in UTC."""

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.tz_localize(None)


def annualise(daily_value: float, periods_per_year: float) -> float:
    """Scale a per-period volatility estimate to yearly terms."""

    return float(daily_value * np.sqrt(periods_per_year))


def compute_window_starts(length: int, window: int, stride: int) -> list[int]:
    """Return sliding-window start offsets matching ``GAFWindowDataset``."""

    window = int(window)
    stride = int(stride)
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive integers")
    if length < window:
        return []
    last_start = length - window
    starts = list(range(0, last_start + 1, stride))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    return starts



def h_at(mfdfa_res: dict[str, np.ndarray], q: float) -> float | None:
    """Extract the multifractal ``h(q)`` estimate at a specific order."""

    qs = mfdfa_res.get("q")
    hs = mfdfa_res.get("h")
    if qs is None or hs is None:
        return None
    matches = np.where(np.isclose(qs, q))[0]
    if matches.size == 0:
        return None
    return float(hs[matches[0]])


# ──────────────────────────────────────────────────────────────────────────────
# statistical summaries
# ──────────────────────────────────────────────────────────────────────────────


def summarise_prices(
    prices: pd.Series,
    returns: pd.Series,
    *,
    periods_per_year: float,
) -> dict[str, Any]:
    """Return headline statistics for ``prices``/``returns``."""

    prices = prices.sort_index()
    returns = returns.sort_index()
    obs = int(len(returns))
    span = f"{prices.index[0].date()} → {prices.index[-1].date()}"
    price_change = float(prices.iloc[-1] / prices.iloc[0] - 1.0)

    ann_return = float(returns.mean() * periods_per_year)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    skew = float(returns.skew())
    kurt = float(returns.kurt())

    return {
        "observations": obs,
        "span": span,
        "price_change": price_change,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "skew": skew,
        "excess_kurtosis": kurt,
    }


@dataclass(slots=True)
class GARCHResult:
    summary: dict[str, Any]
    conditional_volatility: pd.Series
    forecast_daily: list[float]


@dataclass(slots=True)
class HARResult:
    summary: dict[str, Any]
    realised_variance: pd.Series
    forecast_variance: list[float]
    forecast_daily_vol: list[float]


def fit_garch(
    returns: pd.Series,
    *,
    periods_per_year: float,
    mean: str = "AR",
    lags: int = 1,
    vol: str = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> GARCHResult:
    """Estimate an AR(1)-GARCH model and return diagnostics."""

    scaled_returns = (returns * 100).rename("ret_pct")
    am = arch_model(
        scaled_returns,
        mean=mean,
        lags=lags,
        vol=vol,
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100.0
    forecasts = res.forecast(horizon=5, reindex=False)
    variance_forecast = forecasts.variance.iloc[-1]
    daily_forecast = [float(np.sqrt(v) / 100.0) for v in variance_forecast]

    summary = {
        "params": {k: float(v) for k, v in res.params.items()},
        "last_cond_vol_daily": float(cond_vol.iloc[-1]),
        "last_cond_vol_annual": annualise(cond_vol.iloc[-1], periods_per_year),
        "forecast_daily_vol": daily_forecast,
        "forecast_annual_vol": [
            annualise(v, periods_per_year) for v in daily_forecast
        ],
    }

    return GARCHResult(summary, cond_vol, daily_forecast)


def compute_realised_variance(returns: pd.Series) -> pd.Series:
    """Return a daily realised-variance proxy from log-returns."""

    rv = pd.Series(returns, dtype=float) ** 2
    return rv.rename("realised_variance").dropna()


def fit_har_realised_variance(
    returns: pd.Series,
    *,
    periods_per_year: float,
    lags: Sequence[int] = (1, 5, 22),
    horizon: int = 5,
) -> HARResult:
    """Fit a HAR-RV model on realised variance and summarise forecasts."""

    rv = compute_realised_variance(returns)
    if rv.empty:
        raise ValueError("Realised variance series is empty; cannot fit HAR model")

    har = HAR(lags=lags)
    design = har._design_matrix(rv).dropna()
    if design.empty:
        raise ValueError(
            "Insufficient observations for HAR lags; provide ≥ max(lags)+1 returns"
        )

    fitted = har.fit(rv)
    forecast_variance = [float(v) for v in fitted.forecast(h=horizon)]
    forecast_daily_vol = [float(np.sqrt(v)) for v in forecast_variance]

    last_idx = getattr(fitted._last_row_, "name", rv.index[-1])
    try:
        last_realised_var = float(rv.loc[last_idx])
    except KeyError:
        last_idx = rv.index[-1]
        last_realised_var = float(rv.iloc[-1])
    last_realised_vol = float(np.sqrt(last_realised_var))

    params = fitted.params_
    if params is None:
        params = pd.Series(dtype=float)

    summary = {
        "lags": list(fitted.lags),
        "params": {key: float(val) for key, val in params.items()},
        "last_timestamp": last_idx.isoformat() if hasattr(last_idx, "isoformat") else last_idx,
        "last_realised_vol_daily": last_realised_vol,
        "last_realised_vol_annual": annualise(last_realised_vol, periods_per_year),
        "forecast_daily_vol": forecast_daily_vol,
        "forecast_annual_vol": [
            annualise(v, periods_per_year) for v in forecast_daily_vol
        ],
    }

    return HARResult(summary, rv, forecast_variance, forecast_daily_vol)


def fit_msm(returns: pd.Series, *, states: int = 5) -> dict[str, Any]:
    """Fit the Markov-Switching Multifractal model and summarise parameters."""

    params = msm_fit(returns.to_numpy(), K=states)
    return {
        "sigma2": float(params.sigma2),
        "m_L": float(params.m_L),
        "m_H": float(params.m_H),
        "gamma_1": float(params.gamma_1),
        "b": float(params.b),
        "K": int(params.K),
    }


@dataclass(slots=True)
class FractalResult:
    summary: dict[str, Any]
    rs: dict[str, Any]
    dfa: dict[str, Any]
    structure: Any
    mfdfa: dict[str, Any]
    wtmm: dict[str, Any]


def compute_fractal_metrics(
    prices: pd.Series,
    returns: pd.Series,
    *,
    structure_from_levels: bool = False,
) -> FractalResult:
    """Run the suite of fractal estimators and package key diagnostics."""

    rs_res = RS(returns).fit().result_
    dfa_res = DFA(prices, from_levels=True, auto_range=True).fit().result_
    struct_res = StructureFunction(
        returns,
        from_levels=structure_from_levels,
    ).fit().result_
    mfdfa_res = MFDFA(prices, from_levels=True, auto_range=True).fit().result_
    wtmm_res = WTMM(returns, from_levels=structure_from_levels).fit().result_

    summary = {
        "RS_H": float(rs_res["H"]),
        "DFA_H": float(dfa_res["H"]),
        "Structure_H": float(struct_res.H),
        "Structure_lambda": float(struct_res.lambda_),
        "Structure_delta_alpha": float(struct_res.delta_alpha),
        "MFDFA_h2": h_at(mfdfa_res, 2.0),
        "MFDFA_width": float(np.max(mfdfa_res["alpha"]) - np.min(mfdfa_res["alpha"])),
        "WTMM_width": float(
            np.nanmax(wtmm_res["alpha"]) - np.nanmin(wtmm_res["alpha"])
        ),
    }

    return FractalResult(summary, rs_res, dfa_res, struct_res, mfdfa_res, wtmm_res)


def _describe_distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if arr.size > 1:
        stats.update(
            {
                "std": float(np.std(arr, ddof=0)),
                "q05": float(np.quantile(arr, 0.05)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "q95": float(np.quantile(arr, 0.95)),
            }
        )
    else:
        stats.update(
            {
                "std": 0.0,
                "q05": stats["min"],
                "q25": stats["min"],
                "q75": stats["max"],
                "q95": stats["max"],
            }
        )
    return stats


def compute_windowed_fractal_statistics(
    prices: pd.Series,
    returns: pd.Series,
    *,
    window: int,
    stride: int,
    structure_from_levels: bool = False,
) -> dict[str, Any]:
    """Compute fractal summaries on sliding windows aligned with GAF cubes."""

    prices = prices.sort_index()
    returns = returns.sort_index()
    starts = compute_window_starts(len(returns), window, stride)

    metric_values: dict[str, list[float]] = defaultdict(list)
    windows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for start in starts:
        end = start + int(window)
        window_returns = returns.iloc[start:end]
        if len(window_returns) < int(window):
            warnings.append(
                f"Window starting at offset {start} shorter than requested size."
            )
            continue
        start_label = window_returns.index[0]
        start_pos = prices.index.get_indexer([start_label])[0]
        if start_pos == -1:
            warnings.append(
                f"Price index misalignment for window starting {start_label!s}."
            )
            continue
        slice_start = max(0, start_pos - 1)
        slice_end = min(len(prices), start_pos + len(window_returns))
        price_window = prices.iloc[slice_start:slice_end]
        price_window = price_window.dropna()
        if len(price_window) < len(window_returns):
            warnings.append(
                "Insufficient price observations to pair with returns for window "
                f"starting {start_label!s}."
            )
            continue
        try:
            fractal = compute_fractal_metrics(
                price_window,
                window_returns,
                structure_from_levels=structure_from_levels,
            )
        except Exception as exc:  # pragma: no cover - estimator edge cases
            warnings.append(
                "Fractal estimators failed for window starting "
                f"{window_returns.index[0]!s}: {exc}"
            )
            continue

        summary = {
            key: (float(value) if value is not None else None)
            for key, value in fractal.summary.items()
        }
        windows.append(
            {
                "offset": int(start),
                "start": window_returns.index[0],
                "end": window_returns.index[-1],
                "summary": summary,
            }
        )
        for key, value in summary.items():
            if value is None:
                continue
            if not np.isfinite(value):
                continue
            metric_values[key].append(float(value))

    aggregates = {}
    for key, values in metric_values.items():
        distribution = _describe_distribution(values)
        aggregates[key] = {"count": len(values), **distribution}

    return {
        "window": int(window),
        "stride": int(stride),
        "total_windows": len(starts),
        "processed_windows": len(windows),
        "failed_windows": len(starts) - len(windows),
        "aggregates": aggregates,
        "windows": windows,
        "warnings": warnings,
    }



# ──────────────────────────────────────────────────────────────────────────────
# plotting helpers
# ──────────────────────────────────────────────────────────────────────────────


def plot_price_series(
    prices: pd.Series,
    *,
    title: str,
    ylabel: str,
    out_dir: Path,
    filename: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(to_naive(prices.index), prices.to_numpy(), color="#1f77b4", lw=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_dir, filename)


def plot_returns_histogram(
    returns: pd.Series,
    *,
    out_dir: Path,
    filename: str,
    title: str = "Log-returns",
) -> str:
    data = returns.to_numpy() * 100
    idx = to_naive(returns.index)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(idx, data, color="#ff7f0e", lw=0.8)
    axes[0].set_ylabel("Log-return (%)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(data, bins=50, color="#2ca02c", alpha=0.7)
    axes[1].set_xlabel("Log-return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)
    return save_fig(fig, out_dir, filename)


def plot_garch_overlay(
    returns: pd.Series,
    cond_vol: pd.Series,
    *,
    out_dir: Path,
    filename: str,
    title: str = "AR(1)-GARCH(1,1) conditional volatility",
) -> str:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(
        to_naive(returns.index),
        returns.to_numpy() * 100,
        color="grey",
        alpha=0.3,
        width=1.0,
    )
    ax1.set_ylabel("Log-return (%)", color="grey")
    ax1.tick_params(axis="y", labelcolor="grey")
    ax2 = ax1.twinx()
    ax2.plot(to_naive(cond_vol.index), cond_vol * 100, color="#d62728", lw=1.2)
    ax2.set_ylabel("Cond. volatility (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    return save_fig(fig, out_dir, filename)


def _forecast_index(index: pd.DatetimeIndex, steps: int) -> pd.DatetimeIndex:
    if index.size == 0:
        raise ValueError("Cannot extend an empty index")
    if index.size == 1:
        freq = pd.Timedelta(days=1)
    else:
        freq = index[-1] - index[-2]
        if freq <= pd.Timedelta(0):
            freq = pd.Timedelta(days=1)
    start = index[-1] + freq
    return pd.date_range(start=start, periods=steps, freq=freq)


def plot_har_forecast(
    realised_variance: pd.Series,
    forecast_daily_vol: Sequence[float],
    *,
    out_dir: Path,
    filename: str,
    title: str = "HAR-RV realised volatility & forecast",
) -> str:
    rv = pd.Series(realised_variance, dtype=float)
    if rv.empty:
        raise ValueError("Realised variance series is empty; nothing to plot")

    idx = to_naive(rv.index)
    realised_vol = np.sqrt(rv.to_numpy()) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, realised_vol, color="#1f77b4", lw=1.2, label="Realised vol")

    if forecast_daily_vol:
        forecast_daily_vol = [float(v) for v in forecast_daily_vol]
        future_idx = _forecast_index(idx, len(forecast_daily_vol))
        forecast_vol = np.asarray(forecast_daily_vol) * 100
        ax.plot(
            future_idx,
            forecast_vol,
            color="#ff7f0e",
            marker="o",
            linestyle="--",
            label="Forecast",
        )
        ax.axvline(idx[-1], color="grey", linestyle=":", alpha=0.6)

    ax.set_ylabel("Volatility (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_dir, filename)


def plot_mfdfa_spectrum(
    mfdfa_res: dict[str, np.ndarray],
    *,
    out_dir: Path,
    filename: str,
    title: str = "MFDFA singularity spectrum",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mfdfa_res["alpha"], mfdfa_res["f_alpha"], marker="o", lw=1.2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_dir, filename)


def plot_rs_scaling(
    rs_res: dict[str, Any],
    *,
    out_dir: Path,
    filename: str,
    title: str = "Rescaled-range scaling",
) -> str:
    log_scales = np.asarray(rs_res.get("log_scales"), dtype=float)
    log_rs = np.asarray(rs_res.get("log_rs"), dtype=float)
    if log_scales.size == 0 or log_rs.size == 0:
        raise ValueError("RS result missing log-scaled diagnostic data")

    scales = np.exp(log_scales)
    rs_vals = np.exp(log_rs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(scales, rs_vals, marker="o", lw=1.2, label="⟨R/S⟩")

    slope = rs_res.get("H")
    intercept = rs_res.get("intercept")
    if slope is not None and intercept is not None:
        x_fit = np.linspace(log_scales.min(), log_scales.max(), 200)
        y_fit = slope * x_fit + intercept
        ax.loglog(
            np.exp(x_fit),
            np.exp(y_fit),
            color="#d62728",
            linestyle="--",
            label=f"H ≈ {float(slope):.3f}",
        )
        ax.legend()

    ax.set_xlabel("Window size n")
    ax.set_ylabel("⟨R/S⟩")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return save_fig(fig, out_dir, filename)


def plot_dfa_fluctuation(
    dfa_res: dict[str, Any],
    *,
    out_dir: Path,
    filename: str,
    title: str = "DFA fluctuation function",
) -> str:
    log_scales = np.asarray(dfa_res.get("log_scales"), dtype=float)
    log_fluct = np.asarray(dfa_res.get("log_fluct"), dtype=float)
    if log_scales.size == 0 or log_fluct.size == 0:
        raise ValueError("DFA result missing log-scaled diagnostic data")

    scales = np.exp(log_scales)
    fluct = np.exp(log_fluct)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(scales, fluct, marker="o", lw=1.2, label="F(s)")

    slope = dfa_res.get("H")
    intercept = dfa_res.get("intercept")
    fit_start = int(dfa_res.get("fit_start", 0))
    fit_stop = int(dfa_res.get("fit_stop", log_scales.size))
    if (
        slope is not None
        and intercept is not None
        and 0 <= fit_start < fit_stop <= log_scales.size
    ):
        x_fit = log_scales[fit_start:fit_stop]
        y_fit = slope * x_fit + intercept
        ax.loglog(
            np.exp(x_fit),
            np.exp(y_fit),
            color="#d62728",
            linestyle="--",
            label=f"H ≈ {float(slope):.3f}",
        )
        ax.legend()

    ax.set_xlabel("Scale s")
    ax.set_ylabel("Fluctuation F(s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return save_fig(fig, out_dir, filename)


def plot_structure_function_summary(
    struct_res: ScalingResult,
    *,
    out_dir: Path,
    filename: str,
    title: str = "Structure-function diagnostics",
) -> str:
    qs = np.asarray(struct_res.q, dtype=float)
    tau = np.asarray(struct_res.tau, dtype=float)
    alpha = np.asarray(struct_res.alpha, dtype=float)
    f_alpha = np.asarray(struct_res.f_alpha, dtype=float)

    mask_tau = np.isfinite(qs) & np.isfinite(tau)
    mask_spec = np.isfinite(alpha) & np.isfinite(f_alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if mask_tau.any():
        axes[0].plot(qs[mask_tau], tau[mask_tau], marker="o", lw=1.2, label="τ(q)")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("τ(q)")
    axes[0].set_title("Scaling exponents")
    axes[0].grid(True, alpha=0.3)

    if struct_res.H is not None and mask_tau.any():
        q_grid = np.linspace(qs[mask_tau].min(), qs[mask_tau].max(), 200)
        lambda_sq = float(struct_res.lambda_) ** 2
        zeta_fit = struct_res.H * q_grid - 0.5 * lambda_sq * q_grid * (q_grid - 1.0)
        axes[0].plot(
            q_grid,
            zeta_fit,
            color="#d62728",
            linestyle="--",
            label=f"Fit (H ≈ {struct_res.H:.3f})",
        )
        axes[0].legend()

    if mask_spec.any():
        axes[1].plot(
            alpha[mask_spec],
            f_alpha[mask_spec],
            marker="o",
            lw=1.2,
        )
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$f(\alpha)$")
    axes[1].set_title("Singularity spectrum")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, out_dir, filename)


def plot_wtmm_spectrum(
    wtmm_res: dict[str, np.ndarray],
    *,
    out_dir: Path,
    filename: str,
    title: str = "WTMM diagnostics",
) -> str:
    qs = np.asarray(wtmm_res.get("q"), dtype=float)
    tau = np.asarray(wtmm_res.get("tau(q)"), dtype=float)
    alpha = np.asarray(wtmm_res.get("alpha"), dtype=float)
    f_alpha = np.asarray(wtmm_res.get("f(alpha)"), dtype=float)

    mask_tau = np.isfinite(qs) & np.isfinite(tau)
    mask_spec = np.isfinite(alpha) & np.isfinite(f_alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if mask_tau.any():
        axes[0].plot(qs[mask_tau], tau[mask_tau], marker="o", lw=1.2)
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("τ(q)")
    axes[0].set_title("Partition function slopes")
    axes[0].grid(True, alpha=0.3)

    if mask_spec.any():
        axes[1].plot(
            alpha[mask_spec],
            f_alpha[mask_spec],
            marker="o",
            lw=1.2,
        )
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$f(\alpha)$")
    axes[1].set_title("Singularity spectrum")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, out_dir, filename)


# ──────────────────────────────────────────────────────────────────────────────
# timescale inference
# ──────────────────────────────────────────────────────────────────────────────


def infer_periods_per_year(index: pd.Index) -> float:
    """Guess the appropriate annualisation factor from the sample spacing."""

    idx = pd.DatetimeIndex(index).sort_values()
    if idx.size < 2:
        return TRADING_DAYS
    deltas = idx.to_series().diff().dropna().dt.total_seconds().to_numpy()
    median_seconds = float(np.median(deltas)) if deltas.size else 0.0
    day = 24 * 3600

    if median_seconds <= 0:
        return TRADING_DAYS
    if 0.8 * day <= median_seconds <= 1.5 * day:
        return TRADING_DAYS
    if 5 * day <= median_seconds <= 10 * day:
        return 52.0
    if 20 * day <= median_seconds <= 40 * day:
        return 12.0
    if median_seconds < day:
        periods_per_day = day / median_seconds
        return periods_per_day * TRADING_DAYS
    seconds_per_year = 365.25 * day
    return seconds_per_year / median_seconds


__all__ = [
    "TRADING_DAYS",
    "GARCHResult",
    "HARResult",
    "FractalResult",
    "ensure_dir",
    "save_fig",
    "to_naive",
    "annualise",
    "compute_window_starts",

    "h_at",
    "summarise_prices",
    "fit_garch",
    "compute_realised_variance",
    "fit_har_realised_variance",
    "fit_msm",
    "compute_fractal_metrics",
    "compute_windowed_fractal_statistics",

    "plot_price_series",
    "plot_returns_histogram",
    "plot_garch_overlay",
    "plot_har_forecast",
    "plot_mfdfa_spectrum",
    "plot_rs_scaling",
    "plot_dfa_fluctuation",
    "plot_structure_function_summary",
    "plot_wtmm_spectrum",
    "infer_periods_per_year",
]

