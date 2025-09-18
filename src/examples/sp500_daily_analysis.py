"""Comprehensive daily S&P 500 analysis pipeline.

This script downloads S&P 500 closes from Yahoo! Finance, runs the full
volatility and multifractal toolkit used throughout the thesis, saves the
resulting figures under ``analysis_outputs/`` and serialises key statistics to
JSON for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from fractalfinance.estimators import DFA, MFDFA, RS, StructureFunction, WTMM
from fractalfinance.io import load_yahoo
from fractalfinance.models import msm_fit
from fractalfinance.plotting import DEFAULT_OUTPUT_DIR

TRADING_DAYS = 252


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> str:
    _ensure_dir(out_dir)
    target = out_dir / filename
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def _annualise(daily_value: float) -> float:
    return float(daily_value * np.sqrt(TRADING_DAYS))


def _h_at(mfdfa_res: dict[str, np.ndarray], q: float) -> float | None:
    qs = mfdfa_res.get("q")
    hs = mfdfa_res.get("h")
    if qs is None or hs is None:
        return None
    matches = np.where(np.isclose(qs, q))[0]
    if matches.size == 0:
        return None
    return float(hs[matches[0]])


def _to_naive(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.tz_localize(None)


def run(
    *,
    symbol: str = "^GSPC",
    start: str = "2022-01-01",
    end: str | None = None,
    output_subdir: str = "sp500_daily",
) -> dict:
    """Execute the full analysis and return a nested summary dictionary."""

    end = end or pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    output_dir = _ensure_dir(DEFAULT_OUTPUT_DIR / output_subdir)

    prices = load_yahoo(symbol, start=start, end=end, max_retries=6, retry_delay=1.5)
    prices = prices.astype(float)
    returns = np.log(prices).diff().dropna()

    obs = int(len(returns))
    span = f"{prices.index[0].date()} → {prices.index[-1].date()}"
    price_change = float(prices.iloc[-1] / prices.iloc[0] - 1.0)

    ann_return = float(returns.mean() * TRADING_DAYS)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    skew = float(returns.skew())
    kurt = float(returns.kurt())

    scaled_returns = (returns * 100).rename("ret_pct")
    am = arch_model(
        scaled_returns,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    garch_res = am.fit(disp="off")
    cond_vol = garch_res.conditional_volatility / 100.0
    forecasts = garch_res.forecast(horizon=5, reindex=False)
    variance_forecast = forecasts.variance.iloc[-1]
    daily_forecast = [float(np.sqrt(v) / 100.0) for v in variance_forecast]

    garch_summary = {
        "params": {k: float(v) for k, v in garch_res.params.items()},
        "last_cond_vol_daily": float(cond_vol.iloc[-1]),
        "last_cond_vol_annual": _annualise(float(cond_vol.iloc[-1])),
        "forecast_daily_vol": daily_forecast,
        "forecast_annual_vol": [_annualise(v) for v in daily_forecast],
    }

    msm_params = msm_fit(returns.to_numpy(), K=5)
    msm_summary = {
        "sigma2": float(msm_params.sigma2),
        "m_L": float(msm_params.m_L),
        "m_H": float(msm_params.m_H),
        "gamma_1": float(msm_params.gamma_1),
        "b": float(msm_params.b),
        "K": int(msm_params.K),
    }

    rs_res = RS(returns).fit().result_
    dfa_res = DFA(prices, from_levels=True, auto_range=True).fit().result_
    struct_res = StructureFunction(returns, from_levels=False).fit().result_
    mfdfa_res = MFDFA(prices, from_levels=True, auto_range=True).fit().result_
    wtmm_res = WTMM(returns, from_levels=False).fit().result_

    fractal_summary = {
        "RS_H": float(rs_res["H"]),
        "DFA_H": float(dfa_res["H"]),
        "Structure_H": float(struct_res.H),
        "Structure_lambda": float(struct_res.lambda_),
        "Structure_delta_alpha": float(struct_res.delta_alpha),
        "MFDFA_h2": _h_at(mfdfa_res, 2.0),
        "MFDFA_width": float(np.max(mfdfa_res["alpha"]) - np.min(mfdfa_res["alpha"])),
        "WTMM_width": float(np.nanmax(wtmm_res["alpha"]) - np.nanmin(wtmm_res["alpha"])),
    }

    # ── Plots ──────────────────────────────────────────────────────────
    idx = _to_naive(prices.index)
    ret_idx = _to_naive(returns.index)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, prices.to_numpy(), color="#1f77b4", lw=1.5)
    ax.set_title("S&P 500 Daily Close")
    ax.set_ylabel("Index level")
    ax.grid(True, alpha=0.3)
    price_path = _save_fig(fig, output_dir, "sp500_price.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ret_idx, returns.to_numpy() * 100, color="#ff7f0e", lw=0.8)
    axes[0].set_ylabel("Log-return (%)")
    axes[0].set_title("Daily log-returns")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(returns.to_numpy() * 100, bins=50, color="#2ca02c", alpha=0.7)
    axes[1].set_xlabel("Log-return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)
    returns_plot = _save_fig(fig, output_dir, "sp500_returns.png")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(ret_idx, returns.to_numpy() * 100, color="grey", alpha=0.3, width=1.0)
    ax1.set_ylabel("Log-return (%)", color="grey")
    ax1.tick_params(axis="y", labelcolor="grey")
    ax2 = ax1.twinx()
    ax2.plot(_to_naive(cond_vol.index), cond_vol * 100, color="#d62728", lw=1.2)
    ax2.set_ylabel("Cond. volatility (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("AR(1)-GARCH(1,1) conditional volatility")
    ax1.grid(True, alpha=0.3)
    garch_plot = _save_fig(fig, output_dir, "sp500_garch.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mfdfa_res["alpha"], mfdfa_res["f_alpha"], marker="o", lw=1.2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title("MFDFA singularity spectrum")
    ax.grid(True, alpha=0.3)
    mfdfa_plot = _save_fig(fig, output_dir, "sp500_mfdfa.png")

    outputs = {
        "price_path": price_path,
        "returns": returns_plot,
        "garch": garch_plot,
        "mfdfa": mfdfa_plot,
    }

    summary = {
        "symbol": symbol,
        "observations": obs,
        "span": span,
        "price_change": price_change,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "skew": skew,
        "excess_kurtosis": kurt,
        "garch": garch_summary,
        "msm": msm_summary,
        "fractal": fractal_summary,
        "outputs": outputs,
    }

    with open(output_dir / "sp500_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":
    result = run()
    pd.options.display.float_format = "{:.6f}".format
    print(json.dumps(result, indent=2))
