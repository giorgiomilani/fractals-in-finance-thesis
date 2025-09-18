"""Comprehensive daily S&P 500 analysis pipeline.

This script downloads S&P 500 closes from Yahoo! Finance, runs the full
volatility and multifractal toolkit used throughout the thesis, saves the
resulting figures under ``analysis_outputs/`` and serialises key statistics to
JSON for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fractalfinance.analysis.common import (
    TRADING_DAYS,
    compute_fractal_metrics,
    ensure_dir,
    fit_garch,
    fit_msm,
    plot_garch_overlay,
    plot_mfdfa_spectrum,
    plot_price_series,
    plot_returns_histogram,
    summarise_prices,
)
from fractalfinance.io import load_yahoo
from fractalfinance.plotting import DEFAULT_OUTPUT_DIR


def run(
    *,
    symbol: str = "^GSPC",
    start: str = "2022-01-01",
    end: str | None = None,
    output_subdir: str = "sp500_daily",
) -> dict:
    """Execute the full analysis and return a nested summary dictionary."""

    end = end or pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    output_dir = ensure_dir(DEFAULT_OUTPUT_DIR / output_subdir)

    prices = load_yahoo(symbol, start=start, end=end, max_retries=6, retry_delay=1.5)
    prices = prices.astype(float)
    returns = np.log(prices).diff().dropna()

    stats = summarise_prices(prices, returns, periods_per_year=TRADING_DAYS)
    garch = fit_garch(returns, periods_per_year=TRADING_DAYS)
    msm_summary = fit_msm(returns)
    fractal = compute_fractal_metrics(prices, returns)

    price_path = plot_price_series(
        prices,
        title="S&P 500 Daily Close",
        ylabel="Index level",
        out_dir=output_dir,
        filename="sp500_price.png",
    )
    returns_plot = plot_returns_histogram(
        returns,
        out_dir=output_dir,
        filename="sp500_returns.png",
        title="Daily log-returns",
    )
    garch_plot = plot_garch_overlay(
        returns,
        garch.conditional_volatility,
        out_dir=output_dir,
        filename="sp500_garch.png",
    )
    mfdfa_plot = plot_mfdfa_spectrum(
        fractal.mfdfa,
        out_dir=output_dir,
        filename="sp500_mfdfa.png",
    )

    outputs = {
        "price_path": price_path,
        "returns": returns_plot,
        "garch": garch_plot,
        "mfdfa": mfdfa_plot,
    }

    summary = {
        "symbol": symbol,
        **stats,
        "garch": garch.summary,
        "msm": msm_summary,
        "fractal": fractal.summary,
        "outputs": outputs,
    }

    with open(output_dir / "sp500_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":
    result = run()
    pd.options.display.float_format = "{:.6f}".format
    print(json.dumps(result, indent=2))
