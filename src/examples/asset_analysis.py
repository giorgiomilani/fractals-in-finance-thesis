"""Multi-asset analysis runners.

This module generalises the comprehensive workflow implemented for the
S&P 500 so it can be reused with other instruments.  Each asset run downloads
closing prices, computes return statistics, estimates GARCH and MSM models,
derives fractal diagnostics, and saves a consistent set of figures.  The
behaviour is customised via :class:`AssetRunConfig` instances which capture the
nuances of each asset class (e.g. trading calendars for annualisation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from fractalfinance.analysis.common import (
    compute_fractal_metrics,
    ensure_dir,
    fit_garch,
    fit_msm,
    plot_dfa_fluctuation,
    plot_garch_overlay,
    plot_mfdfa_spectrum,
    plot_price_series,
    plot_rs_scaling,
    plot_returns_histogram,
    plot_structure_function_summary,
    plot_wtmm_spectrum,
    summarise_prices,
)
from fractalfinance.io import load_yahoo
from fractalfinance.plotting import DEFAULT_OUTPUT_DIR


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return slug.lower() or "asset"


AssetLoader = Callable[..., pd.Series]


@dataclass(slots=True)
class AssetRunConfig:
    """Configuration describing how to analyse a specific asset."""

    key: str
    symbol: str
    label: str
    asset_type: str
    start: str
    end: str | None = None
    output_subdir: str | None = None
    annualisation_days: float = 252.0
    price_title: str | None = None
    price_ylabel: str = "Close"
    loader: AssetLoader = load_yahoo
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def resolved_output_subdir(self, base_output_subdir: str | None) -> str:
        subdir = self.output_subdir or self.key
        if base_output_subdir:
            return str(Path(base_output_subdir) / subdir)
        return subdir

    @property
    def slug(self) -> str:
        return _slugify(self.key)

    def with_overrides(self, **overrides: Any) -> "AssetRunConfig":
        """Return a copy with the provided attributes replaced."""

        if "loader_kwargs" not in overrides:
            overrides["loader_kwargs"] = dict(self.loader_kwargs)
        return replace(self, **overrides)


def run_asset_analysis(
    config: AssetRunConfig,
    *,
    base_output_subdir: str | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end analysis for ``config`` and return a summary."""

    end = config.end or pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    output_subdir = config.resolved_output_subdir(base_output_subdir)
    output_dir = ensure_dir(DEFAULT_OUTPUT_DIR / output_subdir)

    prices = config.loader(
        config.symbol,
        start=config.start,
        end=end,
        **config.loader_kwargs,
    )
    prices = prices.astype(float)
    returns = np.log(prices).diff().dropna()
    stats = summarise_prices(
        prices,
        returns,
        periods_per_year=config.annualisation_days,
    )
    garch = fit_garch(returns, periods_per_year=config.annualisation_days)
    msm_summary = fit_msm(returns)
    fractal = compute_fractal_metrics(prices, returns)

    price_title = config.price_title or f"{config.label} daily close"
    slug = config.slug

    price_path = plot_price_series(
        prices,
        title=price_title,
        ylabel=config.price_ylabel,
        out_dir=output_dir,
        filename=f"{slug}_price.png",
    )
    returns_plot = plot_returns_histogram(
        returns,
        out_dir=output_dir,
        filename=f"{slug}_returns.png",
        title="Log-returns",
    )
    garch_plot = plot_garch_overlay(
        returns,
        garch.conditional_volatility,
        out_dir=output_dir,
        filename=f"{slug}_garch.png",
    )
    mfdfa_plot = plot_mfdfa_spectrum(
        fractal.mfdfa,
        out_dir=output_dir,
        filename=f"{slug}_mfdfa.png",
    )
    rs_plot = plot_rs_scaling(
        fractal.rs,
        out_dir=output_dir,
        filename=f"{slug}_rs.png",
    )
    dfa_plot = plot_dfa_fluctuation(
        fractal.dfa,
        out_dir=output_dir,
        filename=f"{slug}_dfa.png",
    )
    structure_plot = plot_structure_function_summary(
        fractal.structure,
        out_dir=output_dir,
        filename=f"{slug}_structure.png",
    )
    wtmm_plot = plot_wtmm_spectrum(
        fractal.wtmm,
        out_dir=output_dir,
        filename=f"{slug}_wtmm.png",
    )

    outputs = {
        "price_path": price_path,
        "returns": returns_plot,
        "garch": garch_plot,
        "mfdfa": mfdfa_plot,
        "rs": rs_plot,
        "dfa": dfa_plot,
        "structure": structure_plot,
        "wtmm": wtmm_plot,
    }

    summary = {
        "key": config.key,
        "label": config.label,
        "symbol": config.symbol,
        "asset_type": config.asset_type,
        **stats,
        "annualisation_days": config.annualisation_days,
        "garch": garch.summary,
        "msm": msm_summary,
        "fractal": fractal.summary,
        "outputs": outputs,
        "notes": config.notes,
    }

    summary_path = output_dir / f"{slug}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary["summary_path"] = str(summary_path)
    return summary


def _default_configs() -> list[AssetRunConfig]:
    default_start = "1900-01-01"

    return [
        AssetRunConfig(
            key="sp500",
            symbol="^GSPC",
            label="S&P 500 Index",
            asset_type="equity_index",
            start=default_start,
            annualisation_days=252.0,
            price_ylabel="Index level",
            output_subdir="sp500_daily",
            notes=(
                "Benchmark US equity index used alongside the other core asset"
                " classes in the thesis multi-asset comparison."
            ),
        ),
        AssetRunConfig(
            key="bitcoin",
            symbol="BTC-USD",
            label="Bitcoin / US Dollar",
            asset_type="crypto",
            start=default_start,
            annualisation_days=365.0,
            price_ylabel="Price (USD)",
            output_subdir="bitcoin_daily",
            notes=(
                "Crypto trades 24/7 so a 365-day scaling factor is used for"
                " annualised metrics."
            ),
        ),
        AssetRunConfig(
            key="forex",
            symbol="EURUSD=X",
            label="EUR/USD",
            asset_type="forex",
            start=default_start,
            annualisation_days=260.0,
            price_ylabel="USD per EUR",
            output_subdir="fx_eurusd_daily",
            notes=(
                "Major FX pairs observe a 5-day trading week; 260 trading days"
                " approximate the annualisation horizon."
            ),
        ),
        AssetRunConfig(
            key="apple",
            symbol="AAPL",
            label="Apple Inc.",
            asset_type="equity",
            start=default_start,
            annualisation_days=252.0,
            price_ylabel="Price (USD)",
            output_subdir="apple_daily",
        ),
        AssetRunConfig(
            key="bond",
            symbol="TLT",
            label="iShares 20+ Year Treasury Bond ETF",
            asset_type="bond",
            start=default_start,
            annualisation_days=252.0,
            price_ylabel="Price (USD)",
            output_subdir="bond_tlt_daily",
            notes=(
                "TLT serves as a liquid proxy for long-duration US Treasury"
                " exposure."
            ),
        ),
    ]


DEFAULT_ASSET_CONFIGS = _default_configs()


def run_default_assets(
    *,
    base_output_subdir: str = "multi_asset",
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    write_master_summary: bool = True,
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    """Execute the bundle of default asset runs.

    Parameters
    ----------
    base_output_subdir : str, default "multi_asset"
        Folder inside :data:`DEFAULT_OUTPUT_DIR` where per-asset results will be
        written. Each asset keeps its dedicated subdirectory to avoid name
        clashes.
    overrides : mapping, optional
        Mapping keyed by ``AssetRunConfig.key`` whose values contain attribute
        overrides applied before executing the run. This allows CLI callers to
        tweak symbols or date ranges without rebuilding the defaults.
    write_master_summary : bool, default True
        When ``True`` a JSON file aggregating all asset summaries is written to
        the base output directory.

    Returns
    -------
    tuple(dict, Path | None)
        A dictionary keyed by asset ``key`` with the summary dictionaries
        returned by :func:`run_asset_analysis`, and the path of the master
        summary file when requested.
    """

    overrides = overrides or {}
    results: dict[str, dict[str, Any]] = {}

    for cfg in DEFAULT_ASSET_CONFIGS:
        cfg_overrides = overrides.get(cfg.key, {})
        updated_cfg = cfg.with_overrides(**cfg_overrides)
        results[cfg.key] = run_asset_analysis(
            updated_cfg,
            base_output_subdir=base_output_subdir,
        )

    master_path: Path | None = None
    if write_master_summary:
        master_dir = ensure_dir(DEFAULT_OUTPUT_DIR / base_output_subdir)
        master_path = master_dir / "multi_asset_summary.json"
        with open(master_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    return results, master_path


__all__ = [
    "AssetRunConfig",
    "DEFAULT_ASSET_CONFIGS",
    "run_asset_analysis",
    "run_default_assets",
]

