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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from fractalfinance.estimators import DFA, MFDFA, RS, StructureFunction, WTMM
from fractalfinance.io import load_yahoo
from fractalfinance.models import msm_fit
from fractalfinance.plotting import DEFAULT_OUTPUT_DIR


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> str:
    _ensure_dir(out_dir)
    target = out_dir / filename
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def _annualise(daily_value: float, days_per_year: float) -> float:
    return float(daily_value * np.sqrt(days_per_year))


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
    output_dir = _ensure_dir(DEFAULT_OUTPUT_DIR / output_subdir)

    prices = config.loader(
        config.symbol,
        start=config.start,
        end=end,
        **config.loader_kwargs,
    )
    prices = prices.astype(float)
    returns = np.log(prices).diff().dropna()

    obs = int(len(returns))
    span = f"{prices.index[0].date()} → {prices.index[-1].date()}"
    price_change = float(prices.iloc[-1] / prices.iloc[0] - 1.0)

    ann_return = float(returns.mean() * config.annualisation_days)
    ann_vol = float(
        returns.std(ddof=1) * np.sqrt(config.annualisation_days)
    )
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
        "last_cond_vol_annual": _annualise(
            float(cond_vol.iloc[-1]),
            config.annualisation_days,
        ),
        "forecast_daily_vol": daily_forecast,
        "forecast_annual_vol": [
            _annualise(v, config.annualisation_days) for v in daily_forecast
        ],
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
        "MFDFA_width": float(
            np.max(mfdfa_res["alpha"]) - np.min(mfdfa_res["alpha"])
        ),
        "WTMM_width": float(
            np.nanmax(wtmm_res["alpha"]) - np.nanmin(wtmm_res["alpha"])
        ),
    }

    idx = _to_naive(prices.index)
    ret_idx = _to_naive(returns.index)

    price_title = config.price_title or f"{config.label} daily close"
    slug = config.slug

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, prices.to_numpy(), color="#1f77b4", lw=1.5)
    ax.set_title(price_title)
    ax.set_ylabel(config.price_ylabel)
    ax.grid(True, alpha=0.3)
    price_path = _save_fig(fig, output_dir, f"{slug}_price.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ret_idx, returns.to_numpy() * 100, color="#ff7f0e", lw=0.8)
    axes[0].set_ylabel("Log-return (%)")
    axes[0].set_title("Daily log-returns")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(returns.to_numpy() * 100, bins=50, color="#2ca02c", alpha=0.7)
    axes[1].set_xlabel("Log-return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)
    returns_plot = _save_fig(fig, output_dir, f"{slug}_returns.png")

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
    garch_plot = _save_fig(fig, output_dir, f"{slug}_garch.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mfdfa_res["alpha"], mfdfa_res["f_alpha"], marker="o", lw=1.2)
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("MFDFA singularity spectrum")
    ax.grid(True, alpha=0.3)
    mfdfa_plot = _save_fig(fig, output_dir, f"{slug}_mfdfa.png")

    outputs = {
        "price_path": price_path,
        "returns": returns_plot,
        "garch": garch_plot,
        "mfdfa": mfdfa_plot,
    }

    summary = {
        "key": config.key,
        "label": config.label,
        "symbol": config.symbol,
        "asset_type": config.asset_type,
        "observations": obs,
        "span": span,
        "price_change": price_change,
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "annualisation_days": config.annualisation_days,
        "skew": skew,
        "excess_kurtosis": kurt,
        "garch": garch_summary,
        "msm": msm_summary,
        "fractal": fractal_summary,
        "outputs": outputs,
        "notes": config.notes,
    }

    summary_path = output_dir / f"{slug}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary["summary_path"] = str(summary_path)
    return summary


def _default_configs() -> list[AssetRunConfig]:
    return [
        AssetRunConfig(
            key="bitcoin",
            symbol="BTC-USD",
            label="Bitcoin / US Dollar",
            asset_type="crypto",
            start="2020-01-01",
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
            start="2020-01-01",
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
            start="2020-01-01",
            annualisation_days=252.0,
            price_ylabel="Price (USD)",
            output_subdir="apple_daily",
        ),
        AssetRunConfig(
            key="bond",
            symbol="TLT",
            label="iShares 20+ Year Treasury Bond ETF",
            asset_type="bond",
            start="2020-01-01",
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
        master_dir = _ensure_dir(DEFAULT_OUTPUT_DIR / base_output_subdir)
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

