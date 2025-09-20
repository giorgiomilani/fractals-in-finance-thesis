"""Multi-asset, multi-scale Gramian Angular Field generation without fractal models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fractalfinance.analysis.common import ensure_dir
from fractalfinance.io import load_yahoo
from fractalfinance.plotting import DEFAULT_OUTPUT_DIR

from .multi_scale_analysis import (
    GAFScaleConfig,
    ScaleConfig,
    default_scale_configs,
    gaf_summary,
    serialize_timestamp,
    slugify,
)


@dataclass(slots=True)
class AssetConfig:
    """Describe a single asset to analyse via GAF."""

    key: str
    symbol: str
    start: str
    end: str | None = None


DEFAULT_LONG_LOOKBACK = "1900-01-01"


def default_assets() -> list[AssetConfig]:
    """Return the five assets used in the thesis multi-asset runs."""

    return [
        AssetConfig(key="sp500", symbol="^GSPC", start=DEFAULT_LONG_LOOKBACK),
        AssetConfig(key="bitcoin", symbol="BTC-USD", start=DEFAULT_LONG_LOOKBACK),
        AssetConfig(key="forex", symbol="EURUSD=X", start=DEFAULT_LONG_LOOKBACK),
        AssetConfig(key="apple", symbol="AAPL", start=DEFAULT_LONG_LOOKBACK),
        AssetConfig(key="bond", symbol="TLT", start=DEFAULT_LONG_LOOKBACK),
    ]


def _serialise_span(index: pd.Index) -> str:
    idx = pd.DatetimeIndex(index).sort_values()
    if idx.empty:
        return ""
    return f"{idx[0].date()} → {idx[-1].date()}"


def _compute_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def run_gaf_scale(
    symbol: str,
    *,
    start: str,
    end: str,
    config: ScaleConfig,
    output_dir: Path,
    max_retries: int = 6,
    retry_delay: float = 1.5,
) -> dict[str, object]:
    """Generate GAF cubes for a single Yahoo! Finance interval."""

    slug = slugify(f"{symbol}_{config.label}")
    scale_dir = ensure_dir(output_dir / slug)

    try:
        prices = load_yahoo(
            symbol,
            start=start,
            end=end,
            interval=config.interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    except Exception as exc:  # pragma: no cover - network dependencies
        return {
            "label": config.label,
            "interval": config.interval,
            "error": str(exc),
            "warnings": [str(exc)],
            "gaf": {},
            "outputs": {},
        }

    prices = prices.astype(float).dropna()
    if prices.empty:
        return {
            "label": config.label,
            "interval": config.interval,
            "warnings": ["No data returned for this interval."],
            "gaf": {},
            "outputs": {},
        }

    warnings: list[str] = []
    min_points = config.resolved_min_points()
    if len(prices) < min_points:
        warnings.append(
            "Only {count} observations available; {needed} recommended to populate"
            " {window}-point windows.".format(
                count=len(prices),
                needed=min_points,
                window=config.gaf.window,
            )
        )

    returns = _compute_returns(prices)
    gaf_info, gaf_warnings = gaf_summary(
        returns,
        config=config.gaf,
        out_dir=scale_dir,
        slug=slug,
    )

    if gaf_warnings:
        warnings.extend(gaf_warnings)

    summary: dict[str, object] = {
        "symbol": symbol,
        "label": config.label,
        "interval": config.interval,
        "observations": int(len(prices)),
        "span": _serialise_span(prices.index),
        "data_start": serialize_timestamp(prices.index[0]),
        "data_end": serialize_timestamp(prices.index[-1]),
        "gaf": gaf_info,
        "outputs": gaf_info.get("sample_images", {}),
    }

    if warnings:
        summary["warnings"] = warnings

    summary_path = scale_dir / "gaf_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def run_multi_scale_gaf(
    symbol: str,
    *,
    start: str,
    end: str | None = None,
    scales: Sequence[ScaleConfig] | None = None,
    output_subdir: str = "multi_scale_gaf",
    include_intraday: bool = True,
    max_retries: int = 6,
    retry_delay: float = 1.5,
) -> dict[str, object]:
    """Run the GAF-only workflow across all configured scales."""

    end = end or pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    output_dir = ensure_dir(DEFAULT_OUTPUT_DIR / output_subdir)

    configs = list(scales or default_scale_configs())
    if not include_intraday:
        skip = {"1m", "5m", "15m", "1h"}
        configs = [cfg for cfg in configs if cfg.interval not in skip]

    results: dict[str, dict[str, object]] = {}
    for cfg in configs:
        summary = run_gaf_scale(
            symbol,
            start=start,
            end=end,
            config=cfg,
            output_dir=output_dir,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        slug = slugify(f"{symbol}_{cfg.label}")
        results[slug] = summary

    comparison_rows: list[dict[str, object]] = []
    image_recommendations: list[dict[str, object]] = []

    for summary in results.values():
        if not isinstance(summary, dict):
            continue

        gaf_info = summary.get("gaf") or {}
        if not gaf_info:
            continue

        row = {
            "label": summary.get("label"),
            "interval": summary.get("interval"),
            "observations": summary.get("observations"),
            "span": summary.get("span"),
            "gaf_windows": gaf_info.get("windows"),
            "gaf_channels": gaf_info.get("channels"),
            "gaf_image_size": gaf_info.get("image_size"),
            "gaf_window_span_pretty": gaf_info.get("window_span_pretty"),
            "gaf_label_distribution": gaf_info.get("label_distribution"),
            "gaf_label_distribution_share": gaf_info.get("label_distribution_share"),
            "gaf_label_mode": gaf_info.get("label_mode"),
            "gaf_neutral_threshold": gaf_info.get("neutral_threshold"),
            "gaf_quantile_bins": gaf_info.get("quantile_bins"),
        }
        comparison_rows.append(row)

        evaluation = gaf_info.get("image_evaluation") or {}
        if evaluation and evaluation.get("status") != "ok":
            recommendation = {
                "label": summary.get("label"),
                "interval": summary.get("interval"),
                **evaluation,
            }
            image_recommendations.append(recommendation)

    master_summary = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "scales": comparison_rows,
        "image_recommendations": image_recommendations,
        "results": results,
    }

    master_path = output_dir / f"{slugify(symbol)}_multi_scale_gaf_summary.json"
    with open(master_path, "w", encoding="utf-8") as fh:
        json.dump(master_summary, fh, indent=2)
    master_summary["summary_path"] = str(master_path)

    return master_summary


def run_multi_asset_gaf(
    assets: Sequence[AssetConfig] | None = None,
    *,
    base_output_subdir: str = "multi_asset_gaf",
    include_intraday: bool = True,
    max_retries: int = 6,
    retry_delay: float = 1.5,
) -> dict[str, object]:
    """Run the GAF-only workflow for the supplied assets."""

    asset_list = list(assets or default_assets())
    base_dir = ensure_dir(DEFAULT_OUTPUT_DIR / base_output_subdir)
    combined_results: dict[str, object] = {}
    recommendations: list[dict[str, object]] = []

    for asset in asset_list:
        subdir = Path(base_output_subdir) / asset.key
        result = run_multi_scale_gaf(
            asset.symbol,
            start=asset.start,
            end=asset.end,
            output_subdir=str(subdir),
            include_intraday=include_intraday,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        combined_results[asset.key] = result
        recs = result.get("image_recommendations") or []
        for rec in recs:
            rec_with_asset = {"asset": asset.key, **rec}
            recommendations.append(rec_with_asset)

    overview = {
        "base_output_dir": str(base_dir),
        "assets": [asset.__dict__ for asset in asset_list],
        "results": combined_results,
        "image_recommendations": recommendations,
    }

    overview_path = base_dir / "multi_asset_gaf_summary.json"
    with open(overview_path, "w", encoding="utf-8") as fh:
        json.dump(overview, fh, indent=2)
    overview["summary_path"] = str(overview_path)

    return overview


__all__ = [
    "AssetConfig",
    "GAFScaleConfig",
    "ScaleConfig",
    "default_assets",
    "default_scale_configs",
    "run_gaf_scale",
    "run_multi_scale_gaf",
    "run_multi_asset_gaf",
]
