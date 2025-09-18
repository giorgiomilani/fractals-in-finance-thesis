"""Multi-timescale asset analysis combining fractal and GAF diagnostics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fractalfinance.analysis.common import (
    compute_fractal_metrics,
    ensure_dir,
    fit_garch,
    fit_msm,
    infer_periods_per_year,
    plot_garch_overlay,
    plot_mfdfa_spectrum,
    plot_price_series,
    plot_returns_histogram,
    summarise_prices,
)
from fractalfinance.gaf.dataset import GAFWindowDataset
from fractalfinance.gaf.gaf import save_gaf_png
from fractalfinance.io import load_yahoo
from fractalfinance.plotting import DEFAULT_OUTPUT_DIR


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return slug.lower() or "scale"


def _median_spacing_seconds(index: pd.Index) -> float | None:
    idx = pd.DatetimeIndex(index).sort_values()
    if idx.size < 2:
        return None
    diffs = idx.to_series().diff().dropna().dt.total_seconds().to_numpy()
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds <= 0:
        return None
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    if days >= 2.0:
        return f"{days:.1f} days"
    if hours >= 2.0:
        return f"{hours:.1f} hours"
    if minutes >= 2.0:
        return f"{minutes:.1f} minutes"
    return f"{seconds:.0f} seconds"


@dataclass(slots=True)
class GAFScaleConfig:
    """Parameters used to generate GAF cubes for a specific timescale."""

    window: int
    stride: int
    resolutions: Sequence[int]
    kinds: Sequence[str] = ("gasf", "gadf")
    detrend: bool = False
    scale: str = "symmetric"
    resample: str = "paa"
    to_uint8: bool = False
    image_size: int | None = None


@dataclass(slots=True)
class ScaleConfig:
    """Describe how a given Yahoo! ``interval`` should be analysed."""

    interval: str
    label: str
    gaf: GAFScaleConfig
    min_points: int | None = None

    def resolved_min_points(self) -> int:
        base = self.min_points if self.min_points is not None else 0
        return max(base, self.gaf.window + self.gaf.stride)


def default_scale_configs() -> list[ScaleConfig]:
    """Return a conservative set of Yahoo! intervals spanning multiple scales."""

    return [
        ScaleConfig(
            interval="1m",
            label="1-minute",
            gaf=GAFScaleConfig(
                window=780,  # roughly two trading days of minute bars
                stride=60,
                resolutions=(512, 256),
                image_size=256,
            ),
        ),
        ScaleConfig(
            interval="5m",
            label="5-minute",
            gaf=GAFScaleConfig(
                window=720,
                stride=48,
                resolutions=(512, 256),
                image_size=240,
            ),
        ),
        ScaleConfig(
            interval="15m",
            label="15-minute",
            gaf=GAFScaleConfig(
                window=672,
                stride=32,
                resolutions=(384, 256),
                image_size=224,
            ),
        ),
        ScaleConfig(
            interval="1h",
            label="1-hour",
            gaf=GAFScaleConfig(
                window=600,
                stride=24,
                resolutions=(384, 256),
                image_size=224,
            ),
        ),
        ScaleConfig(
            interval="1d",
            label="Daily",
            gaf=GAFScaleConfig(
                window=512,
                stride=16,
                resolutions=(512, 256, 128),
                image_size=256,
            ),
        ),
        ScaleConfig(
            interval="1wk",
            label="Weekly",
            gaf=GAFScaleConfig(
                window=260,
                stride=8,
                resolutions=(256, 128, 64),
                image_size=192,
            ),
        ),
        ScaleConfig(
            interval="1mo",
            label="Monthly",
            gaf=GAFScaleConfig(
                window=180,
                stride=3,
                resolutions=(180, 120, 60),
                image_size=160,
            ),
        ),
    ]


def _gaf_summary(
    returns: pd.Series,
    *,
    config: GAFScaleConfig,
    out_dir: Path,
    slug: str,
) -> tuple[dict[str, object], list[str]]:
    series = returns.to_numpy(dtype=float)
    dataset = GAFWindowDataset(
        series,
        win=config.window,
        stride=config.stride,
        resize=config.resolutions,
        kinds=config.kinds,
        detrend=config.detrend,
        scale=config.scale,
        resample=config.resample,
        to_uint8=config.to_uint8,
        image_size=config.image_size,
    )
    windows = len(dataset)
    label_distribution: dict[str, int] = {}
    sample_images: dict[str, str] = {}
    warnings: list[str] = []

    first_channels = 0

    if windows == 0:
        warnings.append(
            "Insufficient samples to form a single GAF window at this scale."
        )
    else:
        labels = []
        for idx in range(windows):
            _, label_tensor = dataset[idx]
            labels.append(int(label_tensor.item()))
        uniques, counts = np.unique(labels, return_counts=True)
        label_distribution = {
            str(int(u)): int(c) for u, c in zip(uniques, counts, strict=False)
        }
        cube, _ = dataset[0]
        cube_np = cube.detach().cpu().numpy()
        first_channels = int(cube_np.shape[0])
        image_path = out_dir / f"{slug}_gaf_sample.png"
        save_gaf_png(cube_np, image_path)
        sample_images["first_window"] = str(image_path)

    median_spacing = _median_spacing_seconds(returns.index)
    if median_spacing is None:
        window_span = None
    else:
        window_span = median_spacing * config.window

    summary = {
        "window": int(config.window),
        "stride": int(config.stride),
        "resolutions": [int(r) for r in config.resolutions],
        "image_size": int(config.image_size or max(config.resolutions)),
        "kinds": list(config.kinds),
        "channels": first_channels,
        "windows": int(windows),
        "label_distribution": label_distribution,
        "window_span_seconds": window_span,
        "window_span_pretty": _format_duration(window_span),
        "sample_images": sample_images,
    }
    return summary, warnings


def _serialize_timestamp(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts, tz="UTC")
    return ts.isoformat()


def run_scale(
    symbol: str,
    *,
    start: str,
    end: str,
    config: ScaleConfig,
    output_dir: Path,
    max_retries: int = 6,
    retry_delay: float = 1.5,
) -> dict[str, object]:
    """Execute the full pipeline for a single ``interval``."""

    label = config.label
    slug = _slugify(f"{symbol}_{label}")
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
    except Exception as exc:  # pragma: no cover - network errors in CI
        return {
            "label": label,
            "interval": config.interval,
            "error": str(exc),
            "outputs": {},
            "gaf": {},
        }

    prices = prices.astype(float)
    if prices.empty:
        return {
            "label": label,
            "interval": config.interval,
            "warning": "No data returned for this interval.",
            "outputs": {},
            "gaf": {},
        }

    min_points = config.resolved_min_points()
    if len(prices) < min_points:
        return {
            "label": label,
            "interval": config.interval,
            "warning": (
                f"Only {len(prices)} observations available; need at least {min_points} "
                "to build sliding windows."
            ),
            "observations": int(len(prices)),
            "outputs": {},
            "gaf": {
                "window": int(config.gaf.window),
                "stride": int(config.gaf.stride),
            },
        }

    returns = np.log(prices).diff().dropna()
    periods_per_year = infer_periods_per_year(prices.index)
    stats = summarise_prices(prices, returns, periods_per_year=periods_per_year)
    garch = fit_garch(returns, periods_per_year=periods_per_year)
    msm_summary = fit_msm(returns)
    fractal = compute_fractal_metrics(prices, returns)

    price_path = plot_price_series(
        prices,
        title=f"{label} close",
        ylabel="Price",
        out_dir=scale_dir,
        filename=f"{slug}_price.png",
    )
    returns_path = plot_returns_histogram(
        returns,
        out_dir=scale_dir,
        filename=f"{slug}_returns.png",
        title=f"{label} log-returns",
    )
    garch_path = plot_garch_overlay(
        returns,
        garch.conditional_volatility,
        out_dir=scale_dir,
        filename=f"{slug}_garch.png",
    )
    mfdfa_path = plot_mfdfa_spectrum(
        fractal.mfdfa,
        out_dir=scale_dir,
        filename=f"{slug}_mfdfa.png",
        title=f"{label} MFDFA spectrum",
    )

    gaf_summary, gaf_warnings = _gaf_summary(
        returns,
        config=config.gaf,
        out_dir=scale_dir,
        slug=slug,
    )

    summary = {
        "symbol": symbol,
        "label": label,
        "interval": config.interval,
        "periods_per_year": periods_per_year,
        "data_start": _serialize_timestamp(prices.index[0]),
        "data_end": _serialize_timestamp(prices.index[-1]),
        **stats,
        "garch": garch.summary,
        "msm": msm_summary,
        "fractal": fractal.summary,
        "gaf": gaf_summary,
        "outputs": {
            "price": price_path,
            "returns": returns_path,
            "garch": garch_path,
            "mfdfa": mfdfa_path,
            **gaf_summary.get("sample_images", {}),
        },
    }

    warnings: list[str] = []
    if gaf_warnings:
        warnings.extend(gaf_warnings)
    if warnings:
        summary["warnings"] = warnings

    summary_path = scale_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def run_multi_scale_analysis(
    symbol: str,
    *,
    start: str = "1990-01-01",
    end: str | None = None,
    scales: Sequence[ScaleConfig] | None = None,
    output_subdir: str = "multi_scale",
    max_retries: int = 6,
    retry_delay: float = 1.5,
) -> dict[str, dict[str, object]]:
    """Execute the analysis pipeline for every requested timescale."""

    end = end or pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    output_dir = ensure_dir(DEFAULT_OUTPUT_DIR / output_subdir)
    configs = list(scales or default_scale_configs())

    results: dict[str, dict[str, object]] = {}
    for cfg in configs:
        summary = run_scale(
            symbol,
            start=start,
            end=end,
            config=cfg,
            output_dir=output_dir,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        slug = _slugify(f"{symbol}_{cfg.label}")
        results[slug] = summary

    master_path = output_dir / f"{_slugify(symbol)}_multi_scale_summary.json"
    with open(master_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return {
        "results": results,
        "summary_path": str(master_path),
    }


__all__ = [
    "GAFScaleConfig",
    "ScaleConfig",
    "default_scale_configs",
    "run_scale",
    "run_multi_scale_analysis",
]

