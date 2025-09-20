"""Multi-timescale asset analysis combining fractal and GAF diagnostics."""

from __future__ import annotations

import math

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fractalfinance.analysis.common import (
    compute_fractal_metrics,
    compute_windowed_fractal_statistics,

    ensure_dir,
    fit_garch,
    fit_msm,
    infer_periods_per_year,
    plot_dfa_fluctuation,
    plot_garch_overlay,
    plot_mfdfa_spectrum,
    plot_price_series,
    plot_rs_scaling,
    plot_returns_histogram,
    plot_structure_function_summary,
    plot_wtmm_spectrum,
    plot_windowed_metric_distribution,

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


def _normalise_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _gaf_image_evaluation(
    *,
    config: GAFScaleConfig,
    actual_size: int,
    window: int,
) -> dict[str, object]:
    max_resolution = max(int(r) for r in config.resolutions)
    configured = int(config.image_size or max_resolution)
    ratio = actual_size / max_resolution if max_resolution else None
    points_per_pixel = window / actual_size if actual_size else None

    evaluation: dict[str, object] = {
        "largest_resolution": max_resolution,
        "configured_image_size": configured,
        "actual_image_size": int(actual_size),
        "scale_ratio": float(ratio) if ratio is not None else None,
        "points_per_pixel": float(points_per_pixel)
        if points_per_pixel is not None
        else None,
        "status": "ok",
    }

    if actual_size < max_resolution:
        recommended = max(
            max_resolution,
            int(2 ** math.ceil(math.log2(max_resolution))),
        )
        evaluation.update(
            {
                "status": "increase",
                "recommended_image_size": recommended,
                "message": (
                    "Actual image size is smaller than the highest-resolution "
                    "PAA slice; consider increasing the output image size for "
                    "finer detail."
                ),
            }
        )
    elif actual_size > max_resolution * 2:
        evaluation.update(
            {
                "status": "oversized",
                "recommended_image_size": max_resolution * 2,
                "message": (
                    "Configured image size greatly exceeds the base resolution; "
                    "downsampling may not add fidelity and increases storage."
                ),
            }
        )

    return evaluation


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

    actual_image_size = config.image_size or max(config.resolutions)


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
        actual_image_size = int(cube_np.shape[-1])
        first_channels = int(cube_np.shape[0])
        image_path = out_dir / f"{slug}_gaf_sample.png"
        if first_channels >= 3:
            select = [0, 1, 2]
        else:
            select = [0]
        save_gaf_png(cube_np, image_path, select_channels=select)
        sample_images["first_window_channels"] = select

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
        "image_size": int(actual_image_size),

        "kinds": list(config.kinds),
        "channels": first_channels,
        "windows": int(windows),
        "label_distribution": label_distribution,
        "label_distribution_share": _normalise_counts(label_distribution),
        "window_span_seconds": window_span,
        "window_span_pretty": _format_duration(window_span),
        "sample_images": sample_images,
        "image_evaluation": _gaf_image_evaluation(
            config=config,
            actual_size=actual_image_size,
            window=config.window,
        ),

    }
    return summary, warnings


def _serialize_timestamp(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

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
            "warnings": [str(exc)],
        }

    prices = prices.astype(float).dropna()
    warnings: list[str] = []

    if prices.empty:
        return {
            "label": label,
            "interval": config.interval,
            "warnings": ["No data returned for this interval."],

            "outputs": {},
            "gaf": {},
        }

    min_points = config.resolved_min_points()
    if len(prices) < min_points:
        warnings.append(
            "Only {count} observations available; {needed} recommended to "
            "populate {window}-point windows.".format(
                count=len(prices),
                needed=min_points,
                window=config.gaf.window,
            )
        )


    returns = np.log(prices).diff().dropna()
    periods_per_year = infer_periods_per_year(prices.index)
    stats = summarise_prices(prices, returns, periods_per_year=periods_per_year)

    returns_count = len(returns)

    garch_result = None
    garch_error = None
    if returns_count < 10:
        garch_error = (
            f"Insufficient return observations for GARCH (have {returns_count}, need ≥10)."
        )
        warnings.append(garch_error)
    else:
        try:
            garch_result = fit_garch(returns, periods_per_year=periods_per_year)
        except Exception as exc:  # pragma: no cover - estimator edge cases
            garch_error = str(exc)
            warnings.append(f"GARCH fit failed: {exc}")

    msm_summary: dict[str, object] | None = None
    msm_error = None
    if returns_count < 50:
        msm_error = (
            f"Insufficient return observations for MSM (have {returns_count}, need ≥50)."
        )
        warnings.append(msm_error)
    else:
        try:
            msm_summary = fit_msm(returns)
        except Exception as exc:  # pragma: no cover - estimator edge cases
            msm_error = str(exc)
            warnings.append(f"MSM fit failed: {exc}")

    fractal_result = None
    fractal_error = None
    if returns_count < 32:
        fractal_error = (
            f"Insufficient return observations for fractal metrics (have {returns_count}, need ≥32)."
        )
        warnings.append(fractal_error)
    else:
        try:
            fractal_result = compute_fractal_metrics(prices, returns)
        except Exception as exc:  # pragma: no cover - estimator edge cases
            fractal_error = str(exc)
            warnings.append(f"Fractal estimators failed: {exc}")

    fractal_windowed = compute_windowed_fractal_statistics(
        prices,
        returns,
        window=config.gaf.window,
        stride=config.gaf.stride,
        return_samples=True,
    )

    fractal_windows_serialised: list[dict[str, object]] = []
    for record in fractal_windowed.get("windows", []):
        start_ts = record.get("start")
        end_ts = record.get("end")
        serialised_record = dict(record)
        if start_ts is not None:
            serialised_record["start"] = _serialize_timestamp(start_ts)
        if end_ts is not None:
            serialised_record["end"] = _serialize_timestamp(end_ts)
        fractal_windows_serialised.append(serialised_record)
    fractal_windowed["windows"] = fractal_windows_serialised

    outputs: dict[str, str] = {}

    price_path = plot_price_series(
        prices,
        title=f"{label} close",
        ylabel="Price",
        out_dir=scale_dir,
        filename=f"{slug}_price.png",
    )
    outputs["price"] = price_path

    returns_path = plot_returns_histogram(
        returns,
        out_dir=scale_dir,
        filename=f"{slug}_returns.png",
        title=f"{label} log-returns",
    )
    outputs["returns"] = returns_path
    if garch_result is not None:
        garch_path = plot_garch_overlay(
            returns,
            garch_result.conditional_volatility,
            out_dir=scale_dir,
            filename=f"{slug}_garch.png",
        )
        outputs["garch"] = garch_path
    if fractal_result is not None:
        mfdfa_path = plot_mfdfa_spectrum(
            fractal_result.mfdfa,
            out_dir=scale_dir,
            filename=f"{slug}_mfdfa.png",
            title=f"{label} MFDFA spectrum",
        )
        outputs["mfdfa"] = mfdfa_path
        rs_path = plot_rs_scaling(
            fractal_result.rs,
            out_dir=scale_dir,
            filename=f"{slug}_rs.png",
            title=f"{label} R/S scaling",
        )
        outputs["rs"] = rs_path
        dfa_path = plot_dfa_fluctuation(
            fractal_result.dfa,
            out_dir=scale_dir,
            filename=f"{slug}_dfa.png",
            title=f"{label} DFA fluctuation",
        )
        outputs["dfa"] = dfa_path
        structure_path = plot_structure_function_summary(
            fractal_result.structure,
            out_dir=scale_dir,
            filename=f"{slug}_structure.png",
            title=f"{label} structure-function",
        )
        outputs["structure"] = structure_path
        wtmm_path = plot_wtmm_spectrum(
            fractal_result.wtmm,
            out_dir=scale_dir,
            filename=f"{slug}_wtmm.png",
            title=f"{label} WTMM spectrum",
        )
        outputs["wtmm"] = wtmm_path


    gaf_summary, gaf_warnings = _gaf_summary(
        returns,
        config=config.gaf,
        out_dir=scale_dir,
        slug=slug,
    )

    samples = fractal_windowed.pop("samples", {})
    distribution_plots: dict[str, str] = {}
    for metric_name, values in samples.items():
        if not values:
            continue
        metric_slug = _slugify(metric_name)
        try:
            dist_path = plot_windowed_metric_distribution(
                values,
                metric=metric_name,
                out_dir=scale_dir,
                filename=f"{slug}_{metric_slug}_distribution.png",
                title=f"{label} windowed {metric_name}",
            )
        except ValueError:
            continue
        outputs[f"fractal_windowed_{metric_slug}_distribution"] = dist_path
        distribution_plots[metric_name] = dist_path

    if distribution_plots:
        fractal_windowed["distribution_plots"] = distribution_plots

    if gaf_summary.get("windows") is not None:
        fractal_windowed["expected_gaf_windows"] = int(gaf_summary["windows"])

    if (
        fractal_windowed.get("processed_windows")
        and gaf_summary.get("windows") is not None
        and fractal_windowed["processed_windows"] != gaf_summary["windows"]
    ):
        diff_warn = (
            "Windowed fractal metrics processed"
            f" {fractal_windowed['processed_windows']} samples,"
            f" but GAF generated {gaf_summary['windows']} windows."
        )
        fractal_windowed.setdefault("warnings", []).append(diff_warn)

    garch_summary: dict[str, object] | None
    if garch_result is not None:
        garch_summary = garch_result.summary
    elif garch_error is not None:
        garch_summary = {"error": garch_error}
    else:
        garch_summary = None

    if garch_summary is None:
        outputs.pop("garch", None)

    if msm_summary is None and msm_error is not None:
        msm_summary = {"error": msm_error}

    fractal_summary: dict[str, object] | None
    if fractal_result is not None:
        fractal_summary = fractal_result.summary
    elif fractal_error is not None:
        fractal_summary = {"error": fractal_error}
    else:
        fractal_summary = None

    summary = {
        "symbol": symbol,
        "label": label,
        "interval": config.interval,
        "periods_per_year": periods_per_year,
        "data_start": _serialize_timestamp(prices.index[0]),
        "data_end": _serialize_timestamp(prices.index[-1]),
        **stats,
        "garch": garch_summary,
        "msm": msm_summary,
        "fractal": fractal_summary,
        "fractal_windowed": fractal_windowed,
        "gaf": gaf_summary,
        "outputs": {**outputs, **gaf_summary.get("sample_images", {})},
    }

    if gaf_warnings:
        warnings.extend(gaf_warnings)
    if fractal_windowed.get("warnings"):
        warnings.extend(str(msg) for msg in fractal_windowed["warnings"])

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

    comparison_rows: list[dict[str, object]] = []
    image_recommendations: list[dict[str, object]] = []

    for summary in results.values():
        if not isinstance(summary, dict):
            continue

        gaf_info = summary.get("gaf") or {}
        fractal_info = summary.get("fractal") or {}
        windowed = summary.get("fractal_windowed") or {}
        garch_info = summary.get("garch") or {}
        msm_info = summary.get("msm") or {}

        label = summary.get("label")
        row: dict[str, object] = {
            "label": label,
            "interval": summary.get("interval"),
            "observations": summary.get("observations"),
            "span": summary.get("span"),
            "gaf_windows": gaf_info.get("windows"),
            "gaf_channels": gaf_info.get("channels"),
            "gaf_image_size": gaf_info.get("image_size"),
            "gaf_window_span_pretty": gaf_info.get("window_span_pretty"),
            "gaf_label_distribution": gaf_info.get("label_distribution"),
            "gaf_label_distribution_share": gaf_info.get(
                "label_distribution_share"
            ),
            "fractal_metrics": {
                "DFA_H": fractal_info.get("DFA_H"),
                "RS_H": fractal_info.get("RS_H"),
                "MFDFA_width": fractal_info.get("MFDFA_width"),
                "WTMM_width": fractal_info.get("WTMM_width"),
            },
            "fractal_windowed_processed": windowed.get("processed_windows"),
            "fractal_windowed_expected": windowed.get("expected_gaf_windows"),
            "volatility": {
                "garch_last_cond_vol_annual": garch_info.get(
                    "last_cond_vol_annual"
                ),
                "msm_m_L": msm_info.get("m_L"),
                "msm_m_H": msm_info.get("m_H"),
            },
        }
        comparison_rows.append(row)

        image_eval = gaf_info.get("image_evaluation")
        if isinstance(image_eval, dict) and image_eval.get("status") != "ok":
            image_recommendations.append(
                {
                    "label": label,
                    "interval": summary.get("interval"),
                    **image_eval,
                }
            )

    comparison: dict[str, object] = {"scales": comparison_rows}
    if image_recommendations:
        comparison["image_recommendations"] = image_recommendations

    payload = {
        "results": results,
        "comparison": comparison,
    }

    master_path = output_dir / f"{_slugify(symbol)}_multi_scale_summary.json"
    with open(master_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return {
        "results": results,
        "comparison": comparison,

        "summary_path": str(master_path),
    }


__all__ = [
    "GAFScaleConfig",
    "ScaleConfig",
    "default_scale_configs",
    "run_scale",
    "run_multi_scale_analysis",
]

