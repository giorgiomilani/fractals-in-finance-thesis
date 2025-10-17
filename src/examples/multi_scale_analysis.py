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
import torch

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


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return slug.lower() or "scale"


def median_spacing_seconds(index: pd.Index) -> float | None:
    idx = pd.DatetimeIndex(index).sort_values()
    if idx.size < 2:
        return None
    diffs = idx.to_series().diff().dropna().dt.total_seconds().to_numpy()
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def format_duration(seconds: float | None) -> str | None:
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


MAX_WINDOWS_FOR_GAF_METRICS = 256
PCA_EMBEDDING_DIMENSION = 16
BOX_COUNT_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


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
    label_mode: str = "quantile"
    quantile_bins: int = 3
    neutral_threshold: float = 0.0


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
                image_size=512,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
        ScaleConfig(
            interval="5m",
            label="5-minute",
            gaf=GAFScaleConfig(
                window=720,
                stride=48,
                resolutions=(512, 256),
                image_size=512,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
        ScaleConfig(
            interval="15m",
            label="15-minute",
            gaf=GAFScaleConfig(
                window=672,
                stride=32,
                resolutions=(384, 256),
                image_size=512,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
        ScaleConfig(
            interval="1h",
            label="1-hour",
            gaf=GAFScaleConfig(
                window=600,
                stride=24,
                resolutions=(384, 256),
                image_size=512,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
        ScaleConfig(
            interval="1d",
            label="Daily",
            gaf=GAFScaleConfig(
                window=512,
                stride=16,
                resolutions=(512, 256, 128),
                image_size=512,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
        ScaleConfig(
            interval="1wk",
            label="Weekly",
            gaf=GAFScaleConfig(
                window=260,
                stride=8,
                resolutions=(256, 128, 64),
                image_size=256,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
        ScaleConfig(
            interval="1mo",
            label="Monthly",
            gaf=GAFScaleConfig(
                window=180,
                stride=3,
                resolutions=(180, 120, 60),
                image_size=256,
                label_mode="sign",
                neutral_threshold=1e-06,
            ),
        ),
    ]


def _normalise_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _select_sample_indices(total: int, limit: int = MAX_WINDOWS_FOR_GAF_METRICS) -> list[int]:
    if total <= 0:
        return []
    limit = max(int(limit), 1)
    if total <= limit:
        return list(range(total))
    # ensure the first and last windows are included while spreading selections
    linspace = np.linspace(0, total - 1, num=limit, dtype=int)
    indices = sorted(set(int(idx) for idx in linspace))
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != total - 1:
        indices.append(total - 1)
    return indices[:limit]


def _prepare_cube_batch(dataset: GAFWindowDataset, indices: list[int]) -> np.ndarray:
    cubes: list[np.ndarray] = []
    for idx in indices:
        cube, _ = dataset[idx]
        cubes.append(cube.detach().cpu().numpy())
    if not cubes:
        return np.empty((0, 0, 0, 0), dtype=float)
    return np.stack(cubes)


def _compute_box_counts(binary: np.ndarray, box_size: int) -> int:
    if box_size <= 0:
        return 0
    height, width = binary.shape
    pad_h = (-height) % box_size
    pad_w = (-width) % box_size
    if pad_h or pad_w:
        binary = np.pad(binary, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    reshaped = binary.reshape(binary.shape[0] // box_size, box_size, binary.shape[1] // box_size, box_size)
    occupied = reshaped.any(axis=(1, 3))
    return int(np.count_nonzero(occupied))


def _box_sizes(image_size: int) -> list[int]:
    if image_size <= 0:
        return []
    sizes: list[int] = []
    max_power = int(math.log2(image_size)) if image_size > 0 else 0
    for power in range(1, max_power + 1):
        size = 2**power
        if size <= image_size:
            sizes.append(size)
    if image_size not in sizes:
        sizes.append(image_size)
    return sorted(set(sizes))


def _box_counting_dimension(image: np.ndarray, box_sizes: list[int]) -> float | None:
    if not box_sizes:
        return None
    data = np.asarray(image, dtype=float)
    data = np.nan_to_num(data, copy=False)
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if math.isclose(max_val, min_val):
        return None
    norm = (data - min_val) / (max_val - min_val + 1e-12)
    dims: list[float] = []
    for threshold in BOX_COUNT_THRESHOLDS:
        binary = norm >= threshold
        counts: list[int] = []
        sizes: list[int] = []
        for box_size in box_sizes:
            count = _compute_box_counts(binary, box_size)
            if count > 0:
                counts.append(count)
                sizes.append(box_size)
        if len(counts) < 2:
            continue
        inv_box = np.log(1.0 / np.array(sizes, dtype=float))
        log_counts = np.log(np.array(counts, dtype=float))
        slope, _ = np.polyfit(inv_box, log_counts, 1)
        dims.append(float(slope))
    if not dims:
        return None
    return float(np.mean(dims))


def _aggregate_box_counting(cubes: np.ndarray) -> dict[str, object]:
    if cubes.size == 0:
        return {}
    _, channels, height, width = cubes.shape
    box_sizes = _box_sizes(int(height))
    channel_metrics: list[dict[str, object]] = []
    for channel in range(channels):
        dimensions: list[float] = []
        for cube in cubes:
            dim = _box_counting_dimension(cube[channel], box_sizes)
            if dim is not None and math.isfinite(dim):
                dimensions.append(dim)
        if not dimensions:
            continue
        channel_metrics.append(
            {
                "channel": int(channel),
                "samples": len(dimensions),
                "mean": float(np.mean(dimensions)),
                "std": float(np.std(dimensions, ddof=0)),
            }
        )
    if not channel_metrics:
        return {}
    return {
        "thresholds": list(BOX_COUNT_THRESHOLDS),
        "box_sizes": box_sizes,
        "channel_metrics": channel_metrics,
        "samples": int(cubes.shape[0]),
    }


def _compute_embeddings(
    cubes: np.ndarray, target_dim: int = PCA_EMBEDDING_DIMENSION
) -> tuple[np.ndarray, dict[str, object]]:
    if cubes.size == 0:
        return np.empty((0, 0), dtype=float), {}
    samples = cubes.shape[0]
    flattened = cubes.reshape(samples, -1)
    if flattened.size == 0:
        return np.empty((0, 0), dtype=float), {}
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    dim = int(min(target_dim, samples, centered.shape[1]))
    if dim <= 0:
        return np.empty((0, 0), dtype=float), {}
    tensor = torch.from_numpy(centered).float()
    try:
        _, singular, v = torch.pca_lowrank(tensor, q=dim, center=False)
        embeddings = torch.matmul(tensor, v[:, :dim]).numpy()
        singular_values = singular[:dim].numpy()
    except RuntimeError:
        u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        embeddings = (u[:, :dim] * singular_values[:dim]).astype(float)
    centroid = embeddings.mean(axis=0)
    info: dict[str, object] = {
        "method": "pca_lowrank",
        "target_dimension": target_dim,
        "dimension": int(dim),
        "samples": int(samples),
        "centroid": centroid.tolist(),
    }
    if samples > 1:
        total_var = float(np.sum(centered**2) / (samples - 1))
        explained = (singular_values[:dim] ** 2) / max(samples - 1, 1)
        if total_var > 0:
            info["explained_variance_ratio"] = (explained / total_var).tolist()
        else:
            info["explained_variance_ratio"] = [0.0] * dim
    else:
        info["explained_variance_ratio"] = [0.0] * dim
    return embeddings.astype(float), info


def evaluate_gaf_image(
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


def gaf_summary(
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
        label_mode=config.label_mode,
        quantile_bins=config.quantile_bins,
        neutral_threshold=config.neutral_threshold,
    )
    windows = len(dataset)
    label_distribution: dict[str, int] = {}
    sample_images: dict[str, str] = {}
    warnings: list[str] = []
    box_counting_info: dict[str, object] | None = None
    embedding_info: dict[str, object] | None = None

    first_channels = 0

    actual_image_size = config.image_size or max(config.resolutions)

    metric_sample_indices: list[int] = []
    metric_sample_count = 0

    if windows == 0:
        warnings.append(
            "Insufficient samples to form a single GAF window at this scale."
        )
    else:
        labels_array = getattr(dataset, "labels", None)
        if labels_array is None or len(labels_array) != windows:
            labels_array = np.array([int(dataset[idx][1].item()) for idx in range(windows)])
        else:
            labels_array = np.asarray(labels_array, dtype=int)
        uniques, counts = np.unique(labels_array, return_counts=True)
        label_distribution = {
            str(int(u)): int(c) for u, c in zip(uniques, counts, strict=False)
        }

        metric_sample_indices = _select_sample_indices(windows)
        metric_sample_count = len(metric_sample_indices)
        sample_cubes = _prepare_cube_batch(dataset, metric_sample_indices)

        if sample_cubes.size == 0:
            cube, _ = dataset[0]
            first_cube = cube.detach().cpu().numpy()
        else:
            first_cube = sample_cubes[0]
            box_counting_info = _aggregate_box_counting(sample_cubes)
            embeddings, embedding_meta = _compute_embeddings(sample_cubes)
            if embeddings.size:
                embedding_path = out_dir / f"{slug}_embeddings.npy"
                np.save(embedding_path, embeddings)
                embedding_meta.update(
                    {
                        "embedding_path": str(embedding_path),
                        "sample_indices": [int(i) for i in metric_sample_indices],
                    }
                )
                centroid = np.asarray(embedding_meta.get("centroid", []), dtype=float)
                embedding_meta["centroid_norm"] = float(
                    np.linalg.norm(centroid)
                ) if centroid.size else 0.0
                embedding_info = embedding_meta
        actual_image_size = int(first_cube.shape[-1])
        first_channels = int(first_cube.shape[0])
        image_path = out_dir / f"{slug}_gaf_sample.png"
        if first_channels >= 3:
            select = [0, 1, 2]
        else:
            select = [0]
        save_gaf_png(first_cube, image_path, select_channels=select)
        sample_images["first_window_channels"] = select

        sample_images["first_window"] = str(image_path)

    median_spacing = median_spacing_seconds(returns.index)
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
        "window_span_pretty": format_duration(window_span),
        "sample_images": sample_images,
        "image_evaluation": evaluate_gaf_image(
            config=config,
            actual_size=actual_image_size,
            window=config.window,
        ),
        "label_mode": config.label_mode,
        "quantile_bins": int(config.quantile_bins),
        "neutral_threshold": float(config.neutral_threshold),
        "metric_samples": {
            "count": metric_sample_count,
            "indices": [int(i) for i in metric_sample_indices],
        },

    }
    if box_counting_info:
        summary["box_counting"] = box_counting_info
    if embedding_info:
        summary["embedding"] = embedding_info
    return summary, warnings


def serialize_timestamp(ts: pd.Timestamp) -> str:
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
    slug = slugify(f"{symbol}_{label}")
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
            serialised_record["start"] = serialize_timestamp(start_ts)
        if end_ts is not None:
            serialised_record["end"] = serialize_timestamp(end_ts)
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
    gaf_details, gaf_warnings = gaf_summary(
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
        metric_slug = slugify(metric_name)
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

    if gaf_details.get("windows") is not None:
        fractal_windowed["expected_gaf_windows"] = int(gaf_details["windows"])

    if (
        fractal_windowed.get("processed_windows")
        and gaf_details.get("windows") is not None
        and fractal_windowed["processed_windows"] != gaf_details["windows"]
    ):
        diff_warn = (
            "Windowed fractal metrics processed"
            f" {fractal_windowed['processed_windows']} samples,"
            f" but GAF generated {gaf_details['windows']} windows."
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
        "data_start": serialize_timestamp(prices.index[0]),
        "data_end": serialize_timestamp(prices.index[-1]),
        **stats,
        "garch": garch_summary,
        "msm": msm_summary,
        "fractal": fractal_summary,
        "fractal_windowed": fractal_windowed,
        "gaf": gaf_details,
        "outputs": {**outputs, **gaf_details.get("sample_images", {})},
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
        slug = slugify(f"{symbol}_{cfg.label}")
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
            "gaf_label_mode": gaf_info.get("label_mode"),
            "gaf_neutral_threshold": gaf_info.get("neutral_threshold"),
            "gaf_quantile_bins": gaf_info.get("quantile_bins"),
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

    master_path = output_dir / f"{slugify(symbol)}_multi_scale_summary.json"
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

