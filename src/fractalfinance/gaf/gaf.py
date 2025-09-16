from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def _to_unit(
    x: np.ndarray,
    detrend: bool = False,
    scale: str = "symmetric",
) -> tuple[np.ndarray, float, float]:
    """Normalise ``x`` according to ``scale`` with optional linear detrending."""

    if detrend:
        t = np.arange(len(x))
        a, b = np.polyfit(t, x, 1)
        x = x - (a * t + b)

    x_min, x_max = float(x.min()), float(x.max())
    span = x_max - x_min

    if np.isclose(span, 0.0):
        unit = np.zeros_like(x, dtype=float)
    elif scale == "symmetric":
        unit = 2 * (x - x_min) / span - 1
    elif scale in {"zero-one", "[0, 1]"}:
        unit = (x - x_min) / span
    else:
        raise ValueError("scale must be 'symmetric' or 'zero-one'")

    return unit.astype(float, copy=False), x_min, x_max


def _polar(x: np.ndarray, representation: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(x, -1.0, 1.0)
    if representation == "gasf":
        phi = np.arccos(x)
    elif representation == "gadf":
        phi = np.arcsin(x)
    else:
        raise ValueError("representation must be 'gasf' or 'gadf'")
    rho = np.linspace(0, 1, len(x), dtype=float)
    return rho, phi


def GASF(x: np.ndarray, detrend: bool = False, scale: str = "symmetric") -> np.ndarray:
    u, _, _ = _to_unit(x, detrend=detrend, scale=scale)
    _, phi = _polar(u, "gasf")
    return np.cos(phi[:, None] + phi[None, :])


def GADF(x: np.ndarray, detrend: bool = False, scale: str = "symmetric") -> np.ndarray:
    u, _, _ = _to_unit(x, detrend=detrend, scale=scale)
    _, phi = _polar(u, "gadf")
    return np.sin(phi[None, :] - phi[:, None])


def paa(x: np.ndarray, segments: int) -> np.ndarray:
    """Piecewise aggregate approximation down-sampling."""

    x = np.asarray(x, dtype=float)
    if segments <= 0:
        raise ValueError("segments must be positive")
    if segments > x.size:
        raise ValueError("segments cannot exceed series length")
    parts = np.array_split(x, segments)
    return np.array([p.mean() if p.size else 0.0 for p in parts], dtype=float)


def _resample_series(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if length == x.size:
        return x
    if length < x.size:
        return paa(x, length)
    # Upsample via interpolation when requesting more points
    grid = np.linspace(0, x.size - 1, length)
    return np.interp(grid, np.arange(x.size), x)


def gaf_encode(
    x: np.ndarray,
    kind: str = "gasf",
    resize: int | None = 128,
    return_params: bool = False,
    detrend: bool = False,
    scale: str = "symmetric",
) -> np.ndarray | tuple[np.ndarray, float, float, np.ndarray]:
    """Encode ``x`` into a Gramian Angular Field.

    When ``return_params`` is ``True`` the function also returns the
    ``(min, max)`` pair used for normalisation and the sign of the
    normalised series so that a perfect reconstruction is possible with
    :func:`gaf_decode`.
    """
    kind = kind.lower()
    u, x_min, x_max = _to_unit(x, detrend=detrend, scale=scale)
    _, phi = _polar(u, kind)
    if kind == "gasf":
        img = np.cos(phi[:, None] + phi[None, :])
    elif kind == "gadf":
        img = np.sin(phi[None, :] - phi[:, None])
    else:
        raise ValueError("kind must be 'gasf' or 'gadf'")
    if resize:
        img = np.array(Image.fromarray(img).resize((resize, resize), Image.BILINEAR))
    img = img.astype(np.float32)
    if return_params:
        sign = np.where(u >= 0, 1.0, -1.0)
        return img, x_min, x_max, sign
    return img


def gaf_cube(
    x: np.ndarray,
    *,
    resolutions: Sequence[int] = (256, 128, 64),
    kinds: Sequence[str] = ("gasf", "gadf"),
    detrend: bool = False,
    scale: str = "symmetric",
    resample: str = "paa",
    to_uint8: bool = False,
    image_size: int | None = None,
) -> np.ndarray:
    """Stack multiple GAF encodings across resolutions and kinds."""

    if resample not in {"paa", "interpolate"}:
        raise ValueError("resample must be 'paa' or 'interpolate'")

    x = np.asarray(x, dtype=float)
    channels: list[np.ndarray] = []
    target_size = int(image_size or max(resolutions))
    for res in resolutions:
        if res <= 0:
            raise ValueError("resolutions must contain positive integers")
        series = _resample_series(x, res) if resample == "paa" else np.interp(
            np.linspace(0, x.size - 1, res), np.arange(x.size), x
        )
        for kind in kinds:
            img = gaf_encode(
                series,
                kind=kind,
                resize=target_size,
                detrend=detrend,
                scale=scale,
            )
            channels.append(img)
    cube = np.stack(channels, axis=0).astype(np.float32, copy=False)
    if to_uint8:
        cube = ((cube + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return cube


def gaf_decode(
    diagonal_val: np.ndarray,
    x_min: float,
    x_max: float,
    sign: np.ndarray,
    scale: str = "symmetric",
) -> np.ndarray:
    """Reconstruct the original series from a GASF diagonal.

    Parameters
    ----------
    diagonal_val:
        The diagonal of the GASF matrix.
    x_min, x_max:
        Bounds used during normalisation.
    sign:
        Sign of the normalised series returned by :func:`gaf_encode`.
    """
    diagonal_val = np.clip(diagonal_val, -1.0, 1.0)
    x_unit = np.sqrt((diagonal_val + 1) / 2)
    if scale == "symmetric":
        x_unit = x_unit * sign
        return (x_unit + 1) / 2 * (x_max - x_min) + x_min
    if scale in {"zero-one", "[0, 1]"}:
        return x_unit * (x_max - x_min) + x_min
    raise ValueError("scale must be 'symmetric' or 'zero-one'")


def save_gaf_png(
    cube: np.ndarray,
    path: str | Path,
    *,
    channel_axis: int = 0,
    select_channels: Sequence[int] | None = None,
    mode: str | None = None,
) -> Path:
    """Persist a GAF cube as an 8-bit PNG image."""

    data = np.asarray(cube)
    if select_channels is not None:
        data = np.take(data, select_channels, axis=channel_axis)
    if data.ndim != 3:
        raise ValueError("cube must be a 3-D array (channels, H, W)")
    data = np.moveaxis(data, channel_axis, -1)
    if data.shape[-1] not in {1, 3}:
        raise ValueError("PNG export requires 1 or 3 channels; use select_channels")
    if data.dtype != np.uint8:
        data_min, data_max = float(data.min()), float(data.max())
        if np.isclose(data_max - data_min, 0.0):
            data = np.zeros_like(data, dtype=np.uint8)
        else:
            data = ((data - data_min) / (data_max - data_min) * 255.0).clip(0, 255).astype(np.uint8)
    img: Image.Image
    if data.shape[-1] == 1:
        img = Image.fromarray(data[..., 0], mode=mode or "L")
    else:
        img = Image.fromarray(data, mode=mode or "RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
