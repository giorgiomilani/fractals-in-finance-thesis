"""Utility helpers for generating fractal-finance plots.

The plotting functions default to saving output under an ``analysis_outputs``
folder at the project root. If the caller specifies a custom path the required
parent directories are created automatically, so plots from scripted analyses
end up in a consistent location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fractalfinance.gaf.gaf import gaf_encode
from fractalfinance.models.fbm import fbm
from fractalfinance.models.mmar import simulate

__all__ = [
    "plot_fbm",
    "plot_gaf",
    "plot_mmar",
    "DEFAULT_OUTPUT_DIR",
]

DEFAULT_OUTPUT_DIR = Path("analysis_outputs")

SeriesLike = Union[Sequence[float], Iterable[float], np.ndarray]


def _prepare_path(path: Union[str, Path]) -> Path:
    """Create parent directories for *path* and return it as a :class:`Path`."""

    save_path = Path(path).expanduser()
    if save_path.parent == Path('.'):
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = DEFAULT_OUTPUT_DIR / save_path.name
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def _coerce_series(series: SeriesLike | None, *, name: str) -> np.ndarray | None:
    if series is None:
        return None
    try:
        arr = np.asarray(series, dtype=float)
    except (TypeError, ValueError):
        arr = np.asarray(list(series), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def plot_fbm(
    path: Union[str, Path] = DEFAULT_OUTPUT_DIR / "fbm.png",
    H: float = 0.7,
    n: int = 1024,
    *,
    series: SeriesLike | None = None,
    title: str = "Fractional Brownian Motion",
) -> str:
    """Plot a fractional Brownian motion path or a user-supplied series."""

    data = _coerce_series(series, name="series")
    if data is None:
        data = fbm(H=H, n=n, seed=0)

    plt.figure(figsize=(8, 4))
    plt.plot(data, lw=1)
    plt.title(title)
    plt.tight_layout()
    save_path = _prepare_path(path)
    plt.savefig(save_path)
    plt.close()
    return str(save_path)


def plot_gaf(
    path: Union[str, Path] = DEFAULT_OUTPUT_DIR / "gaf.png",
    H: float = 0.7,
    n: int = 256,
    *,
    series: SeriesLike | None = None,
) -> str:
    """Visualise a series (or FBM sample) together with its GAF encodings."""

    data = _coerce_series(series, name="series")
    if data is None:
        data = fbm(H=H, n=n, seed=0)
    else:
        n = len(data)
    gasf = gaf_encode(data, kind="gasf", resize=n)
    gadf = gaf_encode(data, kind="gadf", resize=n)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(data, lw=1)
    axes[0].set_title("Series")
    axes[1].imshow(gasf, cmap="jet")
    axes[1].set_title("GASF")
    axes[1].axis("off")
    axes[2].imshow(gadf, cmap="jet")
    axes[2].set_title("GADF")
    axes[2].axis("off")
    fig.tight_layout()
    save_path = _prepare_path(path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)


def plot_mmar(
    path: Union[str, Path] = DEFAULT_OUTPUT_DIR / "mmar.png",
    H: float = 0.7,
    n: int = 1024,
    *,
    returns: SeriesLike | None = None,
    levels: SeriesLike | None = None,
    multipliers: SeriesLike | None = None,
) -> str:
    """Plot returns and levels for a cascade simulation or supplied dataset."""

    ret_arr = _coerce_series(returns, name="returns")
    lvl_arr = _coerce_series(levels, name="levels")
    mult_arr = _coerce_series(multipliers, name="multipliers")

    if ret_arr is None and lvl_arr is None and mult_arr is None:
        mult_arr, lvl_arr, ret_arr = simulate(n=n, H=H, seed=0)
    else:
        if ret_arr is None and lvl_arr is not None:
            if len(lvl_arr) < 2:
                raise ValueError("levels must contain at least two observations")
            ret_arr = np.diff(np.log(lvl_arr))
        if lvl_arr is None and ret_arr is not None:
            cumulative = np.cumsum(np.insert(ret_arr, 0, 0.0))
            lvl_arr = np.exp(cumulative)
    if ret_arr is None or lvl_arr is None:
        raise ValueError("plot_mmar requires returns or levels when multipliers are omitted")

    if mult_arr is not None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)
        axes[0].plot(mult_arr, lw=1)
        axes[0].set_title("Multipliers")
        ret_ax = axes[1]
        lvl_ax = axes[2]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=False)
        ret_ax, lvl_ax = axes

    ret_ax.plot(ret_arr, lw=1)
    ret_ax.set_title("Returns")
    lvl_ax.plot(lvl_arr, lw=1)
    lvl_ax.set_title("Price path" if mult_arr is None else "MMAR path")

    fig.tight_layout()
    save_path = _prepare_path(path)
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path)
