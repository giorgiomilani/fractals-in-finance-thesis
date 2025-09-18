"""Thin wrappers around :mod:`fractalfinance.plotting` for example scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import typer

from fractalfinance.plotting import DEFAULT_OUTPUT_DIR, plot_fbm as _plot_fbm
from fractalfinance.plotting import plot_gaf as _plot_gaf
from fractalfinance.plotting import plot_mmar as _plot_mmar

app = typer.Typer(help="Generate example plots for fractal processes.")


def plot_fbm(
    path: Union[str, Path] = DEFAULT_OUTPUT_DIR / "fbm.png", H: float = 0.7, n: int = 1024
) -> str:
    """Create an FBM plot and save it to *path*."""

    return _plot_fbm(path=path, H=H, n=n)


def plot_gaf(
    path: Union[str, Path] = DEFAULT_OUTPUT_DIR / "gaf.png", H: float = 0.7, n: int = 256
) -> str:
    """Create GASF/GADF plots and save them to *path*."""

    return _plot_gaf(path=path, H=H, n=n)


def plot_mmar(
    path: Union[str, Path] = DEFAULT_OUTPUT_DIR / "mmar.png", H: float = 0.7, n: int = 1024
) -> str:
    """Create an MMAR plot and save it to *path*."""

    return _plot_mmar(path=path, H=H, n=n)


@app.command()
def fbm_cmd(path: Path = DEFAULT_OUTPUT_DIR / "fbm.png") -> None:
    """Create an FBM plot and save it to PATH."""

    typer.echo(plot_fbm(path))


@app.command()
def gaf_cmd(path: Path = DEFAULT_OUTPUT_DIR / "gaf.png") -> None:
    """Create GASF/GADF plots and save them to PATH."""

    typer.echo(plot_gaf(path))


@app.command()
def mmar_cmd(path: Path = DEFAULT_OUTPUT_DIR / "mmar.png") -> None:
    """Create an MMAR plot and save it to PATH."""

    typer.echo(plot_mmar(path))


if __name__ == "__main__":
    app()
