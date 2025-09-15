"""
fractalfinance.cli
==================
Typer‑based command‑line interface.

Examples
--------
    python -m fractalfinance.cli --help
    python -m fractalfinance.cli run
    python -m fractalfinance.cli run model=msm dataset=btc_minute
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List

import typer

from . import plotting

# ──────────────────────────────────────────────────────────────────────────────
# create Typer app; disable Rich markup so help text prints safely in the
# Windows CP‑1252 console
# ──────────────────────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False, rich_markup_mode="none")


def _ensure_experiments_on_path() -> None:
    """
    Add the project root to sys.path so `import experiments.run` succeeds
    no matter where the user launches the CLI from.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ──────────────────────────────────────────────────────────────────────────────
# commands
# ──────────────────────────────────────────────────────────────────────────────
@app.command(help="Execute experiments.run (Hydra entry-point).")
def run(
    overrides: List[str] = typer.Argument(
        None,
        help="Hydra override strings, e.g. model=msm dataset=btc_minute",
    )
) -> None:
    """Run an experiment defined in the experiments package."""
    _ensure_experiments_on_path()
    exp = importlib.import_module("experiments.run")
    if hasattr(exp, "main"):
        exp.main(overrides or None)  # pass list or None
    else:
        typer.echo("Error: experiments.run.main() not found", err=True)
        raise typer.Exit(code=1)


# ──────────────────────────────────────────────────────────────────────────────
# data commands
# ──────────────────────────────────────────────────────────────────────────────
data_app = typer.Typer(help="Download market data.")


@data_app.command("yahoo")
def yahoo_cmd(
    symbol: str,
    start: str,
    end: str,
    path: Path = Path("data.csv"),
) -> None:
    """Download prices from Yahoo Finance and save to PATH."""
    from .io import load_yahoo

    series = load_yahoo(symbol, start=start, end=end)
    series.to_frame(name="close").to_csv(path, index_label="timestamp")


app.add_typer(data_app, name="data")


# ──────────────────────────────────────────────────────────────────────────────
# plotting commands
# ──────────────────────────────────────────────────────────────────────────────
plot_app = typer.Typer(help="Generate plots for fractal processes.")


@plot_app.command("fbm")
def fbm_cmd(path: str = "fbm.png") -> None:
    """Create an FBM plot and save it to PATH."""
    typer.echo(plotting.plot_fbm(path))


@plot_app.command("gaf")
def gaf_cmd(path: str = "gaf.png") -> None:
    """Create GASF/GADF plots and save them to PATH."""
    typer.echo(plotting.plot_gaf(path))


@plot_app.command("mmar")
def mmar_cmd(path: str = "mmar.png") -> None:
    """Create an MMAR plot and save it to PATH."""
    typer.echo(plotting.plot_mmar(path))


app.add_typer(plot_app, name="plot")


# ──────────────────────────────────────────────────────────────────────────────
# module entry‑point
# ──────────────────────────────────────────────────────────────────────────────
def _entry_point() -> None:  # invoked by `python -m fractalfinance.cli`
    app()


if __name__ == "__main__":
    _entry_point()
