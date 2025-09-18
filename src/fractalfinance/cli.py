"""
fractalfinance.cli
==================
Typer‑based command‑line interface.

Examples
--------
    python -m fractalfinance.cli --help
    python -m fractalfinance.cli run
    python -m fractalfinance.cli run model=msm dataset=btc_minute
    python -m fractalfinance.cli examples sp500-daily --start 2022-01-01
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

from . import plotting
from .plotting import DEFAULT_OUTPUT_DIR

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
def fbm_cmd(path: Path = DEFAULT_OUTPUT_DIR / "fbm.png") -> None:
    """Create an FBM plot and save it to PATH."""
    typer.echo(plotting.plot_fbm(path))


@plot_app.command("gaf")
def gaf_cmd(path: Path = DEFAULT_OUTPUT_DIR / "gaf.png") -> None:
    """Create GASF/GADF plots and save them to PATH."""
    typer.echo(plotting.plot_gaf(path))


@plot_app.command("mmar")
def mmar_cmd(path: Path = DEFAULT_OUTPUT_DIR / "mmar.png") -> None:
    """Create an MMAR plot and save it to PATH."""
    typer.echo(plotting.plot_mmar(path))


app.add_typer(plot_app, name="plot")


examples_app = typer.Typer(help="Run built-in analysis workflows.")


@examples_app.command("sp500-daily")
def sp500_daily_cmd(
    symbol: str = typer.Option("^GSPC", help="Yahoo Finance ticker symbol."),
    start: str = typer.Option("2022-01-01", help="Start date (YYYY-MM-DD)."),
    end: Optional[str] = typer.Option(None, help="End date (defaults to today)."),
    output_subdir: str = typer.Option("sp500_daily", help="Folder under analysis_outputs to store results."),
    show_summary: bool = typer.Option(False, help="Print the JSON summary after the run."),
) -> None:
    """Download S&P 500 data, run the full analysis, and save figures."""

    _ensure_experiments_on_path()
    from examples import sp500_daily_analysis

    summary = sp500_daily_analysis.run(
        symbol=symbol,
        start=start,
        end=end,
        output_subdir=output_subdir,
    )
    summary_path = DEFAULT_OUTPUT_DIR / output_subdir / "sp500_summary.json"
    typer.echo(f"Summary written to {summary_path}")
    outputs = summary.get("outputs", {})
    if outputs:
        typer.echo("Generated figures:")
        for name, path in outputs.items():
            typer.echo(f"  {name}: {path}")
    if show_summary:
        typer.echo(json.dumps(summary, indent=2))


app.add_typer(examples_app, name="examples")


# ──────────────────────────────────────────────────────────────────────────────
# module entry‑point
# ──────────────────────────────────────────────────────────────────────────────
def _entry_point() -> None:  # invoked by `python -m fractalfinance.cli`
    app()


if __name__ == "__main__":
    _entry_point()
