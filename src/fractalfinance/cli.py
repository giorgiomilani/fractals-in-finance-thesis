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
        exp.main(overrides or None)            # pass list or None
    else:
        typer.echo("Error: experiments.run.main() not found", err=True)
        raise typer.Exit(code=1)


# ──────────────────────────────────────────────────────────────────────────────
# module entry‑point
# ──────────────────────────────────────────────────────────────────────────────
def _entry_point() -> None:  # invoked by `python -m fractalfinance.cli`
    app()


if __name__ == "__main__":
    _entry_point()
