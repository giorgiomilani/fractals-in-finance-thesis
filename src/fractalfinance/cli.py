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


@examples_app.command("multi-asset")
def multi_asset_cmd(
    base_output_subdir: str = typer.Option(
        "multi_asset", help="Root folder under analysis_outputs to store results."
    ),
    bitcoin_symbol: str = typer.Option(
        "BTC-USD", help="Yahoo Finance ticker for the Bitcoin pair."
    ),
    bitcoin_start: str = typer.Option(
        "2020-01-01", help="Start date for the Bitcoin run (YYYY-MM-DD)."
    ),
    bitcoin_end: Optional[str] = typer.Option(
        None, help="Optional end date for the Bitcoin run (YYYY-MM-DD)."
    ),
    forex_symbol: str = typer.Option(
        "EURUSD=X", help="Yahoo Finance ticker for the FX pair."
    ),
    forex_label: str = typer.Option(
        "EUR/USD", help="Display label for the FX pair."
    ),
    forex_start: str = typer.Option(
        "2020-01-01", help="Start date for the FX run (YYYY-MM-DD)."
    ),
    forex_end: Optional[str] = typer.Option(
        None, help="Optional end date for the FX run (YYYY-MM-DD)."
    ),
    apple_start: str = typer.Option(
        "2020-01-01", help="Start date for the Apple run (YYYY-MM-DD)."
    ),
    apple_end: Optional[str] = typer.Option(
        None, help="Optional end date for the Apple run (YYYY-MM-DD)."
    ),
    bond_symbol: str = typer.Option(
        "TLT", help="Yahoo Finance ticker used as the long-term bond proxy."
    ),
    bond_label: str = typer.Option(
        "iShares 20+ Year Treasury Bond ETF",
        help="Display label for the long-term bond instrument.",
    ),
    bond_start: str = typer.Option(
        "2020-01-01", help="Start date for the bond run (YYYY-MM-DD)."
    ),
    bond_end: Optional[str] = typer.Option(
        None, help="Optional end date for the bond run (YYYY-MM-DD)."
    ),
    show_summary: bool = typer.Option(
        False,
        help="Print the combined JSON summary after finishing the runs.",
    ),
) -> None:
    """Execute the four-asset bundle analysis and save figures."""

    _ensure_experiments_on_path()
    from examples import asset_analysis

    overrides = {
        "bitcoin": {
            "symbol": bitcoin_symbol,
            "start": bitcoin_start,
            "end": bitcoin_end,
        },
        "forex": {
            "symbol": forex_symbol,
            "label": forex_label,
            "start": forex_start,
            "end": forex_end,
        },
        "apple": {
            "start": apple_start,
            "end": apple_end,
        },
        "bond": {
            "symbol": bond_symbol,
            "label": bond_label,
            "start": bond_start,
            "end": bond_end,
        },
    }

    overrides = {
        key: {k: v for k, v in value.items() if v is not None}
        for key, value in overrides.items()
    }

    results, master_path = asset_analysis.run_default_assets(
        base_output_subdir=base_output_subdir,
        overrides=overrides,
    )

    if master_path:
        typer.echo(f"Summary written to {master_path}")

    for summary in results.values():
        raw_label = summary.get("label") or summary.get("key") or "asset"
        label = str(raw_label).strip()
        summary_path = summary.get("summary_path")
        typer.echo(f"{label} summary: {summary_path}")
        outputs = summary.get("outputs", {})
        if outputs:
            typer.echo(f"{label} figures:")
            for name, path in outputs.items():
                typer.echo(f"  {name}: {path}")

    if show_summary:
        typer.echo(json.dumps(results, indent=2))


@examples_app.command("multi-scale")
def multi_scale_cmd(
    symbol: str = typer.Argument("^GSPC", help="Yahoo Finance ticker symbol."),
    start: str = typer.Option("1990-01-01", help="Start date (YYYY-MM-DD)."),
    end: Optional[str] = typer.Option(None, help="End date (defaults to today)."),
    output_subdir: str = typer.Option(
        "multi_scale", help="Folder under analysis_outputs for the run."
    ),
    include_intraday: bool = typer.Option(
        True, help="Include minute/hourly intervals in the comparison."
    ),
    show_summary: bool = typer.Option(
        False, help="Print the combined JSON summary after finishing the run."
    ),
) -> None:
    """Run the multi-timescale analysis pipeline for ``symbol``."""

    _ensure_experiments_on_path()
    from examples import multi_scale_analysis

    scales = multi_scale_analysis.default_scale_configs()
    if not include_intraday:
        skip = {"1m", "5m", "15m", "1h"}
        scales = [cfg for cfg in scales if cfg.interval not in skip]

    result = multi_scale_analysis.run_multi_scale_analysis(
        symbol,
        start=start,
        end=end,
        scales=scales,
        output_subdir=output_subdir,
    )

    summary_path = result.get("summary_path")
    if summary_path:
        typer.echo(f"Summary written to {summary_path}")

    for slug, summary in result.get("results", {}).items():
        label = summary.get("label", slug)
        summary_path = summary.get("summary_path")
        typer.echo(f"{label} summary: {summary_path}")
        outputs = summary.get("outputs", {})
        if outputs:
            typer.echo(f"{label} figures:")
            for name, path in outputs.items():
                typer.echo(f"  {name}: {path}")
        if summary.get("warnings"):
            typer.echo("  warnings:")
            for warn in summary["warnings"]:
                typer.echo(f"    - {warn}")
        if summary.get("error"):
            typer.echo(f"  error: {summary['error']}")

    comparison = result.get("comparison", {}) if isinstance(result, dict) else {}
    recommendations = comparison.get("image_recommendations")
    if recommendations:
        typer.echo("Image size recommendations:")
        for rec in recommendations:
            label = rec.get("label", rec.get("interval", "scale"))
            status = rec.get("status")
            message = rec.get("message")
            configured = rec.get("configured_image_size")
            suggested = rec.get("recommended_image_size")
            typer.echo(
                "  - "
                f"{label}: status={status}, configured={configured}, "
                f"recommended={suggested}"
            )
            if message:
                typer.echo(f"    {message}")


    if show_summary:
        typer.echo(json.dumps(result, indent=2))


app.add_typer(examples_app, name="examples")


# ──────────────────────────────────────────────────────────────────────────────
# module entry‑point
# ──────────────────────────────────────────────────────────────────────────────
def _entry_point() -> None:  # invoked by `python -m fractalfinance.cli`
    app()


if __name__ == "__main__":
    _entry_point()
