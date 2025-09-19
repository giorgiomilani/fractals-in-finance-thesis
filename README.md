# Fractals in Finance Thesis

This repository is the working notebook for the "Fractals in Finance" thesis.
It combines classical time-series tooling with modern multifractal diagnostics
so experimental ideas can be evaluated quickly.  The code is organised as a
Python package with a Typer command-line interface, ready-to-run examples, and a
growing library of estimators and stochastic simulators.

## Contents at a glance

- **Estimators** – DFA, MF-DFA, wavelet leaders, structure functions, WTMM and
  rescaled-range analysis, together with helpers that retain the regression
  metadata required for reproducible plots.
- **Stochastic models** – fractional Brownian motion generators, the
  Multifractal Multiplier (MMAR), and Markov-switching multifractal (MSM)
  calibrations.
- **Gramian Angular Fields** – utilities to turn one-dimensional price series
  into images for downstream ML experiments (`gaf_encode`, `gaf_decode`, and a
  configurable GAF dataset pipeline).
- **Risk metrics** – parametric and EVT-based Value at Risk/Expected Shortfall
  estimators with Hydra configuration files for rapid experimentation.
- **Example workflows** – S&P 500, multi-asset, and multi-scale analysis
  scripts that orchestrate downloads, model fitting, and visualisation.

All components ship with lightweight tests and example entry points so they can
be reused in isolation or stitched together into larger experiments.

---

## Installation

The project targets Python 3.11+.  The quickest way to get started is to create
a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### PowerShell one-liner sequence

For Windows users, the following commands mirror the setup used in the thesis
runs:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m fractalfinance.cli --help
```

The Typer help screen confirms the CLI is wired correctly.

## Test suite

Run the full unit-test collection with:

```bash
PYTHONPATH=src pytest -q
```

Optional dependencies such as `torch` and `Pillow` are declared in
`pyproject.toml`; installing the package with `pip install -e .` pulls the
required extras.

---

## Built-in analysis workflows

The `examples` subcommands encapsulate the end-to-end pipelines used in the
thesis.  Each workflow downloads data from Yahoo! Finance, computes summary
statistics, fits AR(1)-GARCH and MSM models, evaluates multifractal diagnostics
(MF-DFA, R/S, DFA, structure functions, WTMM), and writes consistent plots plus
JSON summaries under `analysis_outputs/<run>/<asset>/`.

### S&P 500 daily benchmark

```bash
python -m fractalfinance.cli examples sp500-daily \
    --start 2022-01-01 --end 2024-12-31 \
    --output-subdir sp500_daily --show-summary
```

The command stores price, returns, volatility-overlay, MF-DFA, R/S, DFA,
structure-function, and WTMM figures inside
`analysis_outputs/sp500_daily/sp500_daily/`, alongside
`sp500_summary.json`, which lists every generated artefact.

### Five-asset portfolio study

```bash
python -m fractalfinance.cli examples multi-asset \
    --base-output-subdir multi_asset_full_models --show-summary
```

The default bundle analyses the S&P 500, Bitcoin, EUR/USD, Apple, and TLT over
their full daily histories since 2020.  Each asset receives the complete suite
of plots (price, returns, GARCH, MF-DFA, R/S, DFA, structure, WTMM) plus its own
`*_summary.json`.  A consolidated `multi_asset_summary.json` is written to the
requested base directory so you can discover artefacts at a glance.  CLI options
allow overriding tickers, labels, and date ranges for bespoke studies.

### Multi-scale diagnostics

```bash
python -m fractalfinance.cli examples multi-scale ^GSPC \
    --start 1990-01-01 --include-intraday \
    --output-subdir sp500_multi_scale --show-summary
```

This pipeline sweeps the Yahoo! Finance intervals (1m → 1mo), calibrates GARCH
and MSM models at each scale, computes the fractal metrics, and produces price,
returns, GARCH, MF-DFA, R/S, DFA, structure-function, WTMM, and Gramian Angular
Field cubes.  Intraday downloads may emit throttle warnings; they are recorded
in the per-scale summaries together with any image-size recommendations.

### Plotting utilities

Individual simulators and transforms can be visualised directly:

```bash
python -m fractalfinance.cli plot fbm   # Fractional Brownian motion sample
python -m fractalfinance.cli plot gaf   # Original series with GASF/GADF views
python -m fractalfinance.cli plot mmar  # Multifractal Multiplier cascade
```

When an output path is omitted, images are saved under `analysis_outputs/` so
artefacts for every experiment remain in a single, version-controlled location.

---

## Repository layout

- `src/fractalfinance/analysis/` – shared helpers for loading data, computing
  summary statistics, fitting models, and plotting every diagnostic used in the
  thesis figures.
- `src/examples/` – reproducible scripts for the S&P 500 benchmark, the
  five-asset comparison, and the multi-scale sweep.  They showcase how the
  helper functions compose into full studies.
- `experiments/` – Hydra configurations plus experiment harnesses for risk and
  simulation studies.
- `analysis_outputs/` – sample artefacts generated by the commands above.  These
  folders double as regression fixtures to ensure figures remain discoverable.

Use the layouts as blueprints when adding new assets or extending the
experiments to alternative datasets.
