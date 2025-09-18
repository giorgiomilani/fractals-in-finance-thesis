# Fractals in Finance Thesis

This repository serves as an experimental sandbox for studying fractal
behaviour in financial markets.  It gathers classical and modern tools for
generating and analysing time series with long–memory or multifractal
characteristics.  The code base is intentionally lightweight so that new ideas
can be prototyped quickly.

### What you will find

- **Estimators** – algorithms such as DFA, Wavelet and MFDFA to measure Hurst
  exponents and multifractal spectra.
- **Stochastic models** – fractional Brownian motion, Multifractal
  Multiplier (MMAR) and Markov‑switching multifractal (MSM) generators.
- **Gramian Angular Fields** – utilities to map one‑dimensional price series
  to images via `gaf_encode`/`gaf_decode` for feeding CNN/VAE models.
- **Risk metrics** – parametric and EVT‑based Value at Risk and Expected
  Shortfall calculators.

Each component is documented with small tests and examples so it can be reused
in isolation or composed into larger experiments.


## Installation

The project targets Python 3.11+.  Install the package in editable mode
along with its dependencies:

```bash
pip install -e .
```

## Running the test-suite

Execute all unit tests with:

```bash
PYTHONPATH=src pytest -q
```

Some tests require optional packages such as `torch` and `Pillow` which are
listed as project dependencies.

## Command line interface

The Typer-based CLI exposes experiment runners and helpers:

```bash
# after `pip install -e .` or `export PYTHONPATH=src`
python -m fractalfinance.cli --help
python -m fractalfinance.cli run model=msm dataset=btc_minute
```

Hydra configuration files for datasets, models and risk metrics live in
`experiments/configs`.

### Built-in analyses

The CLI can execute the full S&P 500 daily workflow used in the thesis.  All
outputs land in ``analysis_outputs/<subdir>`` so plots remain discoverable:

```bash
python -m fractalfinance.cli examples sp500-daily --start 2022-01-01 \
    --end 2024-12-31 --output-subdir sp500_daily --show-summary
```

The command fetches data from Yahoo! Finance, fits AR(1)-GARCH and MSM models,
computes fractal diagnostics, and saves both the figures and a JSON summary.  A
list of generated image paths is echoed after the run so you can open them
immediately.

## Plotting

The command line interface can generate visualisations for the main stochastic
processes used in the thesis.  Run any of the commands below to create images in
the current directory:

```bash
python -m fractalfinance.cli plot fbm   # Fractional Brownian motion path
python -m fractalfinance.cli plot gaf   # series with its GASF and GADF
python -m fractalfinance.cli plot mmar  # cascade returns + price path
```

All plotting helpers now accept user-provided series, so you can reuse them to
visualise your own datasets.  When no output path is supplied they write to the
project's ``analysis_outputs`` directory by default, keeping artefacts in a
single location.
