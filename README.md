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
=======
A research playground collecting fractal models, preprocessing utilities and
risk metrics for financial time series.


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
python -m fractalfinance.cli --help
python -m fractalfinance.cli run model=msm dataset=btc_minute
```

Hydra configuration files for datasets, models and risk metrics live in
`experiments/configs`.
