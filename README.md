# Fractals in Finance Thesis

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
