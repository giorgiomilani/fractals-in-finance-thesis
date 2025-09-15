from pathlib import Path

import pandas as pd

from .loaders import load_binance, load_csv, load_yahoo


def save_parquet(series: pd.Series, path: str | Path):
    series.to_frame().to_parquet(path)


__all__ = ["load_csv", "load_binance", "load_yahoo", "save_parquet"]
