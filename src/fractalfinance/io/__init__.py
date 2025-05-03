from .loaders import load_csv, load_binance
import pandas as pd
from pathlib import Path

def save_parquet(series: pd.Series, path: str | Path):
    series.to_frame().to_parquet(path)

__all__ = ["load_csv", "load_binance", "save_parquet"]
