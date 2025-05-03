from importlib.metadata import version

try:
    __version__ = version("fractalfinance")
except Exception:
    __version__ = "0.0.0"

from .io import load_csv, load_binance, save_parquet   # noqa
from .preprocessing import fill_gaps, clean_intraday   # noqa
from .estimators import RS, DFA, MFDFA, WTMM           # noqa

__all__ = [
    "load_csv",
    "load_binance",
    "save_parquet",
    "fill_gaps",
    "clean_intraday",
    "RS",
    "DFA",
    "MFDFA",
    "WTMM",
]
