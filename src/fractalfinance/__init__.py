from importlib.metadata import version

try:
    __version__ = version("fractalfinance")
except Exception:
    __version__ = "0.0.0"

from .estimators import DFA, MFDFA, RS, WTMM  # noqa
try:  # optional data-loading helpers (require third-party deps)
    from .io import load_binance, load_csv, save_parquet  # noqa
except Exception:  # pragma: no cover - dependency missing
    load_binance = load_csv = save_parquet = None
from .preprocessing import clean_intraday, fill_gaps  # noqa

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
