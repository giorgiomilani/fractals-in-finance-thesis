from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


def load_csv(path: str | Path, tz: str = "UTC") -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp").squeeze(
        "columns"
    )
    if getattr(df.index, "tz", None) is None:
        df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)
    return df.sort_index()


def load_binance(symbol: str, interval: str, start: str, end: str) -> pd.Series:
    url = "https://api.binance.com/api/v3/klines"
    params = dict(
        symbol=symbol, interval=interval, startTime=start, endTime=end, limit=1000
    )
    frames = []
    while True:
        js = requests.get(url, params=params, timeout=10).json()
        if not js:
            break
        frames.append(pd.DataFrame(js)[[0, 4]].rename(columns={0: "ts", 4: "close"}))
        params["startTime"] = js[-1][0] + 1
    df = (
        pd.concat(frames)
        .set_index("ts")["close"]
        .astype(float)
        .tz_localize("UTC", unit="ms")
        .rename(symbol)
    )
    return df


def load_yahoo(
    symbol: str, start: str, end: str | None = None, tz: str = "UTC"
) -> pd.Series:
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    series = df["Close"].rename(symbol)
    series.index = pd.to_datetime(series.index).tz_localize(tz)
    return series.sort_index()
