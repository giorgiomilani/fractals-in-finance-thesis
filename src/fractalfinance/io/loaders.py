from pathlib import Path
import pandas as pd
import requests

def load_csv(path: str | Path, tz: str = "UTC") -> pd.Series:
    df = (
        pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        .tz_localize(tz)
        .squeeze("columns")
    )
    df = df.sort_index()
    return df

def load_binance(symbol: str, interval: str, start: str, end: str) -> pd.Series:
    url = "https://api.binance.com/api/v3/klines"
    params = dict(symbol=symbol, interval=interval, startTime=start, endTime=end, limit=1000)
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
