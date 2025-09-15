from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

import aiohttp
import pandas as pd


def load_csv(path: str | Path, tz: str = "UTC") -> pd.Series:
    """Read a CSV file into a timezone-aware :class:`pandas.Series`.

    Parameters
    ----------
    path : str or Path
        File containing at least ``timestamp`` and a single value column.
    tz : str, default "UTC"
        Time zone used to localise the timestamp index.

    Returns
    -------
    pd.Series
        Series of values indexed by timestamp.
    """
    df = (
        pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        .tz_localize(tz)
        .squeeze("columns")
    )
    return df.sort_index()


async def _fetch(session: aiohttp.ClientSession, url: str, params: dict) -> list:
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _gather_klines(
    url: str,
    param_iter: Iterable[dict],
    max_concurrent: int,
    request_delay: float,
) -> list[list]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(params: dict):
        async with sem:
            if request_delay:
                await asyncio.sleep(request_delay)
            return await _fetch(session, url, params)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        tasks = [asyncio.create_task(_bounded(p)) for p in param_iter]
        return await asyncio.gather(*tasks)


def load_binance(
    symbol: str,
    interval: str,
    start: str | int,
    end: str | int,
    *,
    max_concurrent: int = 5,
    max_per_second: float | None = None,
) -> pd.Series:
    """Download close prices from the Binance REST API.

    Parameters
    ----------
    symbol : str
        Trading pair symbol recognised by Binance, e.g. ``"BTCUSDT"``.
    interval : str
        Candlestick interval such as ``"1m"`` or ``"1h"``.
    start, end : str or int
        Start and end times in milliseconds since epoch or any format understood
        by :func:`pandas.to_datetime`.
    max_concurrent : int, default 5
        Maximum number of simultaneous requests when fetching data.
    max_per_second : float, optional
        Cap on request frequency. When provided, requests are spaced so that at
        most this many are fired per second.

    Returns
    -------
    pd.Series
        Series of close prices indexed by UTC timestamps.

    Raises
    ------
    RuntimeError
        If the Binance API request fails.
    ValueError
        If no data is returned for the requested range.
    """
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    interval_ms = int(pd.to_timedelta(interval).total_seconds() * 1000)
    limit = 1000
    step = interval_ms * limit
    starts = list(range(start_ms, end_ms, step))
    params_iter = [
        dict(
            symbol=symbol,
            interval=interval,
            startTime=s,
            endTime=min(s + step - 1, end_ms),
            limit=limit,
        )
        for s in starts
    ]

    delay = 1.0 / max_per_second if max_per_second else 0.0
    try:
        batches = asyncio.run(
            _gather_klines(url, params_iter, max_concurrent, delay)
        )
    except aiohttp.ClientError as exc:
        raise RuntimeError("Binance API request failed") from exc

    frames = [
        pd.DataFrame(js)[[0, 4]].rename(columns={0: "ts", 4: "close"})
        for js in batches
        if js
    ]
    if not frames:
        raise ValueError("No data returned from Binance")

    df = (
        pd.concat(frames)
        .set_index("ts")["close"]
        .astype(float)
        .tz_localize("UTC", unit="ms")
        .rename(symbol)
        .sort_index()
    )
    return df
