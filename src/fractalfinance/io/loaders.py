from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Iterable

import aiohttp
import pandas as pd
import requests


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


def _to_ms(ts: str | int | float) -> int:
    """Convert various timestamp representations to milliseconds.


    Integers (and digit strings) are interpreted directly as milliseconds since
    the Unix epoch. Other inputs are parsed with ``pandas.to_datetime`` and
    converted to UTC before being returned in millisecond precision.
    """
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str) and ts.isdigit():
        return int(ts)
    return int(pd.to_datetime(ts, utc=True).timestamp() * 1000)


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
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
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

    df = pd.concat(frames).set_index("ts")["close"].astype(float)
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df.rename(symbol).sort_index()


def _to_epoch_seconds(ts: str | int | float) -> int:
    """Convert a timestamp understood by pandas to epoch seconds (UTC)."""

    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str) and ts.isdigit():
        return int(ts)
    return int(pd.Timestamp(ts, tz="UTC").timestamp())


def load_yahoo(
    symbol: str,
    *,
    start: str | int,
    end: str | int,
    interval: str = "1d",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> pd.Series:
    """Download prices from the Yahoo! Finance chart endpoint.

    Parameters
    ----------
    symbol : str
        Ticker recognised by Yahoo! Finance (e.g. ``"^GSPC"`` for the S&P 500).
    start, end : str or int
        Start and end boundaries understood by :func:`pandas.Timestamp`.  When
        supplying integers they are interpreted as Unix epoch seconds.  The
        ``end`` boundary is inclusive.
    interval : str, default ``"1d"``
        Sampling interval supported by Yahoo! Finance (``"1d"``, ``"1h"``,
        ``"1m"``, …).

    Returns
    -------
    pandas.Series
        Series of close prices indexed by timezone-aware timestamps ordered in
        ascending order.

    Notes
    -----
    The function relies on Yahoo!'s public chart API, avoiding optional
    third-party dependencies.  It raises :class:`ValueError` when the response
    does not contain price data.
    """

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    start_epoch = _to_epoch_seconds(start)
    end_epoch = _to_epoch_seconds(end) + 86399  # make inclusive

    params = {
        "period1": start_epoch,
        "period2": end_epoch,
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }

    max_retries = max(1, int(max_retries))
    payload: dict | None = None
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        if attempt > 0 and retry_delay > 0:
            time.sleep(retry_delay * attempt)
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.HTTPError as exc:
            last_exc = exc
            status = getattr(exc.response, "status_code", None)
            if status == 429 and attempt + 1 < max_retries:
                continue
            raise RuntimeError("Failed to fetch data from Yahoo! Finance") from exc
        except requests.RequestException as exc:  # pragma: no cover - network errors
            last_exc = exc
            if attempt + 1 < max_retries:
                continue
            raise RuntimeError("Failed to fetch data from Yahoo! Finance") from exc

    if payload is None:
        raise RuntimeError("Failed to fetch data from Yahoo! Finance") from last_exc

    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        closes = quotes["close"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Unexpected Yahoo! Finance response structure") from exc

    if not timestamps or not closes:
        raise ValueError("No price data returned by Yahoo! Finance")

    index = pd.to_datetime(timestamps, unit="s")

    tz_name = None
    meta = result.get("meta", {}) if isinstance(result, dict) else {}
    if isinstance(meta, dict):
        tz_name = meta.get("timezone")
        tz_aliases = {"EDT": "America/New_York", "EST": "America/New_York"}
        if tz_name in tz_aliases:
            tz_name = tz_aliases[tz_name]

    try:
        if tz_name:
            index = index.tz_localize(tz_name)
        else:
            index = index.tz_localize("UTC")
    except (TypeError, ValueError):  # pragma: no cover - already tz-aware or invalid
        if index.tz is None:
            index = index.tz_localize("UTC")

    series = pd.Series(closes, index=index, name=symbol, dtype="float64")
    series = series.dropna()

    if series.empty:
        raise ValueError("Yahoo! Finance returned only NaN closes")

    return series.sort_index()
