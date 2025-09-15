import pandas as pd
from fractalfinance.io import loaders


def test_load_binance_accepts_ms_int(monkeypatch):
    captured = {}

    async def fake_gather(url, param_iter, max_concurrent, request_delay):
        captured["params"] = list(param_iter)
        return [[[0, 0, 0, 0, "1.0", 0, 0, 0, 0, 0, 0, 0]]]

    monkeypatch.setattr(loaders, "_gather_klines", fake_gather)
    series = loaders.load_binance("BTCUSDT", "1m", 0, 60000)

    assert captured["params"][0]["startTime"] == 0
    assert captured["params"][0]["endTime"] == 60000
    assert series.index[0] == pd.Timestamp(0, unit="ms", tz="UTC")
    assert series.iloc[0] == 1.0
