from __future__ import annotations

import types

import pandas as pd
import pytest
import requests

from fractalfinance.io import load_yahoo


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - kept for API compat
        return None

    def json(self) -> dict:
        return self._payload


def _payload(timestamps: list[int], closes: list[float]) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "meta": {"timezone": "America/New_York"},
                    "indicators": {"quote": [{"close": closes}]},
                }
            ]
        }
    }


def test_load_yahoo_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict] = {}

    def fake_get(url: str, params: dict, timeout: int, **kwargs) -> _DummyResponse:
        calls["params"] = params
        assert "^GSPC" in url
        return _DummyResponse(_payload([1704067200, 1704153600], [4700.0, 4710.0]))

    monkeypatch.setattr("requests.get", fake_get)

    series = load_yahoo("^GSPC", start="2024-01-01", end="2024-01-02")

    assert isinstance(series, pd.Series)
    assert series.index.tz is not None
    assert series.index.tz.zone == "America/New_York"
    assert series.index[0] == pd.Timestamp(2024, 1, 1, tz="America/New_York")
    assert series.iloc[-1] == pytest.approx(4710.0)
    assert calls["params"]["interval"] == "1d"
    # end date inclusive → adds almost one day (minus 1 second)
    assert calls["params"]["period2"] > calls["params"]["period1"]


def test_load_yahoo_raises_on_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, params: dict, timeout: int, **kwargs) -> _DummyResponse:
        return _DummyResponse(_payload([], []))

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(ValueError, match="No price data"):
        load_yahoo("^GSPC", start="2024-01-01", end="2024-01-02")


def test_load_yahoo_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class _RateLimitResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError(response=types.SimpleNamespace(status_code=429))

        def json(self) -> dict:
            return {}

    responses = [_RateLimitResponse(), _DummyResponse(_payload([1704067200], [4700.0]))]

    def fake_get(url: str, params: dict, timeout: int, **kwargs):
        calls.append(params)
        return responses[len(calls) - 1]

    monkeypatch.setattr("requests.get", fake_get)

    series = load_yahoo(
        "^GSPC",
        start="2024-01-01",
        end="2024-01-01",
        max_retries=2,
        retry_delay=0.0,
    )

    assert len(calls) == 2
    assert series.index.tz.zone == "America/New_York"
    assert series.iloc[0] == pytest.approx(4700.0)
