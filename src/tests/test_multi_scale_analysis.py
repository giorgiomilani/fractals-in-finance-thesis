from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples import multi_scale_analysis as msa


@pytest.mark.parametrize("count", [6])
def test_run_scale_handles_short_series(monkeypatch, tmp_path, count):
    idx = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    prices = pd.Series(np.linspace(100, 101, count), index=idx)

    def fake_load_yahoo(symbol, start, end, interval, max_retries, retry_delay):
        return prices

    monkeypatch.setattr(msa, "load_yahoo", fake_load_yahoo)

    config = msa.ScaleConfig(
        interval="1d",
        label="Daily",
        gaf=msa.GAFScaleConfig(window=8, stride=4, resolutions=(16,), image_size=16),
    )

    result = msa.run_scale(
        "FAKE",
        start="2024-01-01",
        end="2024-02-01",
        config=config,
        output_dir=tmp_path,
    )

    assert result["gaf"]["windows"] == 0
    assert result["fractal_windowed"]["total_windows"] == 0
    warnings = result.get("warnings", [])
    assert any("Only" in msg and "recommended" in msg for msg in warnings)
    assert any("Insufficient samples to form a single GAF window" in msg for msg in warnings)
    assert "garch" not in result["outputs"]
    assert isinstance(result.get("garch"), dict)
    assert "error" in result["garch"]
    assert "har" not in result["outputs"]
    assert isinstance(result.get("har"), dict)
    assert "error" in result["har"]
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    for key in ("price", "returns"):
        path = Path(result["outputs"][key])
        assert path.exists()
