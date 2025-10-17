from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples import multi_scale_analysis as msa
from examples import multi_scale_gaf as msg


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
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    for key in ("price", "returns"):
        path = Path(result["outputs"][key])
        assert path.exists()


def test_gaf_summary_includes_box_counting_and_embeddings(tmp_path):
    idx = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    values = np.sin(np.linspace(0, 12 * np.pi, idx.size))
    returns = pd.Series(values, index=idx)

    config = msa.GAFScaleConfig(
        window=64,
        stride=16,
        resolutions=(32,),
        image_size=32,
        kinds=("gasf", "gadf"),
        label_mode="quantile",
        quantile_bins=3,
    )

    summary, warnings = msa.gaf_summary(
        returns,
        config=config,
        out_dir=tmp_path,
        slug="unit_test",
    )

    assert not warnings
    box_counting = summary.get("box_counting")
    assert box_counting
    assert box_counting["channel_metrics"]

    embedding = summary.get("embedding")
    assert embedding
    embedding_path = Path(embedding["embedding_path"])
    assert embedding_path.exists()
    embeddings = np.load(embedding_path)
    assert embeddings.shape[1] == embedding["dimension"]
    assert embedding.get("centroid_norm") is not None


def test_run_multi_scale_gaf_cross_similarity(monkeypatch, tmp_path):
    idx = pd.date_range("2024-01-01", periods=512, freq="D", tz="UTC")
    prices = pd.Series(
        100 + np.cumsum(np.sin(np.linspace(0, 24 * np.pi, idx.size))), index=idx
    )

    def fake_load_yahoo(symbol, start, end, interval, max_retries, retry_delay):
        return prices

    monkeypatch.setattr(msg, "load_yahoo", fake_load_yahoo)
    monkeypatch.setattr(msg, "DEFAULT_OUTPUT_DIR", tmp_path)

    scale_a = msa.ScaleConfig(
        interval="1d",
        label="Scale-A",
        gaf=msa.GAFScaleConfig(window=64, stride=16, resolutions=(32,), image_size=32),
    )
    scale_b = msa.ScaleConfig(
        interval="1d",
        label="Scale-B",
        gaf=msa.GAFScaleConfig(window=96, stride=24, resolutions=(32,), image_size=32),
    )

    result = msg.run_multi_scale_gaf(
        "FAKE",
        start="2023-01-01",
        end="2024-12-31",
        scales=[scale_a, scale_b],
        include_intraday=False,
        similarity_permutations=10,
        similarity_random_seed=0,
    )

    cross = result.get("cross_scale_similarity")
    assert cross
    matrix = cross.get("matrix")
    assert matrix
    assert len(cross.get("pairwise_tests", [])) >= 1
    for entry in cross["pairwise_tests"]:
        assert entry["permutations"] == 10
        assert entry["scale_a"] != entry["scale_b"]
