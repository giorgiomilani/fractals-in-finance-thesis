from pathlib import Path

import numpy as np
import pandas as pd

from fractalfinance.analysis import common


def test_compute_windowed_statistics_returns_samples(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=16, freq="D", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 110.0, idx.size), index=idx)
    returns = np.log(prices).diff().dropna()

    def fake_compute_fractal_metrics(price_window, window_returns, **kwargs):
        base = float(window_returns.mean())
        summary = {
            "RS_H": base,
            "DFA_H": base + 0.05,
            "MFDFA_width": base + 0.1,
        }
        return common.FractalResult(summary, {}, {}, {}, {}, {})

    monkeypatch.setattr(common, "compute_fractal_metrics", fake_compute_fractal_metrics)

    result = common.compute_windowed_fractal_statistics(
        prices,
        returns,
        window=6,
        stride=4,
        return_samples=True,
    )

    assert result["processed_windows"] > 0
    samples = result["samples"]
    assert set(samples) == {"RS_H", "DFA_H", "MFDFA_width"}
    for key, values in samples.items():
        assert len(values) == result["aggregates"][key]["count"]


def test_plot_windowed_metric_distribution(tmp_path):
    values = [0.2, 0.25, 0.3, 0.28, 0.27, 0.31]
    path = common.plot_windowed_metric_distribution(
        values,
        metric="MFDFA_width",
        out_dir=tmp_path,
        filename="mfdfa_width_distribution.png",
        title="MFDFA width distribution",
    )

    assert Path(path).exists()
