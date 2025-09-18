import numpy as np
from pathlib import Path

from fractalfinance import plotting


def test_plot_fbm_accepts_external_series(tmp_path):
    path = tmp_path / "fbm.png"
    custom = np.linspace(0.0, 1.0, num=32)
    result = plotting.plot_fbm(path, series=custom, title="Custom series")
    assert Path(result) == path
    assert path.exists()


def test_plot_gaf_accepts_external_series(tmp_path):
    path = tmp_path / "gaf.png"
    custom = np.sin(np.linspace(0.0, np.pi, num=64))
    result = plotting.plot_gaf(path, series=custom)
    assert Path(result) == path
    assert path.exists()


def test_plot_mmar_with_returns_and_levels(tmp_path):
    path = tmp_path / "mmar.png"
    levels = np.linspace(100.0, 110.0, num=16)
    returns = np.diff(np.log(levels))
    result = plotting.plot_mmar(path, returns=returns, levels=levels)
    assert Path(result) == path
    assert path.exists()


def test_plot_mmar_with_returns_only(tmp_path):
    path = tmp_path / "mmar_returns_only.png"
    returns = np.array([0.01, -0.005, 0.007, -0.002])
    result = plotting.plot_mmar(path, returns=returns)
    assert Path(result) == path
    assert path.exists()
