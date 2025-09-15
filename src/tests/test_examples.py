from pathlib import Path

from examples import plotting


def test_plotting_functions(tmp_path: Path):
    fbm_path = tmp_path / "fbm.png"
    gaf_path = tmp_path / "gaf.png"
    mmar_path = tmp_path / "mmar.png"
    plotting.plot_fbm(str(fbm_path))
    plotting.plot_gaf(str(gaf_path))
    plotting.plot_mmar(str(mmar_path))
    assert fbm_path.exists()
    assert gaf_path.exists()
    assert mmar_path.exists()
