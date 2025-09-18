import numpy as np
import pytest

from fractalfinance.gaf.gaf import gaf_cube, gaf_decode, gaf_encode, paa, save_gaf_png
from fractalfinance.gaf.saliency import top_saliency_pairs

try:  # optional torch dependency
    import torch
    from fractalfinance.gaf.dataset import GAFWindowDataset
except Exception:  # pragma: no cover - torch not installed
    torch = None
    GAFWindowDataset = None

def test_gaf_encode_decode_diagonal():
    x = np.random.randn(128)
    img, x_min, x_max, sign = gaf_encode(x, "gasf", resize=None, return_params=True)
    x_rec = gaf_decode(np.diag(img), x_min, x_max, sign)
    assert np.corrcoef(x, x_rec)[0, 1] > 0.99


def test_paa_downsampling():
    x = np.linspace(0, 1, 100)
    reduced = paa(x, 20)
    assert len(reduced) == 20
    assert np.isclose(reduced.mean(), x.mean())


def test_gaf_cube_shape_and_range():
    x = np.sin(np.linspace(0, 4 * np.pi, 300))
    cube = gaf_cube(x, resolutions=(64, 32), kinds=("gasf", "gadf"))
    assert cube.shape == (4, 64, 64)
    assert np.all(cube <= 1.0) and np.all(cube >= -1.0)


def test_save_gaf_png(tmp_path):
    x = np.random.randn(128)
    cube = gaf_cube(x, resolutions=(64,), kinds=("gasf",))
    path = save_gaf_png(cube, tmp_path / "gaf.png")
    assert path.exists()


def test_top_saliency_pairs_indices():
    sal = np.arange(16, dtype=float).reshape(4, 4)
    pairs = top_saliency_pairs(sal, indices=np.arange(4), top_k=2)
    assert pairs[0][0] == pairs[0][1]
    assert pairs[0][2] >= pairs[1][2]

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_dataset_len_stride_and_labels():
    series = np.arange(300)
    ds = GAFWindowDataset(series, win=100, stride=50)
    assert len(ds) == 5
    labels = [int(ds[i][1]) for i in range(len(ds))]
    assert len(labels) == len(ds)
    assert all(0 <= label <= 2 for label in labels)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_dataset_default_regime_labels():
    neg = np.full(150, -0.02)
    flat = np.zeros(150)
    pos = np.full(150, 0.03)
    series = np.concatenate([neg, flat, pos])
    ds = GAFWindowDataset(series, win=30, stride=30)
    labels = np.array([int(ds[i][1]) for i in range(len(ds))])
    assert labels.min() == 0
    assert labels.max() == 2
    # early windows dominated by negative returns, late ones positive
    assert np.all(labels[:2] == 0)
    assert np.all(labels[-2:] == 2)
