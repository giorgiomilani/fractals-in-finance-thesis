import numpy as np
import pytest
from fractalfinance.gaf.gaf import gaf_encode, gaf_decode

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

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_dataset_len_stride():
    series = np.arange(300)
    ds = GAFWindowDataset(series, win=100, stride=50)
    assert len(ds) == 4
