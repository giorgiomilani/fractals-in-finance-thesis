import numpy as np, torch
from fractalfinance.gaf.dataset import GAFWindowDataset
from fractalfinance.gaf.gaf import gaf_encode

def test_gaf_encode_decode_diagonal():
    x = np.random.randn(128)
    img = gaf_encode(x, "gasf", resize=None)
    diag = np.diag(img)
    x_rec = np.sign(diag) * np.sqrt((diag+1)/2)
    assert np.corrcoef(x, x_rec)[0,1] > 0.9

def test_dataset_len_stride():
    series = np.arange(300)
    ds = GAFWindowDataset(series, win=100, stride=50)
    assert len(ds) == 4
