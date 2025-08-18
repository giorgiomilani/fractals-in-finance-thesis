import numpy as np

from fractalfinance.gaf import GASF, gaf_encode


def test_gaf_dimensions():
    x = np.random.randn(256)
    img = gaf_encode(x, "gasf", resize=64)
    assert img.shape == (64, 64)


def test_gasf_symmetry():
    x = np.random.randn(100)
    img = GASF(x)
    assert np.allclose(img, img.T, atol=1e-6)
