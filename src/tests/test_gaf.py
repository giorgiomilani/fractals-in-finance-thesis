import numpy as np

from fractalfinance.gaf import GADF, GASF, gaf_decode, gaf_encode


def test_gaf_dimensions():
    x = np.random.randn(256)
    img = gaf_encode(x, "gasf", resize=64)
    assert img.shape == (64, 64)


def test_gasf_symmetry():
    x = np.random.randn(100)
    img = GASF(x)
    assert np.allclose(img, img.T, atol=1e-6)


def test_gadf_definition_matches_arcsin():
    x = np.linspace(-2.0, 3.0, 10)
    gadf = GADF(x)

    x_min, x_max = x.min(), x.max()
    u = 2 * (x - x_min) / (x_max - x_min) - 1
    phi = np.arcsin(np.clip(u, -1.0, 1.0))
    expected = np.sin(phi[None, :] - phi[:, None])

    assert np.allclose(gadf, expected, atol=1e-6)


def test_gaf_decode_zero_one_scale():
    rng = np.random.default_rng(0)
    x = rng.uniform(10.0, 25.0, size=32)
    img, x_min, x_max, sign = gaf_encode(
        x,
        kind="gasf",
        resize=None,
        return_params=True,
        scale="zero-one",
    )

    reconstructed = gaf_decode(np.diag(img), x_min, x_max, sign, scale="zero-one")
    assert np.allclose(reconstructed, x, atol=1e-6)
