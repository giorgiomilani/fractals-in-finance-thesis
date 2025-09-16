from fractalfinance.geometry.wavelet import estimate_hurst
from fractalfinance.models.fbm import fbm


def test_wavelet_hurst_recovers_true_value():
    path = fbm(H=0.72, n=4096, seed=3)
    res = estimate_hurst(
        path,
        wavelet="db4",
        min_level=2,
        max_level=9,
        from_levels=True,
        kind="fgn",
    )
    assert abs(res.H - 0.72) < 0.05
    assert res.stderr < 0.2
