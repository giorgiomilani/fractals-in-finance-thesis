from fractalfinance.estimators import fbm_mle, fbm_wavelet_whittle
from fractalfinance.models import fbm


def test_fbm_mle_recovers_parameters():
    H_true = 0.72
    sigma_true = 1.4
    increments = sigma_true * fbm(H=H_true, n=512, kind="increment", seed=10)
    res = fbm_mle(increments, from_levels=False)
    assert abs(res["H"] - H_true) < 0.05
    assert abs(res["sigma"] - sigma_true) < 0.2


def test_wavelet_whittle_reasonable_accuracy():
    H_true = 0.65
    sigma_true = 0.9
    increments = sigma_true * fbm(H=H_true, n=512, kind="increment", seed=42)
    res = fbm_wavelet_whittle(increments, from_levels=False, wavelet="db3")
    assert abs(res["H"] - H_true) < 0.1
    assert abs(res["sigma"] - sigma_true) < 0.2
