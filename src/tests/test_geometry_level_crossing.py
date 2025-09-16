from fractalfinance.geometry.level_crossing import crossing_scaling
from fractalfinance.models.fbm import fbm


def test_level_crossing_hurst_estimate():
    path = fbm(H=0.66, n=4096, seed=4)
    res = crossing_scaling(path, from_levels=True, level=0.0)
    assert abs(res.H - 0.66) < 0.12
