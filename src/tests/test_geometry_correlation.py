import numpy as np

from fractalfinance.geometry import correlation


def test_correlation_dimension_white_noise():
    rng = np.random.default_rng(5)
    series = rng.normal(size=1500)
    radii = np.logspace(-2.5, -0.2, 18)
    res = correlation.estimate(
        series,
        m=3,
        tau=2,
        radii=radii,
        theiler=2,
        n_surrogates=2,
        seed=10,
    )

    assert 2.0 < res.D2 < 3.5
    assert res.surrogates_D2 is not None
    assert res.surrogates_D2.shape == (2,)
