import numpy as np
import pandas as pd

from fractalfinance.estimators import WTMM


def fbm(H: float, N: int) -> np.ndarray:
    """
    Quick-and-dirty FBM generator (same Hosking recursion as earlier test).
    Adequate for unit testing only.
    """
    g = np.zeros(N)
    g[0] = 1
    for k in range(1, N):
        g[k] = 0.5 * ((k + 1) ** (2 * H) - 2 * (k) ** (2 * H) + (k - 1) ** (2 * H))
    from scipy.linalg import cholesky, toeplitz

    L = cholesky(toeplitz(g), lower=True)
    return (L @ np.random.randn(N)).cumsum()


def test_wtmm_width_on_fbm():
    np.random.seed(1)
    path = fbm(0.55, 2048)
    est = WTMM(pd.Series(path), from_levels=True).fit()
    width = est.result_["alpha"].max() - est.result_["alpha"].min()
    assert width < 0.25  # monofractal ≈ narrow spectrum
