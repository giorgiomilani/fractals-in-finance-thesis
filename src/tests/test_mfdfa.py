import numpy as np
import pandas as pd

from fractalfinance.estimators import MFDFA


def make_fbm(H, N):
    """very small FBM generator for testing (uses Hosking recursion)."""
    g = np.zeros(N)
    g[0] = 1
    for k in range(1, N):
        g[k] = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + (k - 1) ** (2 * H))
    from scipy.linalg import cholesky, toeplitz

    L = cholesky(toeplitz(g), lower=True)
    z = L @ np.random.randn(N)
    return z.cumsum()


def test_mfdfa_width_on_fbm():
    np.random.seed(0)
    fbm_path = make_fbm(0.6, 2048)
    est = MFDFA(pd.Series(fbm_path)).fit()
    width = est.result_["alpha"].max() - est.result_["alpha"].min()
    # pure FBM is monofractal: width close to zero
    assert width < 0.2
