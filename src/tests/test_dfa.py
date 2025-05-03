import numpy as np
import pandas as pd

from fractalfinance.estimators import DFA


def fbm(h, n):
    """
    Tiny Davies–Harte FBM generator for test only (O(n log n)).
    """
    import scipy.linalg as la

    g = np.fft.fft(
        np.concatenate(
            (
                [1.0],
                0.5
                * (
                    np.arange(1, n) ** (2 * h)
                    - 2 * np.arange(1, n)
                    + np.arange(1, n) ** (2 * h)
                ),
                [0.0],
                0.5
                * (
                    np.arange(n - 1, 0, -1) ** (2 * h)
                    - 2 * np.arange(n - 1, 0, -1)
                    + np.arange(n - 1, 0, -1) ** (2 * h)
                ),
            )
        )
    ).real
        # --- Davies–Harte drawing --------------------------------------------
    # keep only the first n eigen‑values → length match
    lam = np.sqrt(np.clip(g[:n], a_min=0.0, a_max=None))

    w = np.random.normal(size=2 * n)
    w_complex = w[::2] + 1j * w[1::2]          # length n

    z = np.fft.ifft(lam * w_complex).real[:n]
    return z.cumsum()                          # FBM path (level series)


def test_dfa_bias():
    np.random.seed(42)
    n = 4096
    H_true = 0.75
    path = fbm(H_true, n)
    est = DFA(pd.Series(path)).fit()
    assert abs(est.result_["H"] - H_true) < 0.04
