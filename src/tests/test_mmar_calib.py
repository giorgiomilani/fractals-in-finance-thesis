import numpy as np
import pandas as pd

from fractalfinance.models.mmar import calibrate, simulate


def test_calibrator_returns_close_mH():
    np.random.seed(0)
    true_casc = dict(m_L=0.5, m_H=2.0, depth=8, seed=0)
    _, X, _ = simulate(2048, 0.5, **true_casc)
    est = calibrate(X, 0.5, search_mH=(1.7, 2.3, 4))
    assert abs(est.m_H - 2.0) < 0.25
