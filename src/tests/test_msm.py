import numpy as np

from fractalfinance.models import MSMParams, msm_fit, msm_simulate


def test_msm_fit_recovers_mH():
    np.random.seed(0)
    true_par = MSMParams(sigma2=1.0, m_L=0.7, m_H=1.4, gamma_1=0.05, b=2.0, K=5)
    _, r = msm_simulate(1500, true_par, seed=0)
    est_par = msm_fit(r, K=5, grid_m_H=(1.2, 1.6, 3), grid_gamma1=(0.03, 0.07, 3))
    assert abs(est_par.m_H - true_par.m_H) < 0.15
