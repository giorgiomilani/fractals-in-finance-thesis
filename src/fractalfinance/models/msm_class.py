from __future__ import annotations

from itertools import product
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp


class Params(NamedTuple):
    sigma2: float
    m_L: float
    m_H: float
    gamma_1: float
    b: float
    K: int


def _gamma_k(g1: float, b: float, k: int) -> float:
    return 1 - (1 - g1) ** (b ** (k - 1))


def _all_states(K: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=K)), dtype=int)


class MSM:
    """
    After .fit(), call .forecast(h) to get h-step variance forecast.
    """

    def __init__(self, params: Params):
        self.p = params
        self.states = _all_states(params.K)
        self.sigma_state = np.sqrt(
            params.sigma2
            * np.where(self.states == 0, params.m_L, params.m_H).prod(axis=1)
        )
        self.P = self._trans_mat()

    # ---------------- private helpers ---------------- #
    def _trans_mat(self) -> np.ndarray:
        P = np.ones((len(self.states), len(self.states)))
        for k in range(self.p.K):
            gk = _gamma_k(self.p.gamma_1, self.p.b, k + 1)
            same = self.states[:, k, None] == self.states[None, :, k]
            P *= same * (1 - gk) + (~same) * gk * 0.5
        return P

    # ---------------- fitting ------------------------ #
    @classmethod
    def fit(cls, r: ArrayLike, params: Params) -> "MSM":
        """
        Very small wrapper around forward filter – no optimisation here.
        You can grid-search externally using the old loglik() function.
        """
        self = cls(params)
        r = np.asarray(r)
        logP = np.log(self.P)
        const = -0.5 * np.log(2 * np.pi)

        log_alpha = np.full(len(self.states), -np.log(len(self.states)))
        for rt in r:
            log_emit = (
                const - np.log(self.sigma_state) - 0.5 * (rt / self.sigma_state) ** 2
            )
            log_alpha = log_emit + logsumexp(logP + log_alpha[:, None], axis=0)
            log_alpha -= logsumexp(log_alpha)  # normalise
        self.alpha_T = np.exp(log_alpha)  # prob. vector at time T
        return self

    # ---------------- forecast ----------------------- #
    def forecast(self, h: int = 1) -> float:
        """
        Return E[σ²_{T+h} | ℱ_T] using Chapman-Kolmogorov:
            α_{T+h} = α_T P^h
        Then               E σ² = α_{T+h} · σ²_state
        """
        Ph = np.linalg.matrix_power(self.P, h)
        alpha_h = self.alpha_T @ Ph
        return float((alpha_h * self.sigma_state**2).sum())
