"""
Fractional Brownian Motion (FBM) Generator
=========================================
Primary method  : Davies–Harte (O(n log n))
Rare fallback   : Hosking recursion (O(n²))

Returns either:
• FBM **levels**  (length n)
• fractional Gaussian noise **increments** (length n)

References
----------
Davies & Harte (1987); Dieker (2004); Mandelbrot & Van Ness (1968)
"""

from __future__ import annotations
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

__all__ = ["fbm"]


# ------------------------------------------------------------------ #
def _davies_harte(H: float, n: int) -> np.ndarray:
    """Fractional Gaussian noise (FGN) length *n* via circulant embedding."""
    k = np.arange(n)
    gamma = 0.5 * (
        (k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H)
    )
    first_row = np.concatenate([gamma, [0.0], gamma[1:][::-1]])
    eigs = np.fft.fft(first_row).real
    eigs = np.clip(eigs, a_min=0.0, a_max=None)      # numerical safety

    # complex Gaussian vector same length as eigs (2n)
    W = np.random.normal(size=2 * n) + 1j * np.random.normal(size=2 * n)
    coeff = np.sqrt(eigs / (2 * n))
    fft_vec = coeff * W
    fgn = np.fft.ifft(fft_vec).real[:n]
    return fgn


# ------------------------------------------------------------------ #
def _hosking(H: float, n: int) -> np.ndarray:
    """Exact but O(n²) FGN via Hosking recursion."""
    gamma = lambda k: 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + (k - 1) ** (2 * H))
    cov = np.array([gamma(k) for k in range(n)])

    fgn = np.empty(n)
    phi = np.zeros(n)
    var = np.zeros(n)

    fgn[0] = np.random.normal(scale=np.sqrt(cov[0]))
    var[0] = cov[0]

    for k in range(1, n):
        phi_k = (cov[1 : k + 1] - phi[1:k] @ cov[1:k][::-1]) / var[k - 1]
        phi[k] = phi_k
        var[k] = var[k - 1] * (1 - phi_k**2)
        fgn[k] = phi_k * fgn[:k][::-1] @ phi[1 : k + 1] + np.random.normal(
            scale=np.sqrt(var[k])
        )

    return fgn


# ------------------------------------------------------------------ #
def fbm(
    H: float,
    n: int,
    length: float = 1.0,
    kind: Literal["level", "increment"] = "level",
    seed: int | None = None,
) -> ArrayLike:
    """
    Parameters
    ----------
    H : float
        Hurst exponent (0 < H < 1).
    n : int
        Number of points.
    length : float, default 1
        End‑time T (dt = T/n).
    kind : {"level", "increment"}
        Return FBM path or increments.
    seed : int, optional
        RNG seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    # FGN generation
    try:
        fgn = _davies_harte(H, n)
    except Exception:
        fgn = _hosking(H, n)  # extremely rare

    # scale to time length
    scale = (length / n) ** H
    fgn *= scale

    if kind == "increment":
        return fgn
    elif kind == "level":
        return np.cumsum(fgn)
    else:
        raise ValueError("kind must be 'level' or 'increment'")
