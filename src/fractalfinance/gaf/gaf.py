import numpy as np
from PIL import Image

def _to_unit(x: np.ndarray, detrend: bool = False) -> tuple[np.ndarray, float, float]:
    """Normalise ``x`` to ``[-1, 1]`` with optional linear detrending."""
    if detrend:
        t = np.arange(len(x))
        a, b = np.polyfit(t, x, 1)
        x = x - (a * t + b)
    x_min, x_max = float(x.min()), float(x.max())
    unit = 2 * (x - x_min) / (x_max - x_min) - 1
    return unit, x_min, x_max


def _polar(x: np.ndarray):
    phi = np.arccos(x)
    rho = np.linspace(0, 1, len(x))
    return rho, phi


def GASF(x: np.ndarray, detrend: bool = False) -> np.ndarray:
    u, _, _ = _to_unit(x, detrend=detrend)
    rho, phi = _polar(u)
    return np.cos(phi[:, None] + phi[None, :])


def GADF(x: np.ndarray, detrend: bool = False) -> np.ndarray:
    u, _, _ = _to_unit(x, detrend=detrend)
    rho, phi = _polar(u)
    return np.sin(phi[None, :] - phi[:, None])


def gaf_encode(
    x: np.ndarray,
    kind: str = "gasf",
    resize: int | None = 128,
    return_params: bool = False,
    detrend: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float, np.ndarray]:
    """Encode ``x`` into a Gramian Angular Field.

    When ``return_params`` is ``True`` the function also returns the
    ``(min, max)`` pair used for normalisation and the sign of the
    normalised series so that a perfect reconstruction is possible with
    :func:`gaf_decode`.
    """
    u, x_min, x_max = _to_unit(x, detrend=detrend)
    img = GASF(u, detrend=False) if kind.lower() == "gasf" else GADF(u, detrend=False)
    if resize:
        img = np.array(Image.fromarray(img).resize((resize, resize), Image.BILINEAR))
    img = img.astype(np.float32)
    if return_params:
        return img, x_min, x_max, np.sign(u)
    return img


def mtf_encode(x: np.ndarray, bins: int = 8) -> np.ndarray:
    """Encode series into a Markov Transition Field."""
    x_norm = (x - x.min()) / (x.max() - x.min())
    states = np.linspace(0, 1, bins + 1)
    digitized = np.digitize(x_norm, states) - 1
    trans = np.zeros((bins, bins))
    for i in range(len(digitized) - 1):
        trans[digitized[i], digitized[i + 1]] += 1
    prob = trans / trans.sum(axis=1, keepdims=True)
    mtf = prob[digitized][:, digitized]
    return mtf


def gaf_decode(
    diagonal_val: np.ndarray,
    x_min: float,
    x_max: float,
    sign: np.ndarray,
) -> np.ndarray:
    """Reconstruct the original series from a GASF diagonal.

    Parameters
    ----------
    diagonal_val:
        The diagonal of the GASF matrix.
    x_min, x_max:
        Bounds used during normalisation.
    sign:
        Sign of the normalised series returned by :func:`gaf_encode`.
    """
    x_unit = np.sqrt((diagonal_val + 1) / 2) * sign
    return (x_unit + 1) / 2 * (x_max - x_min) + x_min
