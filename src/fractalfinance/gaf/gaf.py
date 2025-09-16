import numpy as np
from PIL import Image


def _to_unit(
    x: np.ndarray,
    detrend: bool = False,
    scale: str = "symmetric",
) -> tuple[np.ndarray, float, float]:
    """Normalise ``x`` according to ``scale`` with optional linear detrending."""

    if detrend:
        t = np.arange(len(x))
        a, b = np.polyfit(t, x, 1)
        x = x - (a * t + b)

    x_min, x_max = float(x.min()), float(x.max())
    span = x_max - x_min

    if np.isclose(span, 0.0):
        unit = np.zeros_like(x, dtype=float)
    elif scale == "symmetric":
        unit = 2 * (x - x_min) / span - 1
    elif scale in {"zero-one", "[0, 1]"}:
        unit = (x - x_min) / span
    else:
        raise ValueError("scale must be 'symmetric' or 'zero-one'")

    return unit.astype(float, copy=False), x_min, x_max


def _polar(x: np.ndarray, representation: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(x, -1.0, 1.0)
    if representation == "gasf":
        phi = np.arccos(x)
    elif representation == "gadf":
        phi = np.arcsin(x)
    else:
        raise ValueError("representation must be 'gasf' or 'gadf'")
    rho = np.linspace(0, 1, len(x), dtype=float)
    return rho, phi


def GASF(x: np.ndarray, detrend: bool = False, scale: str = "symmetric") -> np.ndarray:
    u, _, _ = _to_unit(x, detrend=detrend, scale=scale)
    _, phi = _polar(u, "gasf")
    return np.cos(phi[:, None] + phi[None, :])


def GADF(x: np.ndarray, detrend: bool = False, scale: str = "symmetric") -> np.ndarray:
    u, _, _ = _to_unit(x, detrend=detrend, scale=scale)
    _, phi = _polar(u, "gadf")
    return np.sin(phi[None, :] - phi[:, None])


def gaf_encode(
    x: np.ndarray,
    kind: str = "gasf",
    resize: int | None = 128,
    return_params: bool = False,
    detrend: bool = False,
    scale: str = "symmetric",
) -> np.ndarray | tuple[np.ndarray, float, float, np.ndarray]:
    """Encode ``x`` into a Gramian Angular Field.

    When ``return_params`` is ``True`` the function also returns the
    ``(min, max)`` pair used for normalisation and the sign of the
    normalised series so that a perfect reconstruction is possible with
    :func:`gaf_decode`.
    """
    kind = kind.lower()
    u, x_min, x_max = _to_unit(x, detrend=detrend, scale=scale)
    _, phi = _polar(u, kind)
    if kind == "gasf":
        img = np.cos(phi[:, None] + phi[None, :])
    elif kind == "gadf":
        img = np.sin(phi[None, :] - phi[:, None])
    else:
        raise ValueError("kind must be 'gasf' or 'gadf'")
    if resize:
        img = np.array(Image.fromarray(img).resize((resize, resize), Image.BILINEAR))
    img = img.astype(np.float32)
    if return_params:
        sign = np.where(u >= 0, 1.0, -1.0)
        return img, x_min, x_max, sign
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
    scale: str = "symmetric",
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
    diagonal_val = np.clip(diagonal_val, -1.0, 1.0)
    x_unit = np.sqrt((diagonal_val + 1) / 2)
    if scale == "symmetric":
        x_unit = x_unit * sign
        return (x_unit + 1) / 2 * (x_max - x_min) + x_min
    if scale in {"zero-one", "[0, 1]"}:
        return x_unit * (x_max - x_min) + x_min
    raise ValueError("scale must be 'symmetric' or 'zero-one'")
