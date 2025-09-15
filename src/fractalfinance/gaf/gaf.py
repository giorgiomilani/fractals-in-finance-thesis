import numpy as np
from PIL import Image


def _to_unit(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    return 2 * (x - x_min) / (x_max - x_min) - 1


def _polar(x: np.ndarray):
    phi = np.arccos(x)
    rho = np.linspace(0, 1, len(x))
    return rho, phi


def GASF(x: np.ndarray) -> np.ndarray:
    rho, phi = _polar(_to_unit(x))
    return np.cos(phi[:, None] + phi[None, :])


def GADF(x: np.ndarray) -> np.ndarray:
    rho, phi = _polar(_to_unit(x))
    return np.sin(phi[None, :] - phi[:, None])


def gaf_encode(
    x: np.ndarray, kind: str = "gasf", resize: int | None = 128
) -> np.ndarray:
    img = GASF(x) if kind.lower() == "gasf" else GADF(x)
    if resize:
        img = np.array(Image.fromarray(img).resize((resize, resize), Image.BILINEAR))
    return img.astype(np.float32)


def gaf_decode(diagonal_val: np.ndarray) -> np.ndarray:
    """
    Recover original (normalised) series from GASF diagonal.
    """
    x2 = (diagonal_val + 1) / 2
    return np.sqrt(x2) * np.sign(diagonal_val)
