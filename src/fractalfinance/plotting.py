import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fractalfinance.gaf.gaf import gaf_encode
from fractalfinance.models.fbm import fbm
from fractalfinance.models.mmar import simulate


__all__ = ["plot_fbm", "plot_gaf", "plot_mmar"]


def plot_fbm(path: str = "fbm.png", H: float = 0.7, n: int = 1024) -> str:
    """Generate and save a sample Fractional Brownian Motion path."""
    series = fbm(H=H, n=n, seed=0)
    plt.figure(figsize=(8, 4))
    plt.plot(series, lw=1)
    plt.title("Fractional Brownian Motion")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_gaf(path: str = "gaf.png", H: float = 0.7, n: int = 256) -> str:
    """Generate a series and visualise its GASF and GADF."""
    series = fbm(H=H, n=n, seed=0)
    gasf = gaf_encode(series, kind="gasf", resize=n)
    gadf = gaf_encode(series, kind="gadf", resize=n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(series, lw=1)
    axes[0].set_title("Series")
    axes[1].imshow(gasf, cmap="jet")
    axes[1].set_title("GASF")
    axes[1].axis("off")
    axes[2].imshow(gadf, cmap="jet")
    axes[2].set_title("GADF")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_mmar(path: str = "mmar.png", H: float = 0.7, n: int = 1024) -> str:
    """Simulate a Multifractal Multivariate AR path and its multipliers."""
    theta, X, r = simulate(n=n, H=H, seed=0)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(theta, lw=1)
    axes[0].set_title("Multipliers")
    axes[1].plot(r, lw=1)
    axes[1].set_title("Returns")
    axes[2].plot(X, lw=1)
    axes[2].set_title("MMAR path")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
