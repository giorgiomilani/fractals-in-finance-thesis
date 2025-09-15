import numpy as np
import pandas as pd


def fill_gaps(series: pd.Series, freq: str) -> pd.Series:
    """Return *series* aligned to a regular frequency.

    The series is reindexed to ``freq`` using ``Series.asfreq``. Single missing
    observations are forward-filled while longer gaps remain as ``NaN`` so they
    can be handled explicitly downstream.

    Parameters
    ----------
    series : pd.Series
        Input time series indexed by timestamp.
    freq : str
        Target frequency string understood by pandas (e.g. ``"1min"``).

    Returns
    -------
    pd.Series
        Series resampled to ``freq`` with isolated gaps filled and larger ones
        left as ``NaN``.
    """
    return series.asfreq(freq).ffill(limit=1)


def clean_intraday(series: pd.Series, z_thresh: float = 6.0) -> pd.Series:
    """Return a copy of *series* with extreme intraday moves removed.

    Parameters
    ----------
    series : pd.Series
        Input price series indexed by timestamp.
    z_thresh : float, default 6.0
        Observations with a modified z-score greater than this threshold
        are treated as outliers.

    Returns
    -------
    pd.Series
        Cleaned series with outliers interpolated. The original input is
        left unmodified.
    """
    s = series.copy()
    diff = s.diff().dropna()
    mad = np.median(np.abs(diff - np.median(diff)))
    z = 0.6745 * diff / mad
    mask = (np.abs(z) < z_thresh).reindex(s.index, fill_value=True)
    s.loc[~mask] = np.nan
    return s.interpolate("time")


def wavelet_detrend(
    series: pd.Series,
    wavelet: str = "db4",
    level: int | None = None,
) -> pd.Series:
    """Remove low-frequency trends using discrete wavelet decomposition.

    The function performs a :func:`pywt.wavedec` on the input and sets the
    approximation coefficients (up to ``level``) to zero before reconstructing
    the series.  This acts as an orthogonal detrending step which preserves
    high‑frequency fluctuations important for multifractal analysis.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    wavelet : str, default "db4"
        Mother wavelet name understood by :mod:`pywt`.
    level : int, optional
        Decomposition level.  When ``None`` the maximum level permitted by the
        data length is used.

    Returns
    -------
    pd.Series
        Detrended series aligned with the original index.
    """
    import pywt

    coeffs = list(pywt.wavedec(series.values, wavelet, level=level))
    # Zero out approximation coefficients to remove trend
    coeffs[0] = np.zeros_like(coeffs[0])
    reconstructed = pywt.waverec(coeffs, wavelet)
    return pd.Series(reconstructed[: len(series)], index=series.index)
