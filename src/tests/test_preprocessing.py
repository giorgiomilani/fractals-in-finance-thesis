import pandas as pd

from fractalfinance.preprocessing.core import clean_intraday, fill_gaps


def test_fill_gaps_handles_single_and_multi_gaps():
    idx = pd.date_range("2024-01-01", periods=6, freq="1min")
    series = pd.Series([1, 3, 6], index=[idx[0], idx[2], idx[5]])
    filled = fill_gaps(series, "1min")

    # single-step gap at idx[1] is forward-filled
    assert filled.loc[idx[1]] == 1
    # second step of the multi-gap remains missing
    assert pd.isna(filled.loc[idx[4]])


def test_clean_intraday_interpolates_outliers():
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    s = pd.Series([100, 101, 500, 103, 104], index=idx)
    cleaned = clean_intraday(s, z_thresh=1)

    # outlier is removed and interpolated linearly
    assert not cleaned.isna().any()
    assert cleaned.loc[idx[2]] == (s.loc[idx[1]] + s.loc[idx[3]]) / 2
