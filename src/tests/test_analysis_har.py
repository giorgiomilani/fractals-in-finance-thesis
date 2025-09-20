import numpy as np
import pandas as pd

from fractalfinance.analysis.common import (
    fit_har_realised_variance,
    plot_har_forecast,
)


def test_fit_har_and_plot(tmp_path):
    rng = np.random.default_rng(123)
    idx = pd.date_range("2023-01-01", periods=250, freq="D", tz="UTC")
    returns = pd.Series(rng.normal(scale=0.01, size=idx.size), index=idx)

    result = fit_har_realised_variance(returns, periods_per_year=252)
    assert result.realised_variance.index.equals(returns.index)
    assert len(result.forecast_daily_vol) == 5
    assert all(v > 0 for v in result.forecast_daily_vol)

    path = plot_har_forecast(
        result.realised_variance,
        result.forecast_daily_vol,
        out_dir=tmp_path,
        filename="har.png",
    )
    assert tmp_path.joinpath("har.png").exists()
    assert path.endswith("har.png")
