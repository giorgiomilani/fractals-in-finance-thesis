import abc
import numpy as np
import pandas as pd
from typing import Any

class BaseEstimator(abc.ABC):
    def __init__(self, series: pd.Series):
        self.series = series.astype(float).to_numpy()
        self.result_: dict[str, Any] | None = None

    @abc.abstractmethod
    def fit(self, **kwargs): ...
