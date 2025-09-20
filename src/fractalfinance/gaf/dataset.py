from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .gaf import gaf_cube


class GAFWindowDataset(Dataset):
    """Sliding-window dataset returning stacked GAF cubes."""

    def __init__(
        self,
        series: np.ndarray,
        *,
        win: int = 256,
        stride: int = 16,
        resize: int | Sequence[int] = 128,
        kinds: Sequence[str] = ("gasf",),
        detrend: bool = False,
        scale: str = "symmetric",
        resample: str = "paa",
        to_uint8: bool = False,
        image_size: int | None = None,
        labels: np.ndarray | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        label_mode: str = "quantile",
        quantile_bins: int = 3,
        neutral_threshold: float = 0.0,
    ) -> None:
        self.series = np.asarray(series, dtype=float)
        self.win = int(win)
        self.stride = int(stride)
        if isinstance(resize, int):
            self.resolutions = (int(resize),)
        else:
            self.resolutions = tuple(int(r) for r in resize)
        self.kinds = tuple(kinds)
        self.detrend = bool(detrend)
        self.scale = scale
        self.resample = resample
        self.to_uint8 = bool(to_uint8)
        self.image_size = int(image_size) if image_size is not None else None
        self.transform = transform
        self.label_mode = label_mode.lower()
        self.quantile_bins = int(quantile_bins)
        self.neutral_threshold = float(abs(neutral_threshold))
        if self.win <= 0 or self.stride <= 0:
            raise ValueError("win and stride must be positive integers")

        self._starts = self._compute_window_starts()
        self.labels = self._prepare_labels(labels)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._starts[idx]
        window = self.series[start : start + self.win]
        cube = gaf_cube(
            window,
            resolutions=self.resolutions,
            kinds=self.kinds,
            detrend=self.detrend,
            scale=self.scale,
            resample=self.resample,
            to_uint8=self.to_uint8,
            image_size=self.image_size,
        )
        tensor = torch.tensor(cube)
        if not self.to_uint8:
            tensor = tensor.float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        label = int(self.labels[idx]) if len(self.labels) else 0
        y = torch.tensor(label, dtype=torch.long)
        return tensor, y

    def _compute_window_starts(self) -> list[int]:
        if len(self.series) < self.win:
            return []
        last_start = len(self.series) - self.win
        starts = list(range(0, last_start + 1, self.stride))
        if starts and starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _prepare_labels(self, labels: np.ndarray | None) -> np.ndarray:
        if not self._starts:
            return np.array([], dtype=int)
        if labels is not None:
            arr = np.asarray(labels)
            if arr.shape[0] != len(self._starts):
                raise ValueError("labels length must match number of windows")
            return arr.astype(int)
        return self._default_labels()

    def _default_labels(self) -> np.ndarray:
        window_returns = np.array(
            [float(self.series[start : start + self.win].sum()) for start in self._starts]
        )
        if np.allclose(window_returns, window_returns[0]):
            return np.zeros_like(window_returns, dtype=int)

        if self.label_mode == "sign":
            return self._sign_labels(window_returns)

        if self.label_mode != "quantile":
            raise ValueError(
                "label_mode must be either 'quantile' or 'sign'"
            )

        bins = max(int(self.quantile_bins), 1)
        if bins <= 1:
            return np.zeros_like(window_returns, dtype=int)

        quantile_levels = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        quantiles = np.quantile(window_returns, quantile_levels)

        if np.allclose(quantiles, quantiles[0]):
            return self._sign_labels(window_returns)

        edges = np.concatenate(([-np.inf], quantiles, [np.inf]))
        labels = np.digitize(window_returns, edges) - 1
        return labels.astype(int)

    def _sign_labels(self, window_returns: np.ndarray) -> np.ndarray:
        threshold = self.neutral_threshold
        labels = np.full(window_returns.shape, 1, dtype=int)
        if threshold <= 0.0:
            labels[window_returns < 0.0] = 0
            labels[window_returns > 0.0] = 2
            return labels

        neg_mask = window_returns < -threshold
        pos_mask = window_returns > threshold
        labels[neg_mask] = 0
        labels[pos_mask] = 2
        return labels
