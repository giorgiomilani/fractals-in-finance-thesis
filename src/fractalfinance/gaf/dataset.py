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

        q1, q2 = np.quantile(window_returns, [1 / 3, 2 / 3])
        if np.isclose(q1, q2):
            labels = np.where(
                window_returns < 0,
                0,
                np.where(window_returns > 0, 2, 1),
            )
            return labels.astype(int)

        bins = [-np.inf, q1, q2, np.inf]
        labels = np.digitize(window_returns, bins) - 1
        return labels.astype(int)
