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
        self.labels = labels
        self.transform = transform
        if self.win <= 0 or self.stride <= 0:
            raise ValueError("win and stride must be positive integers")

    def __len__(self) -> int:
        total = len(self.series) - self.win
        if total < 0:
            return 0
        return total // self.stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.stride
        window = self.series[i : i + self.win]
        cube = gaf_cube(
            window,
            resolutions=self.resolutions,
            kinds=self.kinds,
            detrend=self.detrend,
            scale=self.scale,
            resample=self.resample,
            to_uint8=self.to_uint8,
        )
        tensor = torch.tensor(cube)
        if not self.to_uint8:
            tensor = tensor.float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if self.labels is None:
            y = torch.tensor(0)
        else:
            y = torch.tensor(self.labels[idx])
        return tensor, y
