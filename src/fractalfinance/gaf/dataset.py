import numpy as np, torch
from torch.utils.data import Dataset
from .gaf import gaf_encode

class GAFWindowDataset(Dataset):
    """
    Roll a sliding window over a 1-D series and return GASF images.
    Optionally attach a label per window (e.g., volatility regime).
    """
    def __init__(self, series: np.ndarray, win: int = 256,
                 stride: int = 16, resize: int = 128,
                 labels: np.ndarray | None = None):
        self.series = series
        self.win = win
        self.stride = stride
        self.resize = resize
        self.labels = labels

    def __len__(self): return (len(self.series) - self.win) // self.stride

    def __getitem__(self, idx):
        i = idx * self.stride
        x = self.series[i : i + self.win]
        img = gaf_encode(x, "gasf", self.resize)   # (H,W)
        img = torch.tensor(img).unsqueeze(0)       # C×H×W
        if self.labels is None:
            y = torch.tensor(0)
        else:
            y = torch.tensor(self.labels[idx])
        return img.float(), y
