import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fractalfinance.gaf import gaf_encode
import numpy as np

class GAFDataset(Dataset):
    def __init__(self, series: np.ndarray, labels: np.ndarray):
        self.x = np.stack([gaf_encode(s, "gasf") for s in series])
        self.y = labels

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).unsqueeze(0), self.y[i]

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, 1), nn.ReLU(), nn.Flatten(),
    nn.Linear(8 * 62 * 62, 3)
)
