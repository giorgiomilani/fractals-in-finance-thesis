from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def grad_cam(model, x: torch.Tensor, target_class: int) -> torch.Tensor:
    """Basic gradient-based saliency map for a trained model."""

    x = x.requires_grad_()
    scores = model(x)
    score = scores[:, target_class].sum()
    score.backward()
    return x.grad.abs().mean(0)


def top_saliency_pairs(
    saliency_map: np.ndarray | torch.Tensor,
    *,
    indices: Sequence[int] | None = None,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Return the most salient (i, j) coordinates from a saliency map."""

    if isinstance(saliency_map, torch.Tensor):
        sal = saliency_map.detach().cpu().numpy()
    else:
        sal = np.asarray(saliency_map, dtype=float)
    flat = np.argsort(sal.ravel())[::-1][:top_k]
    coords = np.column_stack(np.unravel_index(flat, sal.shape))
    values = sal.ravel()[flat]
    if indices is not None:
        idx = np.asarray(indices)
        coords = np.column_stack((idx[coords[:, 0]], idx[coords[:, 1]]))
    return [(int(i), int(j), float(v)) for (i, j), v in zip(coords, values)]
