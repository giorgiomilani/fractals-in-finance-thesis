import torch
def grad_cam(model, x: torch.Tensor, target_class: int):
    x = x.requires_grad_()
    scores = model(x)
    score = scores[:, target_class].sum()
    score.backward()
    return x.grad.abs().mean(0)       # simple gradient map
