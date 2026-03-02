import torch

def apply_dirichlet(u: torch.Tensor) -> torch.Tensor:
    v = u.clone()
    v[..., 0, :] = 0
    v[..., -1, :] = 0
    v[..., :, 0] = 0
    v[..., :, -1] = 0
    return v