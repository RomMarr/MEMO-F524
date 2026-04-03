import torch 
from M2.Utils.source import source
import torch.nn.functional as F


def loss_fn(model, x, y, x0, y0, t):
    pred = model(x, y, x0, y0, t)
    ones = torch.ones_like(pred)
    dp_dx, dp_dy, dp_dt = torch.autograd.grad(pred, [x, y, t], grad_outputs=ones, create_graph=True)
    dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]
    pde_res = dp_dtt - (5**2) * (dp_dxx + dp_dyy) - source(x, y, t, cx=x0, cy=y0)
    return (pde_res**2).mean()


def get_c_at_points(c, x, y, x_min=-5, x_max=5, y_min=-5, y_max=5):
    """
    Returns c evaluated at the sampled points (x, y).

    Parameters
    ----------
    c : float, int, torch.Tensor
        Either a scalar wave speed or a 2D tensor of shape (Nx, Ny)
        representing the wave speed field on the rectangular domain.
    x, y : torch.Tensor
        Sampled coordinates of shape (N, 1)
    x_min, x_max, y_min, y_max : float
        Spatial domain bounds for the matrix case
    """
    if isinstance(c, (int, float)):
        return torch.full_like(x, float(c))

    if torch.is_tensor(c):
        if c.ndim == 0:
            return torch.full_like(x, c.item())

        if c.ndim != 2:
            raise ValueError("If c is a tensor, it must be either a scalar or a 2D matrix.")

        c_field = c.to(device=x.device, dtype=x.dtype)

        # grid_sample expects input of shape (N_batch, C, H, W)
        c_field = c_field.unsqueeze(0).unsqueeze(0)

        # normalize coordinates to [-1, 1]
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

        # grid_sample expects grid shape (N_batch, H_out, W_out, 2)
        # last dimension order is (x, y)
        grid = torch.stack((x_norm.squeeze(-1), y_norm.squeeze(-1)), dim=-1)
        grid = grid.view(1, -1, 1, 2)

        c_interp = F.grid_sample(
            c_field,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return c_interp.view(-1, 1)
    raise ValueError("c must be either a scalar (int or float) or a 2D torch.Tensor.")

def loss_fn2(model, x, y, x0, y0, t, c, x_min, x_max, y_min, y_max):
    pred = model(x, y, x0, y0, t)
    ones = torch.ones_like(pred)

    dp_dx, dp_dy, dp_dt = torch.autograd.grad(
        pred, [x, y, t], grad_outputs=ones, create_graph=True
    )
    dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]

    c_val = get_c_at_points(c, x, y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    pde_res = dp_dtt - (c_val ** 2) * (dp_dxx + dp_dyy) - source(x, y, t, cx=x0, cy=y0)

    return (pde_res ** 2).mean()