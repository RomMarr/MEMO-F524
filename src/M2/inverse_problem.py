import math
import torch
import torch.nn as nn
from tqdm import tqdm



def inverse_function(
    forward,               # object with: forward(e_x, e_y) -> traces_pred (Nt,K)
    traces_obs,            # (Nt,K) torch tensor
    dt,                    # float 
    t_star=0,              # ignore data before t_star
    init=(0, 0),           # initial guess (e_x0, e_y0)
    steps: int=50,         # outer steps
    lr=1,
    lam=1e-6,
    c_init = None,
    c_min = 0.5, 
    c_max = 2, 
    lam_c = 1e-6,
    device="cpu",
    dtype=torch.float32
):
    """
    Inverse solver.
    Forward(e_x,e_y) returns predicted seismograms

    Returns:
      e_hat: (2,) tensor [e_x_hat, e_y_hat]
      traces_pred_final: (Nt,K) tensor
      n_star: int
    """
    traces_obs = traces_obs.to(device=device, dtype=dtype)

    Nt = traces_obs.shape[0]
    n_star = int(round(t_star / dt))
    n_star = max(0, min(n_star, Nt - 1))

    # trainable epicenter
    e_x = nn.Parameter(torch.tensor(float(init[0]), device=device, dtype=dtype))
    e_y = nn.Parameter(torch.tensor(float(init[1]), device=device, dtype=dtype))
    if c_init is not None:
        c = nn.Parameter(torch.tensor(float(c_init), device=device, dtype=dtype))

    def clamp():
        (x_min, x_max), (y_min, y_max) = forward.get_bounds()
        with torch.no_grad():
            e_x.clamp_(x_min + 1e-3, x_max - 1e-3)
            e_y.clamp_(y_min + 1e-3, y_max - 1e-3)
            if c_init is not None:
                c.clamp_(c_min + 1e-3, c_max - 1e-3)
    clamp()

    # fixed scaling based on observed data (stabilizes 2-sensors case)
    scale = traces_obs[n_star:].std(dim=0, unbiased=False).clamp_min(1e-6)

    if c_init is None:
        opt = torch.optim.LBFGS([e_x, e_y], lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    else: 
        opt = torch.optim.LBFGS([e_x, e_y, c], lr=lr, max_iter=20, line_search_fn="strong_wolfe")

    def loss_and_pred():
        traces_pred = forward.forward(e_x, e_y) if c_init is None else forward.forward(e_x, e_y, c) # (Nt,K)
        pred = traces_pred[n_star:]
        obs = traces_obs[n_star:]

        pred = pred / scale
        obs = obs / scale

        loss_data = torch.sum((pred - obs) ** 2)
        loss_reg = lam * (e_x**2 + e_y**2) if c_init is None else lam * (e_x**2 + e_y**2) + lam_c * (c - c_init)**2
        return loss_data + loss_reg, loss_data, traces_pred

    for it in tqdm(range(steps)):
        def closure():
            opt.zero_grad()
            loss, _, _ = loss_and_pred()
            loss.backward()
            return loss

        opt.step(closure)
        clamp()

    with torch.no_grad():
        _, _, traces_pred_final = loss_and_pred()

    e_hat = torch.stack([e_x.detach(), e_y.detach()])
    return e_hat, traces_pred_final.detach(), n_star