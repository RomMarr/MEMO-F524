import math
import torch
import torch.nn as nn
from tqdm import tqdm


# Inverse problem solver for epicenter estimation
def inverse_function(
    forward,               # forward(e_x, e_y) -> traces_pred (Nt,K)
    traces_obs,            # (Nt,K) torch tensor
    dt, t_star=0,          # ignore data before t_star
    init=(0, 0),           # initial epicenter estimate/guess (e_x0, e_y0)
    steps: int=50,         # outer steps
    lr=1, lam=1e-6, max_iter=3,
    device="cpu", dtype=torch.float32, show_progress=True
):
    traces_obs = traces_obs.to(device=device, dtype=dtype)

    Nt = traces_obs.shape[0]
    n_star = int(round(t_star / dt))
    n_star = max(0, min(n_star, Nt - 1))

    # trainable epicenter
    e_x = nn.Parameter(torch.tensor(float(init[0]), device=device, dtype=dtype))
    e_y = nn.Parameter(torch.tensor(float(init[1]), device=device, dtype=dtype))

    def clamp():
        (x_min, x_max), (y_min, y_max) = forward.get_bounds()
        with torch.no_grad():
            e_x.clamp_(-3.5, 3.5)
            e_y.clamp_(-3.5, 3.5)
    clamp()

    # fixed scaling based on observed data (stabilizes 2-sensors case)
    scale = traces_obs[n_star:].std(dim=0, unbiased=False).clamp_min(1e-6)

    opt = torch.optim.LBFGS([e_x, e_y], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
   
   # closure function that computes the loss and predictions for the current epicenter estimate
    def loss_and_pred():
        traces_pred = forward.forward(e_x, e_y) # (Nt,K)
        pred = traces_pred[n_star:]
        obs = traces_obs[n_star:]

        pred = pred / scale
        obs = obs / scale

        loss_data = torch.sum((pred - obs) ** 2)
        loss_reg = lam * (e_x**2 + e_y**2)
        return loss_data + loss_reg, loss_data, traces_pred
    
    # store history of epicenter estimates for visualization
    history = []
    iterator = range(steps) if not show_progress else tqdm(range(steps))
    for it in iterator:
        def closure():
            opt.zero_grad()
            loss, _, _ = loss_and_pred()
            loss.backward()
            return loss

        opt.step(closure)
        clamp()
        history.append((e_x.item(), e_y.item()))

    # final evaluation of the loss and predictions after optimization
    with torch.no_grad():
        _, _, traces_pred_final = loss_and_pred()

    # return the estimated epicenter, final predicted traces, index of t_star, and history of epicenter estimates
    e_hat = torch.stack([e_x.detach(), e_y.detach()])
    return e_hat, traces_pred_final.detach(), n_star, history


# Differentiable version of the inverse problem solver for epicenter estimation
def inverse_function_differentiable(
    forward, traces_obs,          # (Nt, K) —> not detached, gradient flows through
    dt, t_star=0, init=(0, 0),
    steps=20, lr=0.05,lam=1e-6,
    device="cpu", dtype=torch.float32,
):
    Nt = traces_obs.shape[0]
    n_star = int(round(t_star / dt))
    n_star = max(0, min(n_star, Nt - 1))

    scale = traces_obs[n_star:].std(dim=0, unbiased=False).clamp_min(1e-6).detach()

    # initialise epicenter as a plain tensor
    e_x = torch.tensor(float(init[0]), device=device, dtype=dtype, requires_grad=True)
    e_y = torch.tensor(float(init[1]), device=device, dtype=dtype, requires_grad=True)

    # (x_min, x_max), (y_min, y_max) = forward.get_bounds()

    for _ in range(steps):
        traces_pred = forward.forward(e_x, e_y)
        pred = traces_pred[n_star:] / scale
        obs  = traces_obs[n_star:]  / scale

        loss = torch.sum((pred - obs) ** 2) + lam * (e_x**2 + e_y**2)

        grad_x, grad_y = torch.autograd.grad(loss, [e_x, e_y], create_graph=True)

        # fixed-step gradient descent (no optimizer object, fully in-graph)
        e_x = e_x - lr * grad_x
        e_y = e_y - lr * grad_y

        # differentiable clamp (soft clamp keeps gradients alive)
        e_x = e_x.clamp(-3.5, 3.5)
        e_y = e_y.clamp(-3.5, 3.5)

    return torch.stack([e_x, e_y])


