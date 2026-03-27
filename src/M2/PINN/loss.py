import torch 
from M2.Utils.source import source

def loss_fn(model, x, y, x0, y0, t):
    pred = model(x, y, x0, y0, t)
    ones = torch.ones_like(pred)
    dp_dx, dp_dy, dp_dt = torch.autograd.grad(pred, [x, y, t], grad_outputs=ones, create_graph=True)
    dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]
    pde_res = dp_dtt - (5**2) * (dp_dxx + dp_dyy) - source(x, y, t, cx=x0, cy=y0)
    return (pde_res**2).mean()