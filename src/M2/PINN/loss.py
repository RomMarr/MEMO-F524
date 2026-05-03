import torch 
import torch.nn.functional as F

from M2.Utils.source import source
# from M2.Utils.conditions import c

def laplacian(u, h):
    # u: (B,1,H,W)
    k = torch.tensor([[0,1,0],
                      [1,-4,1],
                      [0,1,0]], device=u.device, dtype=u.dtype).view(1,1,3,3) / (h*h)
    return F.conv2d(u, k, padding=1)



# def loss_fn(model, x, y, x0, y0, t):
#     pred = model(x, y, x0, y0, t)
#     ones = torch.ones_like(pred)
#     dp_dx, dp_dy, dp_dt = torch.autograd.grad(pred, [x, y, t], grad_outputs=ones, create_graph=True)
#     dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
#     dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
#     dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]
#     pde_res = dp_dtt - (5**2) * (dp_dxx + dp_dyy) - source(x, y, t, cx=x0, cy=y0)
#     return (pde_res**2).mean()


def loss_fn2(model, x, y, x0, y0, t):
    pred = model(x, y, x0, y0, t)
    ones = torch.ones_like(pred)

    dp_dx, dp_dy, dp_dt = torch.autograd.grad(pred, [x, y, t], grad_outputs=ones, create_graph=True)
    dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]

    c_val = c(x, y, t).to(device=x.device, dtype=x.dtype)

    pde_res = dp_dtt - (c_val ** 2) * (dp_dxx + dp_dyy) - source(x, y, t, cx=x0, cy=y0)

    return (pde_res ** 2).mean()

def loss_fn(model, x, y, t, x0, y0, t0):
    pred = model(x, y, t, x0, y0, t0)
    s_xyt = source(x, y, t, cx=x0, cy=y0, t0=t0)
    ones = torch.ones_like(pred)
    dp_dx, dp_dy, dp_dt = torch.autograd.grad(pred, [x, y, t], grad_outputs=ones, create_graph=True)
    dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]
    pde_res = dp_dtt - (c(x, y, t)**2) * (dp_dxx + dp_dyy) - s_xyt
    return (pde_res**2).mean()