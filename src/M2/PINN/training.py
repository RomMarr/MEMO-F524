from M2.PINN.model import PINN
from M2.PINN.loss import laplacian, pde_loss
from M2.Utils.source import ricker_source
from M2.Utils.conditions import apply_dirichlet

import torch
from tqdm import tqdm

def train_pinn(epochs: int, dt: float, h: float, Nt: int, c:float,lr:float, device="cpu"):
    model = PINN().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    c = c * torch.ones((1,1,101,101), device=device)
    H = W = 101
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")   # both (W,H)
    x_grid = x_grid.T.contiguous()  # -> (H,W)
    y_grid = y_grid.T.contiguous()  # -> (H,W)

    for epoch in (info:=tqdm(range(epochs))):
        u_prev = torch.zeros((1,1,H,W), device=device)
        u_curr = torch.zeros((1,1,H,W), device=device)
    
        ex = 2 * torch.rand((), device=device) - 1  # scalar tensor
        ey = 2 * torch.rand((), device=device) - 1  # scalar tensor
        f = ricker_source(Nt, H, W,dt, x_grid, y_grid, ex, ey, device=device)    
        for n in range(Nt-1):
            optimizer.zero_grad()

            # u_next = model(u_curr, c) + ((dt*dt) * f[n])   # f[n] : (1,1,H,W)
            u_next = model(u_prev, u_curr, c) + (dt*dt) * f[n]  # f[n] : (1,1,H,W)
            u_next = apply_dirichlet(u_next)

            loss = pde_loss(u_prev, u_curr, u_next, c, f[n], dt, h)
            loss.backward()
            optimizer.step()

            u_prev = u_curr.detach()
            u_curr = u_next.detach()

        if epoch % 10 == 0:
            info.set_description(f"loss={loss.item():.4e}")
    return model