import torch

def ricker_source(Nt, H, W, dt, x_grid, y_grid, ex, ey,A=5e-1, f0=5, t0=0.4, gamma=50, device="cpu"):
    t = torch.arange(Nt, device=device) * dt
    tau = t - t0
    pi2f2 = (torch.pi**2) * (f0**2)
    ricker = (1 - 2*pi2f2*(tau**2)) * torch.exp(-pi2f2*(tau**2))  # (Nt,)

    space = torch.exp(-gamma * ((x_grid - ex)**2 + (y_grid - ey)**2))  # (H,W)

    f = A * ricker[:, None, None] * space[None, :, :]  # (Nt,H,W)
    return f.unsqueeze(1)  # (Nt,1,H,W)
