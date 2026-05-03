import torch

def ricker_source(Nt, H, W, dt, x_grid, y_grid, ex, ey,A=5, f0=10, t0=0.1, gamma=50, device="cpu"):
    t = torch.arange(Nt, device=device) * dt
    tau = t - t0
    pi2f2 = (torch.pi**2) * (f0**2)
    ricker = (1 - 2*pi2f2*(tau**2)) * torch.exp(-pi2f2*(tau**2))  # (Nt,)

    space = torch.exp(-gamma * ((x_grid - ex)**2 + (y_grid - ey)**2))  # (H,W)

    f = A * ricker[:, None, None] * space[None, :, :]  # (Nt,H,W)
    return f.unsqueeze(1)  # (Nt,1,H,W)


def source(x, y, t, cx=0, cy=0, sigma=1, amplitude_g=1e2, f=1.0, amplitude_r=1.0, t0=1.0):
    dx, dy = x - cx, y - cy
    gaussian = amplitude_g * torch.exp(-(dx**2/(2*sigma**2)+dy**2/(2*sigma**2)))
    tau = t - t0
    pi2f2tau2 = (torch.pi * f)**2 * tau**2
    ricker = amplitude_r * (1.0 - 2.0 * pi2f2tau2) * torch.exp(-pi2f2tau2)
    return gaussian * ricker