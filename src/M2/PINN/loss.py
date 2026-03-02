import torch
import torch.nn.functional as F

def laplacian(u, h):
    # u: (B,1,H,W)
    k = torch.tensor([[0,1,0],
                      [1,-4,1],
                      [0,1,0]], device=u.device, dtype=u.dtype).view(1,1,3,3) / (h*h)
    return F.conv2d(u, k, padding=1)

def pde_loss(u_prev, u_curr, u_next, c, f_n, dt, h):
    utt = (u_next - 2*u_curr + u_prev) / (dt*dt)
    lap = laplacian(u_curr, h)
    R = utt - (c*c) * lap - f_n
    return (R*R).mean()
