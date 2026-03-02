import torch
import torch.nn as nn
import torch.nn.functional as F

from M2.Utils.conditions import apply_dirichlet

class PINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        # self.layer1 = nn.Conv2d(2,1,kernel_size=3,padding="same")
        # self.layer1 = nn.Conv2d(3,1,kernel_size=3,padding="same")
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.Conv2d(width, width, kernel_size=3, padding="same"),
            nn.Tanh(),
            nn.Conv2d(width, 1, kernel_size=3, padding="same"),
        )

        
    def forward(self, u_prev, u_curr, c):
        # return self.layer1(torch.cat([u_prev, u_curr, c], dim=1))
        return self.net(torch.cat([u_prev, u_curr, c], dim=1))
    # def forward(self, u_t, c):
    #     return self.layer1(torch.cat([u_t, c], dim=1))


class PinnForwardSolver:
    def __init__(self, model, sensors, x_min, x_max, y_min, y_max, Nx, Ny, Nt, dt, h, device="cpu"):
        self.model = model
        self.device = device

        self.sensors = sensors.to(device)
        self.K = self.sensors.shape[0]

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.dt, self.h = dt, h

        x = torch.linspace(self.x_min, self.x_max, self.Nx, device=device)
        y = torch.linspace(self.y_min, self.y_max, self.Ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        self.X = X.T.contiguous()
        self.Y = Y.T.contiguous()

    def get_bounds(self):
        return (self.x_min, self.x_max), (self.y_min, self.y_max)

    def _sample_sensors(self, u):
        # u: (1,1,Ny,Nx) -> (K,)
        x = self.sensors[:, 0]
        y = self.sensors[:, 1]
        x_n = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_n = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        grid = torch.stack([x_n, y_n], dim=-1).view(1, -1, 1, 2)  # (1,K,1,2)
        vals = F.grid_sample(u, grid, mode="bilinear", align_corners=True)  # (1,1,K,1)
        return vals.view(-1)

    def _make_source_F(self, ex, ey, A=5, f0=10, t0=0.1, gamma=50):
        t = torch.arange(self.Nt, device=self.device) * self.dt
        tau = t - t0
        pi2f2 = (torch.pi**2) * (f0**2)
        ricker = (1 - 2*pi2f2*(tau**2)) * torch.exp(-pi2f2*(tau**2))  # (Nt,)

        space = torch.exp(-gamma * ((self.X - ex)**2 + (self.Y - ey)**2))  # (Ny,Nx)
        Fsrc = A * ricker[:, None, None] * space[None, :, :]              # (Nt,Ny,Nx)
        return Fsrc.unsqueeze(1)  # (Nt,1,Ny,Nx)

    def forward(self, ex, ey):
        """
        ex,ey: torch scalars 
        """
        c_scalar = torch.as_tensor(1.0, device=self.device, dtype=self.X.dtype)
        c_field = c_scalar * torch.ones((1,1,self.Ny,self.Nx), device=self.device, dtype=self.X.dtype)

        F = self._make_source_F(ex, ey)  # (Nt,1,Ny,Nx)

        u_prev = torch.zeros((1,1,self.Ny,self.Nx), device=self.device, dtype=self.X.dtype)
        u_curr = torch.zeros((1,1,self.Ny,self.Nx), device=self.device, dtype=self.X.dtype)

        traces = torch.empty((self.Nt, self.K), device=self.device, dtype=self.X.dtype)

        for n in range(self.Nt):
            traces[n] = self._sample_sensors(u_curr)
            if n < self.Nt - 1:
                u_next = self.model(u_prev, u_curr, c_field) + (self.dt*self.dt) * F[n]                
                # u_next = self.model(u_curr, c_field) + (self.dt*self.dt) * F[n]
                u_next = apply_dirichlet(u_next)
                u_prev, u_curr = u_curr, u_next 

        return traces