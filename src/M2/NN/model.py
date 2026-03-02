import torch
import torch.nn as nn
import torch.nn.functional as F
from M2.Utils.conditions import apply_dirichlet

class Neural_Network(nn.Module):
    def __init__(self, width=256, depth=5, c=None):
        super().__init__()
        layers = [nn.Linear(5 + (1 if c is not None else 0), width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t, e_x, e_y, c=None):
        X = torch.stack([x, y, t, e_x, e_y], dim=-1)  # (B,5)
        if c is not None:
            X = torch.cat([X, c.unsqueeze(-1)], dim=-1)  # (B,6)
        return self.net(X).squeeze(-1)


class NNForwardSolver:
    def __init__(self, model, sensors, Nt, T, device, X_mean, X_std, y_mean, y_std):
        self.model = model
        self.sensors = sensors.to(device)
        self.Nt = Nt
        self.T = T
        self.dt = T / (Nt - 1)
        self.t_grid = torch.linspace(0, T, Nt, device=device)

        self.X_mean = torch.tensor(X_mean, device=device, dtype=torch.float32)
        self.X_std  = torch.tensor(X_std,  device=device, dtype=torch.float32)
        self.y_mean = y_mean
        self.y_std  = y_std

    def get_bounds(self):
        return (-1, 1), (-1, 1)  # hard coded for now

    def forward(self, e_x, e_y, c=None):
        K = self.sensors.shape[0]
        traces = torch.empty((self.Nt, K), device=self.sensors.device)

        xs = self.sensors[:, 0]
        ys = self.sensors[:, 1]

        for n, t in enumerate(self.t_grid):
            ts = t.expand_as(xs)
            ex = e_x.expand_as(xs)
            ey = e_y.expand_as(xs)

            X = torch.stack([xs, ys, ts, ex, ey], dim=-1)
            Xn = (X - self.X_mean) / self.X_std

            u_n = self.model(Xn[:,0], Xn[:,1], Xn[:,2], Xn[:,3], Xn[:,4])
            u = u_n * self.y_std + self.y_mean
            traces[n] = u

        return traces



# class Neural_Network(nn.Module):
#     def __init__(self, hidden=32, in_ch=4):  # u_prev,u_curr,c,f
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, hidden, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(hidden, hidden, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(hidden, 1, 3, padding=1),
#         )

#     def forward(self, u_prev, u_curr, c, f):
#         x = torch.cat([u_prev, u_curr, c, f], dim=1)
#         return self.net(x)

# class NNForwardSolver:
#     def __init__(self, model, dp, sensors, device="cpu"):
#         self.model = model
#         self.dp = dp
#         self.device = device
#         self.sensors = sensors.to(device=device, dtype=dp.dtype)
#         self.K = self.sensors.shape[0]

#     def get_bounds(self):
#         return self.dp.get_bounds()

#     def _sample_sensors(self, u):
#         x = self.sensors[:, 0]
#         y = self.sensors[:, 1]
#         x_n = 2*(x - self.dp.x_min)/(self.dp.x_max - self.dp.x_min) - 1
#         y_n = 2*(y - self.dp.y_min)/(self.dp.y_max - self.dp.y_min) - 1
#         grid = torch.stack([x_n, y_n], dim=-1).view(1, -1, 1, 2)
#         vals = F.grid_sample(u, grid, mode="bilinear", align_corners=True)
#         return vals.view(-1)

#     def forward(self, e_x, e_y):
#         # keep e_x,e_y as tensors for inversion gradients
#         if not isinstance(e_x, torch.Tensor):
#             e_x = torch.tensor(float(e_x), device=self.device, dtype=self.dp.dtype)
#         if not isinstance(e_y, torch.Tensor):
#             e_y = torch.tensor(float(e_y), device=self.device, dtype=self.dp.dtype)

#         c_field = self.dp.c * torch.ones((1,1,self.dp.Ny,self.dp.Nx), device=self.device, dtype=self.dp.dtype)
#         u_prev = torch.zeros((1,1,self.dp.Ny,self.dp.Nx), device=self.device, dtype=self.dp.dtype)
#         u_curr = torch.zeros((1,1,self.dp.Ny,self.dp.Nx), device=self.device, dtype=self.dp.dtype)

#         traces = torch.empty((self.dp.Nt, self.K), device=self.device, dtype=self.dp.dtype)

#         for n in range(self.dp.Nt):
#             traces[n] = self._sample_sensors(u_curr)
#             if n < self.dp.Nt-1:
#                 t = torch.as_tensor(n*self.dp.dt, device=self.device, dtype=self.dp.dtype)
#                 f_t = self.dp._source(t, e_x, e_y)  # (1,1,Ny,Nx)
#                 u_next = self.model(u_prev, u_curr, c_field, f_t)
#                 u_next = apply_dirichlet(u_next)
#                 u_prev, u_curr = u_curr, u_next
#         return traces