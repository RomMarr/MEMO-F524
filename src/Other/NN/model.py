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