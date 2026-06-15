import torch.nn as nn
import torch

# Causal multiplier function to ensure that the source is null before activation time t0
def g(t, t0, alpha=5.0):
    return 0.5 * (torch.tanh(alpha * (t + t0)) + 1)

# Physics-Informed Neural Network (PINN) for the 2D acoustic wave equation
class PINN(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(6, width), nn.Mish())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.Tanh())
            for _ in range(depth - 1)
        ])
        self.output_layer = nn.Linear(width, 1)

    def forward(self, x, y, t, x0, y0, t0):
        inp = torch.cat([x, y, t, x0, y0, t0], dim=-1)
        h = self.input_layer(inp)
        for layer in self.hidden_layers:
            h = layer(h) + h
        p = self.output_layer(h)
        return p * g(t - t0, t0/10)


# PINN-based forward solver that evaluates the PINN at sensor locations and time steps
class PINNForwardSolver:
    def __init__(self, model, sensors, t_max=5.0, n_t=500,t0=1.0,
                 x_min=-5, x_max=5, y_min=-5, y_max=5,
                 device="cpu"):
        self.model = model
        self.sensors = sensors.to(device)
        self.K = sensors.shape[0]
        self.t_max = t_max
        self.t0 = t0
        self.Nt = n_t
        self.dt = t_max / (n_t - 1)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.device = device

    def get_bounds(self):
        return (self.x_min, self.x_max), (self.y_min, self.y_max)

    # Evaluate the PINN at sensor locations and time steps to get seismograms
    def forward(self, e_x, e_y):
        if not isinstance(e_x, torch.Tensor):
            e_x = torch.tensor(float(e_x), device=self.device)
        if not isinstance(e_y, torch.Tensor):
            e_y = torch.tensor(float(e_y), device=self.device)

        t = torch.linspace(0, self.t_max, self.Nt, device=self.device).unsqueeze(1)
        t0 = torch.full_like(t, float(self.t0))
        traces = []
        for k in range(self.K):
            sx = self.sensors[k, 0].expand(self.Nt, 1)
            sy = self.sensors[k, 1].expand(self.Nt, 1)
            x0 = e_x.expand(self.Nt, 1)
            y0 = e_y.expand(self.Nt, 1)
            p = self.model(sx, sy, t, x0, y0, t0).squeeze()
            traces.append(p)
        return torch.stack(traces, dim=1)  # (Nt, K)