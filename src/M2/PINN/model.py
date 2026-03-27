import torch.nn as nn
import torch


def g(t, alpha=1.0):
    return 1-1/torch.cosh(5*t)

class AcousticPINN(nn.Module):
    def __init__(self, n_layers=8, layer_width=32):
        super().__init__()
        self.n_layers = n_layers
        self.layer_width = layer_width
        
        self.lift = nn.Sequential(nn.Linear(5, self.layer_width), nn.Tanh())
        self.layers1 = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_width, self.layer_width), nn.Tanh()) for i in range(self.n_layers)])
        self.layers2 = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_width, self.layer_width), nn.Tanh()) for i in range(self.n_layers)])
        self.compress = nn.Sequential(nn.Linear(self.layer_width, self.layer_width), nn.Tanh())
        self.output = nn.Sequential(nn.Linear(self.layer_width, 1))
                                
    def forward(self, x, y, x0, y0, t):
        inits = torch.cat([x, y, x0, y0, t], dim=-1)
        o = [self.lift(inits)]
        for i in range(self.n_layers):
            o.append(self.layers1[i](o[-1]))
        p = [o[-1]]
        for i in range(self.n_layers):
            p.append(self.layers2[i](p[-1])+o[-(i+1)])            
        return self.output(self.compress(p[-1])) * g(t)
     

class PINNForwardSolver:
    def __init__(self, model, sensors, t_max=5.0, n_t=500,
                 x_min=-5, x_max=5, y_min=-5, y_max=5,
                 device="cpu"):
        self.model = model
        self.sensors = sensors.to(device)
        self.K = sensors.shape[0]
        self.t_max = t_max
        self.Nt = n_t
        self.dt = t_max / (n_t - 1)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.device = device

    def get_bounds(self):
        return (self.x_min, self.x_max), (self.y_min, self.y_max)

    def forward(self, e_x, e_y):
        t = torch.linspace(0, self.t_max, self.Nt, device=self.device).unsqueeze(1)
        traces = []
        for k in range(self.K):
            sx = torch.full((self.Nt, 1), self.sensors[k, 0].item(), device=self.device)
            sy = torch.full((self.Nt, 1), self.sensors[k, 1].item(), device=self.device)
            x0 = torch.full((self.Nt, 1), 1.0, device=self.device) * e_x
            y0 = torch.full((self.Nt, 1), 1.0, device=self.device) * e_y
            p = self.model(sx, sy, x0, y0, t).squeeze()
            traces.append(p)
        return torch.stack(traces, dim=1)  # (Nt, K)