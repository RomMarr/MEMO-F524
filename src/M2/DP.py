import torch
import torch.nn.functional as F
import math

from M2.Utils.conditions import apply_dirichlet
from M2.PINN.loss import laplacian

class DPForwardSolver:
    """
    Differentiable-physics forward solver for the 2D acoustic wave equation
    using an explicit second-order finite-difference time-stepping scheme.

    forward(e_x, e_y) -> seismograms (Nt, K)
    """

    def __init__(
        self,
        sensors: torch.Tensor,         # (K,2)
        c=1,
        x_min=-1, x_max=1, y_min=-1, y_max=1,
        Nx=101, Ny=101, Nt=401, T=2,
        A=5, t0=0.1, f0=10, gamma=50, # source params
        device="cpu", dtype=torch.float32
    ):

        self.device = device
        self.dtype = dtype

        # domain / grid
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.Nx, self.Ny, self.Nt, self.T = Nx, Ny, Nt, T

        # physics
        self.c = torch.as_tensor(c, device=self.device, dtype=self.dtype)

        # source
        self.A = A
        self.t0 = t0
        self.f0 = f0
        self.gamma = gamma

        # sensors
        self.sensors = sensors.to(self.device, dtype=self.dtype)
        self.K = self.sensors.shape[0]

        # build grid
        self.x = torch.linspace(self.x_min, self.x_max, self.Nx, device=self.device, dtype=self.dtype)
        self.y = torch.linspace(self.y_min, self.y_max, self.Ny, device=self.device, dtype=self.dtype)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        if abs(self.dx - self.dy) > 1e-7:
            raise ValueError("DPForwardSolver2D requires dx == dy (uniform square grid) for this stencil.")
        self.h = self.dx

        self.dt = self.T / (self.Nt - 1)

        # CFL check
        if self.c * self.dt / self.h > 1 / math.sqrt(2):
            raise ValueError("CFL unstable: decrease dt (increase Nt), increase h (decrease Nx/Ny), or reduce c.")

        # meshgrid for source evaluation (Ny, Nx)
        X, Y = torch.meshgrid(self.x, self.y, indexing="xy")  # (Nx,Ny)
        self.X = X.T.contiguous()  # (Ny,Nx)
        self.Y = Y.T.contiguous()        

    def get_bounds(self):
        return (self.x_min, self.x_max), (self.y_min, self.y_max)

    def _source(self, t: torch.Tensor, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
        # returns (1,1,Ny,Nx)

        # Ricker wavelet in time
        tau = t - self.t0
        pi2f2 = (math.pi**2) * (self.f0**2)
        time_env = self.A * (1 - 2 * pi2f2 * tau**2) * torch.exp(-pi2f2 * tau**2)

        # Spatial localization
        space_env = torch.exp(-self.gamma * ((self.X - e_x)**2 + (self.Y - e_y)**2))
        f = time_env * space_env
        return f.unsqueeze(0).unsqueeze(0)

    def _sample_sensors(self, u: torch.Tensor) -> torch.Tensor:
        # u: (1,1,Ny,Nx) -> (K,)
        x = self.sensors[:, 0]
        y = self.sensors[:, 1]
        x_n = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_n = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        grid = torch.stack([x_n, y_n], dim=-1).view(1, -1, 1, 2)  # (1,K,1,2)
        vals = F.grid_sample(u, grid, mode="bilinear", align_corners=True)  # (1,1,K,1)
        return vals.view(-1)

    def forward(self, e_x, e_y, return_field=False, c=None):
        """
        e_x, e_y: epicenter coordinates (float, tensor, or nn.Parameter)
        c: wave speed 

        seismograms: (Nt, K)
        """
        if not isinstance(e_x, torch.Tensor):
            e_x = torch.tensor(float(e_x), device=self.device, dtype=self.dtype)
        else:
            e_x = e_x.to(device=self.device,dtype=self.dtype)

        if not isinstance(e_y, torch.Tensor):
            e_y = torch.tensor(float(e_y), device=self.device, dtype=self.dtype)
        else:
            e_y = e_y.to(device=self.device,dtype=self.dtype)

        if c is None:
            c = self.c
        else:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(float(c), device=self.device, dtype=self.dtype)
            else:
                c = c.to(device=self.device, dtype=self.dtype)

        u_prev = torch.zeros((1, 1, self.Ny, self.Nx), device=self.device, dtype=self.dtype)
        u_curr = torch.zeros((1, 1, self.Ny, self.Nx), device=self.device, dtype=self.dtype)

        seismograms = torch.zeros((self.Nt, self.K), device=self.device, dtype=self.dtype)
        if return_field:
            u_hist = torch.empty((self.Nt, self.Ny, self.Nx), device=self.device, dtype=self.dtype)


        cdt2 = (c * self.dt) ** 2
        dt2 = (self.dt ** 2)

        for n in range(self.Nt):
            t = torch.as_tensor(n * self.dt, device=self.device, dtype=self.dtype)
            seismograms[n] = self._sample_sensors(u_curr)

            if return_field:
                u_hist[n] = u_curr[0, 0]

            if n < self.Nt - 1:
                lap = laplacian(u_curr, self.h)
                f = self._source(t, e_x, e_y)
                u_next = 2 * u_curr - u_prev + cdt2 * lap + dt2 * f
                u_next = apply_dirichlet(u_next)
                u_prev, u_curr = u_curr, u_next
        if return_field:
            return seismograms, u_hist
        return seismograms