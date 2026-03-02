import numpy as np
from tqdm import tqdm
import torch

# This function builds a supervised dataset to train a NN. It learns the seismograms at sensors, not the full wavefield u(x,y,t).
#  -> Function generated using ChatGPT 
def make_dataset(
    dp_forward_solver,                 # must return u_grid when asked
    n_epicenters=256,
    n_points_per_epi=4096,             # (x,y,t) samples per epicenter
    x_min=-1, x_max=1, y_min=-1, y_max=1,
    seed=42,
):
    rng = np.random.default_rng(seed)

    Nt = dp_forward_solver.Nt
    T  = float(dp_forward_solver.T)
    Nx = dp_forward_solver.Nx
    Ny = dp_forward_solver.Ny

    # grid coordinates (must match DP grid)
    x_grid = np.linspace(x_min, x_max, Nx, dtype=np.float32)
    y_grid = np.linspace(y_min, y_max, Ny, dtype=np.float32)
    t_grid = np.linspace(0.0, T, Nt, dtype=np.float32)

    N = int(n_epicenters) * int(n_points_per_epi)
    X = np.empty((N, 5), dtype=np.float32)   # [x,y,t,ex,ey]
    y = np.empty((N,),   dtype=np.float32)   # u(x,y,t;ex,ey)

    ex_all = rng.uniform(x_min, x_max, size=n_epicenters).astype(np.float32)
    ey_all = rng.uniform(y_min, y_max, size=n_epicenters).astype(np.float32)

    for i, (ex, ey) in tqdm(enumerate(zip(ex_all, ey_all))):
        # DP must provide full wavefield on its grid
        with torch.no_grad():
            # expected: u_grid torch tensor (Nt, Ny, Nx) on CPU ok
            _, u_grid = dp_forward_solver.forward(float(ex), float(ey), return_field=True)
            u_grid = u_grid.detach().cpu().numpy().astype(np.float32)

        # sample indices on the grid
        idx_x = rng.integers(0, Nx, size=n_points_per_epi)
        idx_y = rng.integers(0, Ny, size=n_points_per_epi)

        # time sampling (keep your late-time bias if you want)
        n = n_points_per_epi
        n1 = n // 5
        n2 = n - n1
        idx_t1 = rng.integers(0, Nt, size=n1)
        idx_t2 = rng.integers(int(0.75 * Nt), Nt, size=n2)
        idx_t = np.concatenate([idx_t1, idx_t2])
        rng.shuffle(idx_t)

        xs = x_grid[idx_x]
        ys = y_grid[idx_y]
        ts = t_grid[idx_t]

        j0 = i * n_points_per_epi
        j1 = j0 + n_points_per_epi

        X[j0:j1, 0] = xs
        X[j0:j1, 1] = ys
        X[j0:j1, 2] = ts
        X[j0:j1, 3] = ex
        X[j0:j1, 4] = ey

        # label from wavefield (note ordering u_grid[t, y, x])
        y[j0:j1] = u_grid[idx_t, idx_y, idx_x]

    return X, y


# # Build a dataset to fit the NN model and predict the wavefield u -> Function generated using ChatGPT
# def make_dataset(dp, n_epi=64, device="cpu"):
#     """
#     Build a supervised dataset for learning the wave update operator.

#     Returns:
#         samples: list of tuples
#             Each element is (u_t, c_field, f_t, u_next)
#             Shapes:
#                 u_t, f_t, u_next: (1, Ny, Nx)
#                 c_field:          (1, Ny, Nx)
#     """

#     samples = []

#     (x_min, x_max), (y_min, y_max) = dp.get_bounds()

#     for _ in tqdm(range(n_epi)):

#         # Sample a random epicenter
#         ex = (x_max - x_min) * torch.rand((), device=device) + x_min
#         ey = (y_max - y_min) * torch.rand((), device=device) + y_min

#         # Get full wavefield history from DP solver
#         # u_hist: (Nt, Ny, Nx)
#         with torch.no_grad():
#             _, u_hist = dp.forward(ex, ey, return_field=True)

#         u_hist = u_hist.unsqueeze(1)  # (Nt, 1, Ny, Nx)

#         # Reconstruct source term history
#         f_hist = []
#         for n in range(dp.Nt):
#             t = torch.as_tensor(n * dp.dt, device=device, dtype=dp.dtype)
#             f_hist.append(dp._source(t, ex, ey))
#         f_hist = torch.cat(f_hist, dim=0)  # (Nt, 1, Ny, Nx)

#         # Constant wave speed field
#         c_field = dp.c * torch.ones((1, 1, dp.Ny, dp.Nx),
#                                     device=device,
#                                     dtype=dp.dtype)

#         # Build time-step training pairs
#         for n in range(dp.Nt - 1):

#             u_t   = u_hist[n]       # (1, Ny, Nx)
#             f_t   = f_hist[n]       # (1, Ny, Nx)
#             u_next = u_hist[n + 1]  # (1, Ny, Nx)

#             samples.append((u_t, c_field[0], f_t, u_next))

#     return samples

# Create a dataset to fit a ML model and predict the epicenter.
def make_inverse_dataset(dp_forward_solver, n_epicenters=256,
                         x_min=-1, x_max=1, y_min=-1, y_max=1,
                         seed=42):
    """
    Build supervised dataset for a direct inversion :
      X = seismograms
      y = (e_x, e_y)
    Returns : X (M, Nt*K), y (M, 2)
    """
    rng = np.random.default_rng(seed)
    Nt, K = dp_forward_solver.Nt, dp_forward_solver.K

    ex_all = rng.uniform(x_min, x_max, size=n_epicenters).astype(np.float32)
    ey_all = rng.uniform(y_min, y_max, size=n_epicenters).astype(np.float32)

    X = np.empty((n_epicenters, Nt * K), dtype=np.float32)
    Y = np.stack([ex_all, ey_all], axis=1).astype(np.float32)

    for i, (ex, ey) in tqdm(enumerate(zip(ex_all, ey_all))):
        with torch.no_grad():
            tr = dp_forward_solver.forward(float(ex), float(ey)).detach().cpu().numpy().astype(np.float32)  # (Nt,K)
        X[i] = tr.reshape(-1)  # flatten (Nt*K,)

    return X, Y