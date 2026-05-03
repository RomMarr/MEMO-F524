import torch
import numpy as np

def apply_dirichlet(u: torch.Tensor) -> torch.Tensor:
    v = u.clone()
    v[..., 0, :] = 0
    v[..., -1, :] = 0
    v[..., :, 0] = 0
    v[..., :, -1] = 0
    return v

# def c(x, y, t=0):
#     if type(y) == np.ndarray:
#         y = torch.tensor(y)
#     if type(x) == np.ndarray:
#         x = torch.tensor(x)

#     step = 10 / 6

#     result = torch.where((y > 8*step)  & (x > 3),    6.5,
#              torch.where((y > 8*step)  & (x <= 3),   11.8,
#              torch.where((y > 6*step)  & (x > 5),    8.3,
#              torch.where((y > 6*step)  & (x > 1),    4.7,
#              torch.where((y > 6*step)  & (x <= 1),   13.1,
#              torch.where((y > 4*step)  & (x > 7),    7.1,
#              torch.where((y > 4*step)  & (x > 2),    9.4,
#              torch.where((y > 4*step)  & (x <= 2),   14.2,
#              torch.where((y > 2*step)  & (x > 6),    5.5,
#              torch.where((y > 2*step)  & (x > 3),    10.9,
#              torch.where((y > 2*step)  & (x > 0),    12.7,
#              torch.where((y > 2*step)  & (x <= 0),   4.3,
#              torch.where((y > step)    & (x > 8),    8.2,
#              torch.where((y > step)    & (x > 4),    13.1,
#              torch.where((y > step)    & (x <= 4),   6.6,
#              torch.where((y > 0)       & (x > 5),    11.9,
#              torch.where((y > 0)       & (x > 2),    14.4,
#              torch.where((y > 0)       & (x <= 2),   5.8,
#              torch.where((y > -step)   & (x > 6),    9.3,
#              torch.where((y > -step)   & (x > 1),    7.7,
#              torch.where((y > -step)   & (x <= 1),   4.6,
#              torch.where((y > -2*step) & (x > 4),    12.5,
#              torch.where((y > -2*step) & (x > 0),    14.1,
#              torch.where((y > -2*step) & (x <= 0),   6.8,
#              torch.where((y > -4*step) & (x > 7),    5.2,
#              torch.where((y > -4*step) & (x > 3),    10.9,
#              torch.where((y > -4*step) & (x <= 3),   8.5,
#              torch.where((y > -6*step) & (x > 5),    13.1,
#              torch.where((y > -6*step) & (x > 1),    4.6,
#              torch.where((y > -6*step) & (x <= 1),   11.4,
#              torch.where((y > -8*step) & (x > 6),    7.2,
#              torch.where((y > -8*step) & (x > 2),    14.7,
#              torch.where((y > -8*step) & (x <= 2),   9.7,
#                                                       6.5
#              )))))))))))))))))))))))))))))))))

#     return result

# from scipy.ndimage import zoom
# from scipy.interpolate import RegularGridInterpolator

# c_map = np.load("Density_Test.npy")[8].reshape((51, 51))
# c_map = zoom(c_map, 1024/51)

# coords = np.linspace(-10, 10, 1024)
# interp = RegularGridInterpolator((coords, coords), c_map, method='nearest', bounds_error=False, fill_value=None)

# def c(x, y, t=0):
#     if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
#     if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
#     return torch.tensor(interp(np.stack([y.ravel(), x.ravel()], axis=-1)).reshape(x.shape), dtype=torch.float32).to(device)
    