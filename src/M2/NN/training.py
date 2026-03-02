from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from M2.NN.model import Neural_Network
from M2.Utils.conditions import apply_dirichlet

# def train_nn(dp, epochs=1000, lr=1e-3, device="cpu", print_every=50):
#     model = Neural_Network(hidden=32).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)

#     H, W = dp.Ny, dp.Nx
#     c_field = (dp.c * torch.ones((1,1,H,W), device=device, dtype=dp.dtype))

#     (x_min,x_max),(y_min,y_max) = dp.get_bounds()

#     for ep in (info:=tqdm(range(epochs))):
#         # random epicenter each epoch
#         ex = (x_max-x_min)*torch.rand((), device=device) + x_min
#         ey = (y_max-y_min)*torch.rand((), device=device) + y_min

#         # DP trajectory (labels)
#         with torch.no_grad():
#             _, u_hist = dp.forward(ex, ey, return_field=True)   # (Nt,Ny,Nx)
#         u_hist = u_hist.unsqueeze(1).to(device)                 # (Nt,1,Ny,Nx)

#         # choose how many time steps you train on each epoch
#         # full rollout is expensive; you can subsample
#         n0 = 1
#         n1 = dp.Nt - 2
#         idx = torch.arange(n0, n1, device=device)
    
#         total = 0
#         for n in idx.tolist():
#             u_prev = u_hist[n-1:n]          # (1,1,H,W)
#             u_curr = u_hist[n:n+1]
#             u_next_true = u_hist[n+1:n+2]
#             t = torch.as_tensor(n * dp.dt, device=device, dtype=dp.dtype)
#             f_n = dp._source(t, ex, ey)

#             pred = model(u_prev, u_curr, c_field, f_n)
#             pred = apply_dirichlet(pred)

#             loss = F.mse_loss(pred, u_next_true)

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()

#             total += float(loss.detach())

#         if ep % print_every == 0:
#             # denom = len(idx) if hasattr(idx, "__len__") else 1
#             info.set_description(f"mse={total/len(idx):.3e}")

#     model.eval()
#     return model



def train_nn(model, X, y, epochs=50, batch_size=4096, lr=1e-3, device="cpu"):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    model.train()
    for ep in range(epochs):
        total = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch[:,0], x_batch[:,1], x_batch[:,2], x_batch[:,3], x_batch[:,4])
            loss = loss_fn(pred, y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * x_batch.size(0)

        if ep % 10 == 0:
            print(f"ep {ep:03d}  mse={total/len(dataset):.3e}")

    model.eval()
    return model