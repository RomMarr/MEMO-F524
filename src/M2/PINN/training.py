from tqdm.auto import tqdm
import torch
import torch.optim as optim
from M2.Utils.visualization import check_seismograms
from M2.PINN.loss import loss_fn


def train_pinn(model, fd_seis,t_fd, n_iterations=10001, n_points=int(1e4), lr=1e-3, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=5e0, patience=200)

    for step in (pbar:=tqdm(range(n_iterations))):
        optimizer.zero_grad()
        x = (torch.rand(n_points, 1, requires_grad=True, device=device)-0.5)*10
        y = (torch.rand(n_points, 1, requires_grad=True, device=device)-0.5)*10
        x0 = (torch.rand(n_points, 1, requires_grad=True, device=device)-0.5)*5
        y0 = (torch.rand(n_points, 1, requires_grad=True, device=device)-0.5)*5
        t = (torch.rand(n_points, 1, requires_grad=True, device=device))*15

        loss_pde = loss_fn(model, x, y, x0, y0, t)

        pbar.set_description(f"Loss PDE: {loss_pde.item():.5e}")#, l_r: {scheduler.get_last_lr()[-1]:.3e}")
        loss_pde.backward()
        optimizer.step()
        # scheduler.step(loss_pde)
        if step%500 == 0:
            print(f"Iteration {step}")
            check_seismograms(model, fd_seis, t_fd=t_fd, device=device)
    return model