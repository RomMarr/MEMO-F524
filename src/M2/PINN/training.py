from tqdm.auto import tqdm
import torch
import torch.optim as optim
from M2.Utils.visualization import check_seismograms, check_seismograms2
from M2.PINN.loss import loss_fn, loss_fn2


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


from tqdm.auto import tqdm
import torch
import torch.optim as optim
from M2.Utils.visualization import check_seismograms
from M2.PINN.loss import loss_fn


def train_pinn2(
    model,
    fd_seis,
    t_fd,
    sensors,
    e_x,
    e_y,
    n_iterations=10001,
    n_points=int(1e4),
    lr=1e-3,
    device="cpu",
    c=5,
    x_min=-5, x_max=5,
    y_min=-5, y_max=5,
    x0_range=(-2.5, 2.5),
    y0_range=(-2.5, 2.5),
    t_range=(0, 5),
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=5e0, patience=200)

    for step in (pbar := tqdm(range(n_iterations))):
        optimizer.zero_grad()

        x = (torch.rand(n_points, 1, device=device) * (x_max - x_min) + x_min).requires_grad_(True)
        y = (torch.rand(n_points, 1, device=device) * (y_max - y_min) + y_min).requires_grad_(True)
        x0 = (torch.rand(n_points, 1, device=device) * (x0_range[1] - x0_range[0]) + x0_range[0]).requires_grad_(True)
        y0 = (torch.rand(n_points, 1, device=device) * (y0_range[1] - y0_range[0]) + y0_range[0]).requires_grad_(True)
        t = (torch.rand(n_points, 1, device=device) * (t_range[1] - t_range[0]) + t_range[0]).requires_grad_(True)

        loss_pde = loss_fn2(
            model, x, y, x0, y0, t, c=c,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max
        )

        pbar.set_description(f"Loss PDE: {loss_pde.item():.5e}")
        loss_pde.backward()
        optimizer.step()
        # scheduler.step(loss_pde)
        if step % 500 == 0:
            print(f"Iteration {step}")
            check_seismograms2(
                model,
                fd_seis,
                t_fd=t_fd,
                sensors=sensors,
                e_x=e_x,
                e_y=e_y,
                device=device,
            )
    return model