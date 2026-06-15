import torch
import torch.nn as nn
from M2.PINN.model import PINNForwardSolver
from M2.inverse_problem import inverse_function, inverse_function_differentiable


# Sensor placement optimization using bilevel optimization with a PINN 
# forward solver and differentiable inner inversion
def optimize_sensors(
    pinn_model, sensors_init, 
    epicenters,             # (M, 2), sampled epicenter positions
    guess_init=(1.0, 1.0),
    t_max=5.0, n_t=500,
    x_min=-5, x_max=5,
    y_min=-5, y_max=5,
    inner_steps=20, inner_lr=0.05, outer_steps=100, outer_lr=0.01,
    device="cpu",
):
    # make sensor positions trainable
    sensors = nn.Parameter(sensors_init.clone().to(device))

    solver = PINNForwardSolver(
        model=pinn_model,
        sensors=sensors,
        t_max=t_max, n_t=n_t,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        device=device,
    )

    # outer optimizer for sensor positions
    outer_opt = torch.optim.Adam([sensors], lr=outer_lr)
    history = []

    # outer loop for sensor placement optimization
    for step in range(outer_steps):
        outer_opt.zero_grad()

        total_error = torch.tensor(0.0, device=device)
        for m in range(epicenters.shape[0]): # loop over all epicenters in the batch
            ex_true = epicenters[m, 0]
            ey_true = epicenters[m, 1]

            # observed traces at current sensor positions (keep in graph)
            traces_obs = solver.forward(ex_true, ey_true)

            # inner inversion
            e_hat = inverse_function_differentiable(
                forward=solver, traces_obs=traces_obs,
                dt=solver.dt, init=guess_init,
                steps=inner_steps, lr=inner_lr,
                device=device,
            )
            total_error = total_error + (e_hat[0] - ex_true)**2 + (e_hat[1] - ey_true)**2

        # regularization terms to encourage good sensor placement
        dist_between = torch.norm(sensors[0] - sensors[1])
        reg_separation =  0.1 * torch.exp(-dist_between)  # penalize collapse
        reg_center = 0.01 * (sensors ** 2).mean() # penalize sensors far from center
        loss = total_error / epicenters.shape[0] + reg_center + reg_separation
        loss.backward()

        if step == 0:
            print("Initial sensors:", sensors_init.cpu().numpy())
            print("sensors.grad after backward:", sensors.grad)
            print("grad norm:", sensors.grad.norm().item() if sensors.grad is not None else "None")
            
        outer_opt.step()

        # clamp sensors inside domain
        with torch.no_grad():
            sensors.clamp_(-3.5, 3.5)  # keep some margin from the edges

        history.append(sensors.detach().clone())

        if step % 10 == 0:
            print(f"Step {step:3d} | Outer loss: {loss.item():.4f} | "
                  f"Sensors: {sensors.detach().cpu().numpy()}")

    return sensors.detach(), history

# Local search around the best sensor configuration found by random search
def local_search(
    pinn_model, sensors_best, epicenters,
    n_neighbors=10, perturbation=0.5,
    inner_steps=1, inner_lr=0.05,
    outer_steps=40, outer_lr=0.01,
    device="cpu",
):
    best_loss = evaluate_loss(pinn_model, sensors_best, epicenters, device)
    print(f"Initial loss: {best_loss:.4f}")

    # loop over random perturbations of the best sensor configuration
    for i in range(n_neighbors):
        # perturb current best
        noise = torch.randn_like(sensors_best) * perturbation
        sensors_init = (sensors_best + noise).clamp(-3.5, 3.5)

        # refine with gradient descent
        sensors_opt, history = optimize_sensors(
            pinn_model=pinn_model,
            sensors_init=sensors_init,
            epicenters=epicenters,
            inner_steps=inner_steps, inner_lr=inner_lr,
            outer_steps=outer_steps, outer_lr=outer_lr,
            device=device,
        )

        loss = evaluate_loss(pinn_model, sensors_opt, epicenters, device)
        if loss < best_loss:
            best_loss = loss
            sensors_best = sensors_opt
    return sensors_best, best_loss


# Evaluate the average loss over a set of epicenters for a given sensor configuration
def evaluate_loss(pinn_model, sensors, epicenters, device):
    solver = PINNForwardSolver(
        model=pinn_model, sensors=sensors,
        t_max=5.0, n_t=500,
        x_min=-5, x_max=5, y_min=-5, y_max=5,
        device=device,
    )
    total = 0
    # loop over all epicenters and compute the average squared error of the estimated epicenter from the true one
    for m in range(epicenters.shape[0]):
        ex, ey = epicenters[m, 0].item(), epicenters[m, 1].item()
        traces_obs = solver.forward(ex, ey).detach()
        e_hat, _, _, _ = inverse_function(
            forward=solver, traces_obs=traces_obs,
            dt=solver.dt, init=(0, 0),
            steps=30, lr=1, max_iter=3, device=device, show_progress=False
        )
        total += ((e_hat[0] - ex)**2 + (e_hat[1] - ey)**2).item()
    return total / epicenters.shape[0]