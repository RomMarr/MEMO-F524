import matplotlib.pyplot as plt
import numpy as np
import torch


import numpy as np
import torch
import matplotlib.pyplot as plt


def check_seismograms2(
    model,
    fd_seis,
    t_fd,
    sensors,
    e_x=0,
    e_y=0,
    device="cpu",
):
    model.eval()

    t = torch.tensor(t_fd, dtype=torch.float32, device=device).unsqueeze(1)

    plt.figure(figsize=(10, 4 * len(fd_seis)))

    for i, (key, trace_true) in enumerate(fd_seis.items(), start=1):
        sx, sy = sensors[i - 1]

        sx_t = torch.full((len(t_fd), 1), float(sx), device=device)
        sy_t = torch.full((len(t_fd), 1), float(sy), device=device)
        x0_t = torch.full((len(t_fd), 1), float(e_x), device=device)
        y0_t = torch.full((len(t_fd), 1), float(e_y), device=device)

        with torch.no_grad():
            trace_pred = model(sx_t, sy_t, x0_t, y0_t, t).squeeze().cpu().numpy()

        plt.subplot(len(fd_seis), 1, i)
        plt.plot(t_fd, trace_true, label="DP / observed")
        plt.plot(t_fd, trace_pred, "--", label="PINN")
        plt.title(f"Sensor {key}")
        plt.xlabel("t")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def check_seismograms(model, fd_seis: dict,t_fd: np.ndarray, c_val: float = 5.0, t_max: float = 5.0, n_t: int = 500, device: str = "cpu"):
    c  = c_val 
    x  = torch.ones(n_t, 1, device=device)
    y  = torch.zeros(n_t, 1, device=device)
    t  = torch.linspace(0, t_max, n_t, device=device).unsqueeze(1)
 
    with torch.no_grad():
        p_neg1 = model(-x, y, torch.zeros_like(x), torch.zeros_like(y), t).squeeze().cpu().numpy()
        # p_zero = model(0*x, y, torch.zeros_like(x), torch.zeros_like(y), t).squeeze().cpu().numpy()
        p_pos1 = model(x, y, torch.zeros_like(x), torch.zeros_like(y), t).squeeze().cpu().numpy()
 
    t_np = t.squeeze().cpu().numpy()
    # pinn_seis = {'(-1,0)': p_neg1, '(0,0)': p_zero, '(1,0)': p_pos1}
    pinn_seis = {'(-1,0)': p_neg1, '(1,0)': p_pos1}

    
    fig, axes = plt.subplots(len(pinn_seis),1, figsize=(12, 7))
    if len(pinn_seis) == 2:
        receiver_labels = ['(-1,0)', '(1,0)']


 
    for ax, label in zip(axes, receiver_labels):
        ax.plot(t_np,  pinn_seis[label], label='PINN',linewidth=1.5)
        ax.plot(t_fd,  fd_seis[label],   label='Reference', linewidth=1.5,
                linestyle='--', color='tomato')
        ax.set_title(f"Receiver {label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
 
    # fig.suptitle(f"Seismograms — c = {c_val}  (PINN vs Finite Difference reference)",
    fig.suptitle(f"Seismograms : observed vs PINN",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_epicenters(e_true,
                    e_hat_dp=None,
                    e_hat_pinn=None,
                    e_hat_nn=None,
                    e_hat_ml=None,
                    e_hat_pinn_t03=None,
                    x_min=-1, x_max=1,
                    y_min=-1, y_max=1):

    plt.figure(figsize=(6,6))

    # Draw square domain
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             'k-', linewidth=1)

    # Grid
    ticks = np.linspace(-1, 1, 9)
    for t in ticks:
        plt.plot([t, t], [y_min, y_max], color='lightgray', linewidth=0.5)
        plt.plot([x_min, x_max], [t, t], color='lightgray', linewidth=0.5)

    # True epicenter
    plt.scatter(e_true[0], e_true[1],
                marker='*', s=200, label='True')

    # Estimates
    if e_hat_dp is not None:
        plt.scatter(e_hat_dp[0], e_hat_dp[1],
                    marker='.', s=100, label='DP')

    if e_hat_pinn is not None:
        plt.scatter(e_hat_pinn[0], e_hat_pinn[1],
                    marker='.', s=100, label='PINN')

    if e_hat_nn is not None:
        plt.scatter(e_hat_nn[0], e_hat_nn[1],
                    marker='.', s=100, label='NN')
    if e_hat_ml is not None:
        plt.scatter(e_hat_ml[0], e_hat_ml[1],
                    marker='.', s=100, label='RF')
    if e_hat_pinn_t03 is not None:
        plt.scatter(e_hat_pinn_t03[0], e_hat_pinn_t03[1],
                    marker='.', s=100, label='PINN (t* = 0.8)')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted epicenters")
    plt.legend()
    plt.gca().set_aspect('equal', 'box')
    plt.show()



def plot_seimograms(traces_obs, traces_nn, traces_pinn, T, Nt):
    t = np.linspace(0, T, Nt)
    K = traces_obs.shape[1]
    plt.figure(figsize=(10, 4 * K))

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(t, traces_obs[:, k], label="DP (true)", linewidth=2)
        plt.plot(t, traces_nn[:, k], "--", label="NN", linewidth=2)
        plt.plot(t, traces_pinn[:, k], ":", label="PINN - t* = 0.0", linewidth=2)
        # plt.plot(t, traces_pinn_t03[:, k], ":", label="PINN - t* = 0.3", linewidth=2)
        plt.ylabel(f"Sensor {k}")
        plt.legend()
        plt.grid(True)

    plt.xlabel("Time")
    plt.suptitle("True vs predicted seismograms (forward operator surrogate)")
    plt.tight_layout()
    plt.show()