import matplotlib.pyplot as plt
import numpy as np


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

    plt.xlim(0.1, 0.5)
    plt.ylim(-0.4, 0)
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