import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from M1.utils import NMSE_by_coordinate


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=3, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, padding=1, groups=dim),  # Add padding to Conv2d
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

def train_convmixer(X_train, y_train, X_test, dim, depth, kernel_size, patch_size, lr, n_epochs):
    model = ConvMixer(dim, depth, kernel_size=kernel_size, patch_size=patch_size, n_classes=3)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reshape to 4D input expected by Conv2d: [B, C, H, W]
    height = 100
    width = X_train.shape[1] // height  # make sure X_train.shape[1] % height == 0
    assert X_train.shape[1] % height == 0, "Input cannot be reshaped evenly, change the height."
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, height, width)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, height, width)
    
    # Repeat channels to simulate RGB (ConvMixer expects multiple channels)
    X_train_tensor = X_train_tensor.repeat(1, 3, 1, 1)
    X_test_tensor = X_test_tensor.repeat(1, 3, 1, 1)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Train the model
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Predict on the test data
    y_pred_test = model(X_test_tensor).detach().numpy()
    return y_pred_test


def cross_validate_convmixer(X, y, kf, dim, depth, kernel_size, patch_size, lr, n_epochs):
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        y_pred = train_convmixer(X_train, y_train, X_test, dim, depth, kernel_size, patch_size, lr, n_epochs)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    return NMSE_by_coordinate(y_true_all, y_pred_all) # returns NMSE, NMSE_x, NMSE_y, NMSE_z


def convmixer_loop(X, y, kf, dims, depths, kernel_sizes, patch_sizes, learning_rates, n_epochs_list):
    errors = []
    model_names = []
    #print("X shape:", X.shape)
    #print("y shape:", y.shape)
    errors = []
    model_names = []
    for dim in dims:
        for depth in depths:
            for kernel_size in kernel_sizes:
                for patch_size in patch_sizes:
                    for lr in learning_rates:
                        for n_epochs in n_epochs_list:
                            nmse, nmse_x, nmse_y, nmse_z = cross_validate_convmixer(
                                X, y, kf, dim, depth, kernel_size, patch_size, lr, n_epochs
                            )

                            model_name = f"ConvMixer (dim={dim}, depth={depth}, kernel={kernel_size}, patch={patch_size}, lr={lr}, epochs={n_epochs})"
                            print(f"{model_name}: NMSE = {nmse}")

                            errors.append([nmse, nmse_x, nmse_y, nmse_z])
                            model_names.append(model_name)
    return errors, model_names
