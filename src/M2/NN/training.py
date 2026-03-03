from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from M2.NN.model import Neural_Network
from M2.Utils.conditions import apply_dirichlet

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