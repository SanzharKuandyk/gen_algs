# ============================================================
# Energy-Based Models (EBM) with Langevin Dynamics
# - Base: MLP EBM
# - Modified: CNN EBM
# - Dataset: MNIST
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

def langevin_sampling(model, x, steps=20, step_size=0.01, noise_scale=0.01):
    x = x.clone().detach().to(device)
    x.requires_grad_(True)
    for _ in range(steps):
        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        x = x - step_size * grad
        x = x + noise_scale * torch.randn_like(x)
        x = x.detach().requires_grad_(True)
    return x.detach()

class EnergyModelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze()

mlp_model = EnergyModelMLP().to(device)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=1e-4)

epochs = 3
mlp_loss_history = []

print("Training MLP EBM")

for epoch in range(epochs):
    total_loss = 0.0
    for real, _ in loader:
        real = real.to(device)
        fake = torch.randn_like(real)
        fake = langevin_sampling(mlp_model, fake)
        real_energy = mlp_model(real)
        fake_energy = mlp_model(fake)
        loss = real_energy.mean() - fake_energy.mean()
        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()
        total_loss += loss.item()
        mlp_loss_history.append(loss.item())
    print(f"MLP Epoch {epoch+1}/{epochs} Loss {total_loss / len(loader):.4f}")

def generate_samples(model, n=16):
    x = torch.randn(n, 1, 28, 28).to(device)
    x = langevin_sampling(model, x, steps=40)
    return x.cpu()

mlp_samples = generate_samples(mlp_model)

class EnergyModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

cnn_model = EnergyModelCNN().to(device)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)

cnn_loss_history = []

print("Training CNN EBM")

for epoch in range(epochs):
    total_loss = 0.0
    for real, _ in loader:
        real = real.to(device)
        fake = torch.randn_like(real)
        fake = langevin_sampling(cnn_model, fake, steps=30, step_size=0.005)
        real_energy = cnn_model(real)
        fake_energy = cnn_model(fake)
        loss = real_energy.mean() - fake_energy.mean()
        cnn_optimizer.zero_grad()
        loss.backward()
        cnn_optimizer.step()
        total_loss += loss.item()
        cnn_loss_history.append(loss.item())
    print(f"CNN Epoch {epoch+1}/{epochs} Loss {total_loss / len(loader):.4f}")

cnn_samples = generate_samples(cnn_model)

plt.figure(figsize=(6, 4))
plt.plot(mlp_loss_history, label="MLP")
plt.plot(cnn_loss_history, label="CNN")
plt.legend()
plt.title("Loss Comparison")
plt.show()

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(mlp_samples[i][0], cmap="gray")
    plt.axis("off")
plt.suptitle("MLP Samples")
plt.show()

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(cnn_samples[i][0], cmap="gray")
    plt.axis("off")
plt.suptitle("CNN Samples")
plt.show()

print("Done")
