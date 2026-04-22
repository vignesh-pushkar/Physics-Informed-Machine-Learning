
# PINNS

# In[1]:
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==========================
# Hyperparameters
# ==========================
batch_size = 32
Nx, Ny = 64, 64
Cin, Cout = 3, 1
width = 64
modes_x, modes_y = 16, 16
num_layers = 4
epochs = 750
learning_rate = 1e-3
lambda_pde = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dx = 1.0 / (Nx - 1)
dy = 1.0 / (Ny - 1)

# ==========================
# Grid
# ==========================
x = torch.linspace(0, 1, Nx)
y = torch.linspace(0, 1, Ny)
X, Y = torch.meshgrid(x, y, indexing="ij")
X = X.to(device)
Y = Y.to(device)

# ==========================
# Spectral Conv
# ==========================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

        self.modes_x = modes_x
        self.modes_y = modes_y

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3]//2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.weights1)

        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# ==========================
# FNO Model
# ==========================
class FNO2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.lift = nn.Conv2d(Cin, width, 1)
        self.activation = nn.GELU()

        self.spec_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes_x, modes_y)
            for _ in range(num_layers)
        ])

        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(num_layers)
        ])

        self.proj = nn.Conv2d(width, Cout, 1)

    def forward(self, x):
        x = self.activation(self.lift(x))

        for spec, w in zip(self.spec_layers, self.ws):
            x = self.activation(spec(x) + w(x))

        return self.proj(x)

# ==========================
# Learnable PDE parameters
# ==========================
k1 = torch.nn.Parameter(torch.tensor(0.5, device=device))
k2 = torch.nn.Parameter(torch.tensor(0.5, device=device))

true_k1 = 1.0
true_k2 = 1.0

# ==========================
# Dataset
# ==========================
N_samples = 512

X_grid = X.unsqueeze(0).repeat(N_samples, 1, 1)
Y_grid = Y.unsqueeze(0).repeat(N_samples, 1, 1)

inputs = torch.stack([
    X_grid,
    Y_grid,
    torch.rand_like(X_grid)
], dim=1)

targets = torch.sin(np.pi * X) * torch.sin(np.pi * Y)
targets = targets.unsqueeze(0).unsqueeze(0).repeat(N_samples, 1, 1, 1)

dataset = torch.utils.data.TensorDataset(inputs, targets)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================
# Finite Difference Laplacian
# ==========================
def laplacian(u):
    u_xx = (u[:, :, 2:, 1:-1] - 2*u[:, :, 1:-1, 1:-1] + u[:, :, :-2, 1:-1]) / dx**2
    u_yy = (u[:, :, 1:-1, 2:] - 2*u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2]) / dy**2
    return u_xx, u_yy

# ==========================
# PDE Residual
# ==========================
def pde_residual(u):
    u_xx, u_yy = laplacian(u)

    forcing = -2 * (np.pi ** 2) * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    forcing = forcing[1:-1, 1:-1]  # match interior
    forcing = forcing.unsqueeze(0).unsqueeze(0).to(device)

    return k1 * u_xx + k2 * u_yy - forcing

# ==========================
# Model + Optimizer
# ==========================
model = FNO2D().to(device)

optimizer = optim.Adam(
    list(model.parameters()) + [k1, k2],
    lr=learning_rate
)

scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = nn.MSELoss()

# ==========================
# Training
# ==========================
losses = []
k1_errors = []
k2_errors = []

print("Training PINO Inverse Problem...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        pred = model(batch_x)

        loss_data = criterion(pred, batch_y)

        residual = pde_residual(pred)
        loss_pde = torch.mean(residual ** 2)

        loss = loss_data + lambda_pde * loss_pde
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    epoch_loss /= len(loader)
    losses.append(epoch_loss)

    k1_errors.append(abs(k1.item() - true_k1))
    k2_errors.append(abs(k2.item() - true_k2))

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss {epoch_loss:.6f} | k1 {k1.item():.4f} | k2 {k2.item():.4f}")

# ==========================
# Plots
# ==========================
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(k1_errors)
plt.title("k1 Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

plt.figure()
plt.plot(k2_errors)
plt.title("k2 Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

print(f"Learned k1: {k1.item():.4f}, True: {true_k1}")
print(f"Learned k2: {k2.item():.4f}, True: {true_k2}")

# In[2]:
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==========================
# Hyperparameters
# ==========================
Nx, Ny = 64, 64
epochs = 15000
lr = 1e-3
lambda_pde = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Grid
# ==========================
x = torch.linspace(0, 1, Nx)
y = torch.linspace(0, 1, Ny)
X, Y = torch.meshgrid(x, y, indexing="ij")

X = X.reshape(-1, 1).to(device)
Y = Y.reshape(-1, 1).to(device)

XY = torch.cat([X, Y], dim=1)

# Exact solution
u_exact = torch.sin(np.pi * X) * torch.sin(np.pi * Y)

# ==========================
# PINN Model
# ==========================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN().to(device)

# ==========================
# Learnable PDE parameters
# ==========================
k1 = torch.nn.Parameter(torch.tensor(0.5, device=device))
k2 = torch.nn.Parameter(torch.tensor(0.5, device=device))

true_k1 = 1.0
true_k2 = 1.0

# ==========================
# PDE Residual (autograd)
# ==========================
def pde_residual(model, x, y):
    x.requires_grad_(True)
    y.requires_grad_(True)

    input = torch.cat([x, y], dim=1)
    u = model(input)

    # First derivatives
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

    forcing = -2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

    return k1 * u_xx + k2 * u_yy - forcing

# ==========================
# Optimizer
# ==========================
optimizer = torch.optim.Adam(
    list(model.parameters()) + [k1, k2],
    lr=lr
)

# ==========================
# Training
# ==========================
losses = []
k1_errors = []
k2_errors = []

print("Training PINN Inverse Problem...")

for epoch in range(epochs):
    optimizer.zero_grad()

    pred = model(XY)

    # Data loss
    loss_data = torch.mean((pred - u_exact)**2)

    # PDE loss
    res = pde_residual(model, X, Y)
    loss_pde = torch.mean(res**2)

    loss = loss_data + lambda_pde * loss_pde

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    k1_errors.append(abs(k1.item() - true_k1))
    k2_errors.append(abs(k2.item() - true_k2))

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f} | k1 {k1.item():.4f} | k2 {k2.item():.4f}")

# ==========================
# Plots (same style as PINO)
# ==========================
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(k1_errors)
plt.title("k1 Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

plt.figure()
plt.plot(k2_errors)
plt.title("k2 Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

print(f"Learned k1: {k1.item():.4f}, True: {true_k1}")
print(f"Learned k2: {k2.item():.4f}, True: {true_k2}")