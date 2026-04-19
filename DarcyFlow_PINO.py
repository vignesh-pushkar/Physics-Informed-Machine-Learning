import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Configuration
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 61                     # paper says train/eval on 61x61
TRAIN_SAMPLES = 1000       # paper says 1000 coefficient conditions
TEST_SAMPLES = 200         # test count not explicitly fixed in excerpt; adjust if you want more
EPOCHS = 500               # Darcy uses Burgers PINO params
BATCH_SIZE = 20
LR = 1e-3
MODE = 15                  # Darcy PINO says same params as Burgers; Burgers uses mode=15
WIDTH = 64
FOURIER_LAYERS = 4
PDE_WEIGHT = 1.0
DATA_WEIGHT = 1.0
NUM_WORKERS = 2

CACHE_DIR = Path("./darcy_generated")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
TRAIN_CACHE = CACHE_DIR / f"darcy_train_N{N}_{TRAIN_SAMPLES}_seed{SEED}.npz"
TEST_CACHE  = CACHE_DIR / f"darcy_test_N{N}_{TEST_SAMPLES}_seed{SEED+1}.npz"


# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def sample_gaussian_field_fourier(n, rng, tau=3.0):
    kx = np.fft.fftfreq(n, d=1.0/n)
    ky = np.fft.fftfreq(n, d=1.0/n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    lam = 4.0 * math.pi**2 * (KX**2 + KY**2) + tau**2

    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    coeff = noise / lam
    g = np.fft.ifft2(coeff).real

    g = (g - g.mean()) / (g.std() + 1e-8)
    return g.astype(np.float32)


def threshold_coeff(g):
    return np.where(g >= 0.0, 12.0, 3.0).astype(np.float32)


# Finite-difference Darcy solver on nodal 61x61 grid
def solve_darcy(a):
    n = a.shape[0]
    assert a.shape == (n, n)
    h = 1.0 / (n - 1)
    m = n - 2
    nn = m * m

    rows, cols, vals = [], [], []
    rhs = np.ones(nn, dtype=np.float64)

    def idx(i, j):
        return (i - 1) * m + (j - 1)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            p = idx(i, j)

            ac = a[i, j]
            ae = 0.5 * (ac + a[i + 1, j])
            aw = 0.5 * (ac + a[i - 1, j])
            an = 0.5 * (ac + a[i, j + 1])
            ass = 0.5 * (ac + a[i, j - 1])

            diag = (ae + aw + an + ass) / (h * h)
            rows.append(p); cols.append(p); vals.append(diag)

            if i + 1 <= n - 2:
                rows.append(p); cols.append(idx(i + 1, j)); vals.append(-ae / (h * h))
            if i - 1 >= 1:
                rows.append(p); cols.append(idx(i - 1, j)); vals.append(-aw / (h * h))
            if j + 1 <= n - 2:
                rows.append(p); cols.append(idx(i, j + 1)); vals.append(-an / (h * h))
            if j - 1 >= 1:
                rows.append(p); cols.append(idx(i, j - 1)); vals.append(-ass / (h * h))

    A = sp.csr_matrix((vals, (rows, cols)), shape=(nn, nn))
    u_inner = spla.spsolve(A, rhs)

    u = np.zeros((n, n), dtype=np.float32)
    u[1:-1, 1:-1] = u_inner.reshape(m, m).astype(np.float32)
    return u


# Dataset generation and caching
def generate_dataset(n_samples, n, seed, cache_path):
    if cache_path.exists():
        data = np.load(cache_path)
        return data["a"], data["u"]

    rng = np.random.default_rng(seed)
    A = np.empty((n_samples, n, n), dtype=np.float32)
    U = np.empty((n_samples, n, n), dtype=np.float32)

    t0 = time.time()
    for s in range(n_samples):
        g = sample_gaussian_field_fourier(n, rng, tau=3.0)
        a = threshold_coeff(g)
        u = solve_darcy(a)
        A[s] = a
        U[s] = u

        if (s + 1) % 50 == 0 or s == 0:
            print(f"generated {s+1}/{n_samples} samples | elapsed {(time.time()-t0)/60:.2f} min")

    np.savez_compressed(cache_path, a=A, u=U)
    return A, U


print("Generating/loading training data...")
train_a, train_u = generate_dataset(TRAIN_SAMPLES, N, SEED, TRAIN_CACHE)

print("Generating/loading test data...")
test_a, test_u = generate_dataset(TEST_SAMPLES, N, SEED + 1, TEST_CACHE)


# Torch dataset
class DarcyDataset(Dataset):
    def __init__(self, a, u):
        self.a = torch.from_numpy(a).float().unsqueeze(-1)
        self.u = torch.from_numpy(u).float().unsqueeze(-1)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.u[idx]


train_ds = DarcyDataset(train_a, train_u)
test_ds = DarcyDataset(test_a, test_u)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)


# Grid, mollifier, relative L2
def make_grid(batch, n, device):
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    g = torch.stack([X, Y], dim=-1)
    return g.unsqueeze(0).repeat(batch, 1, 1, 1)

def mollifier(grid):
    x = grid[..., 0]
    y = grid[..., 1]
    return (torch.sin(math.pi * x) * torch.sin(math.pi * y)).unsqueeze(-1)

def rel_l2(pred, target, eps=1e-12):
    num = torch.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return num / den


# Strong-form residual
def darcy_residual(u, a):
    u = u[..., 0]
    a = a[..., 0]

    n = u.shape[1]
    h = 1.0 / (n - 1)

    uc = u[:, 1:-1, 1:-1]
    ue = u[:, 2:,   1:-1]
    uw = u[:, :-2,  1:-1]
    un = u[:, 1:-1, 2:]
    us = u[:, 1:-1, :-2]

    ac = a[:, 1:-1, 1:-1]
    ae = 0.5 * (ac + a[:, 2:,   1:-1])
    aw = 0.5 * (ac + a[:, :-2,  1:-1])
    an = 0.5 * (ac + a[:, 1:-1, 2:])
    ass = 0.5 * (ac + a[:, 1:-1, :-2])

    div_x = (ae * (ue - uc) - aw * (uc - uw)) / (h * h)
    div_y = (an * (un - uc) - ass * (uc - us)) / (h * h)

    return -(div_x + div_y) - 1.0


# FNO / PINO
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        b, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            b, self.out_channels, nx, ny // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weight1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weight2
        )

        return torch.fft.irfft2(out_ft, s=(nx, ny))


class FNO2d(nn.Module):
    def __init__(self, modes=15, width=64, layers=4):
        super().__init__()
        self.width = width
        self.layers = layers

        self.fc0 = nn.Linear(3, width)

        self.specs = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes) for _ in range(layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(layers)
        ])

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        b, nx, ny, _ = x.shape
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for k in range(self.layers):
            x1 = self.specs[k](x)
            x2 = self.ws[k](x.reshape(b, self.width, -1)).reshape(b, self.width, nx, ny)
            x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class DarcyPINO(nn.Module):
    def __init__(self, modes=15, width=64, layers=4):
        super().__init__()
        self.backbone = FNO2d(modes=modes, width=width, layers=layers)

    def forward(self, a, grid):
        x = torch.cat([a, grid], dim=-1)
        out = self.backbone(x)
        return out * mollifier(grid)


model = DarcyPINO(modes=MODE, width=WIDTH, layers=FOURIER_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)


# Training
def evaluate(model, loader):
    model.eval()
    errs = []
    with torch.no_grad():
        for a, u in loader:
            a = a.to(DEVICE)
            u = u.to(DEVICE)
            grid = make_grid(a.shape[0], a.shape[1], DEVICE)
            pred = model(a, grid)
            errs.extend(rel_l2(pred, u).cpu().numpy().tolist())
    return float(np.mean(errs))


best = float("inf")
grid_train = make_grid(BATCH_SIZE, N, DEVICE)

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0.0
    data_sum = 0.0
    pde_sum = 0.0
    nseen = 0

    for a, u in train_loader:
        a = a.to(DEVICE, non_blocking=True)
        u = u.to(DEVICE, non_blocking=True)

        if a.shape[0] != grid_train.shape[0]:
            grid = make_grid(a.shape[0], N, DEVICE)
        else:
            grid = grid_train

        optimizer.zero_grad(set_to_none=True)
        pred = model(a, grid)

        loss_data = torch.mean(rel_l2(pred, u))
        loss_pde = torch.mean(darcy_residual(pred, a) ** 2)
        loss = DATA_WEIGHT * loss_data + PDE_WEIGHT * loss_pde

        loss.backward()
        optimizer.step()

        bs = a.shape[0]
        nseen += bs
        loss_sum += loss.item() * bs
        data_sum += loss_data.item() * bs
        pde_sum += loss_pde.item() * bs

    scheduler.step()

    if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
        test_err = evaluate(model, test_loader)
        elapsed = (time.time() - t0) / 60.0
        print(
            f"epoch {epoch:03d} | "
            f"train={loss_sum/nseen:.6f} | "
            f"data={data_sum/nseen:.6f} | "
            f"pde={pde_sum/nseen:.6f} | "
            f"test_relL2={100.0*test_err:.4f}% | "
            f"time={elapsed:.2f} min"
        )
        if test_err < best:
            best = test_err
            torch.save(model.state_dict(), "darcy_pino_from_scratch.pt")

print(f"best test relative L2 = {100.0*best:.4f}%")


# Reload best model and make figure
model.load_state_dict(torch.load("darcy_pino_from_scratch.pt", map_location=DEVICE))
model.eval()

a0 = test_ds[0][0].unsqueeze(0).to(DEVICE)
u0 = test_ds[0][1].unsqueeze(0).to(DEVICE)
g0 = make_grid(1, N, DEVICE)

with torch.no_grad():
    up = model(a0, g0).cpu().numpy()[0, ..., 0]

a_np = a0.cpu().numpy()[0, ..., 0]
u_np = u0.cpu().numpy()[0, ..., 0]

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

ax[0, 0].imshow(a_np, origin="upper")
ax[0, 0].set_title("input coefficient a", fontsize=14)

ax[0, 1].imshow(a_np, origin="upper")
ax[0, 1].set_title("forward PINO input", fontsize=14)

ax[1, 0].imshow(u_np, origin="upper")
ax[1, 0].set_title("ground truth solution u", fontsize=14)

ax[1, 1].imshow(up, origin="upper")
ax[1, 1].set_title("forward PINO prediction", fontsize=14)

for i in range(2):
    for j in range(2):
        ax[i, j].set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax[i, j].set_yticks([0, 10, 20, 30, 40, 50, 60])

plt.tight_layout()
plt.show()
