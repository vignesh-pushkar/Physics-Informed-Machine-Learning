import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# Reproducibility and device

SEED = 25619
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Model and training parameters 

NX = 64
NY = 64

K_MODES = 4

TRAIN_N = 512
VAL_N = 64
TEST_N = 64

BATCH_SIZE = 16
EPOCHS = 500
LR = 2e-3
WEIGHT_DECAY = 1e-6

lambda_weak = 1.0
lambda_data = 1.0
lambda_jump = 0.5

P_MAX = 3
Q_MAX = 3

LOG_EVERY = 25

# Grid

x = torch.linspace(0.0, 1.0, NX)
y = torch.linspace(0.0, 1.0, NY)
X, Y = torch.meshgrid(x, y, indexing="ij")

dx = x[1] - x[0]
dy = y[1] - y[0]

iface_idx = torch.argmin(torch.abs(x - 0.5)).item()
x_iface = x[iface_idx].item()

print("Interface x-grid location:", x_iface, "index:", iface_idx)

# Manufactured operator family
# tutorial case is recovered when a = 0

def gD0(X):
    return torch.where(X <= 0.5, X**2, (X - 1.0)**2)

def s_of_x(X):
    return torch.where(X <= 0.5, X, 1.0 - X)

def q_of_y(Y, a):
    B = a.shape[0]
    out = torch.zeros(B, Y.shape[0], Y.shape[1], device=Y.device)
    for k in range(1, K_MODES + 1):
        out = out + a[:, k - 1].view(B, 1, 1) * torch.sin(k * math.pi * Y).unsqueeze(0)
    return out

def qyy_of_y(Y, a):
    B = a.shape[0]
    out = torch.zeros(B, Y.shape[0], Y.shape[1], device=Y.device)
    for k in range(1, K_MODES + 1):
        out = out - ((k * math.pi) ** 2) * a[:, k - 1].view(B, 1, 1) * torch.sin(k * math.pi * Y).unsqueeze(0)
    return out

def exact_solution(a, Xd, Yd):
    B = a.shape[0]
    GD = gD0(Xd).unsqueeze(0).repeat(B, 1, 1)
    S = s_of_x(Xd).unsqueeze(0).repeat(B, 1, 1)
    Q = q_of_y(Yd, a)
    return GD + S * Q

def forcing_field(a, Xd, Yd):
    B = a.shape[0]
    S = s_of_x(Xd).unsqueeze(0).repeat(B, 1, 1)
    QYY = qyy_of_y(Yd, a)
    return -2.0 - S * QYY

def interface_jump(a, ygrid):
    B = a.shape[0]
    yy = ygrid.view(1, -1)
    out = torch.zeros(B, ygrid.shape[0], device=a.device)
    for k in range(1, K_MODES + 1):
        out = out + a[:, k - 1].view(B, 1) * torch.sin(k * math.pi * yy)
    return 2.0 + 2.0 * out

def tutorial_case_coeffs(device_="cpu"):
    return torch.zeros(1, K_MODES, device=device_)

def sample_a(n, device_):
    coeffs = torch.empty(n, K_MODES, device=device_)
    coeffs[:, 0] = torch.empty(n, device=device_).uniform_(-0.40, 0.40)
    coeffs[:, 1] = torch.empty(n, device=device_).uniform_(-0.25, 0.25)
    coeffs[:, 2] = torch.empty(n, device=device_).uniform_(-0.15, 0.15)
    coeffs[:, 3] = torch.empty(n, device=device_).uniform_(-0.10, 0.10)
    return coeffs

# Input construction

def build_input(a, Xd, Yd):
    B = a.shape[0]

    q = q_of_y(Yd, a)
    signed = (Xd - 0.5).unsqueeze(0).repeat(B, 1, 1)
    xx = Xd.unsqueeze(0).repeat(B, 1, 1)
    yy = Yd.unsqueeze(0).repeat(B, 1, 1)
    left_mask = (Xd <= 0.5).float().unsqueeze(0).repeat(B, 1, 1)
    iface_feat = torch.exp(-35.0 * torch.abs(Xd - 0.5)).unsqueeze(0).repeat(B, 1, 1)

    return torch.stack([q, signed, xx, yy, left_mask, iface_feat], dim=1)

# Dataset generation on CPU

def make_dataset(n):
    a = sample_a(n, device_="cpu")
    inp = build_input(a, X, Y)
    u = exact_solution(a, X, Y)
    return a, inp, u

a_train, inp_train, u_train = make_dataset(TRAIN_N)
a_val, inp_val, u_val = make_dataset(VAL_N)
a_test, inp_test, u_test = make_dataset(TEST_N)

train_loader = DataLoader(
    TensorDataset(a_train, inp_train, u_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

print("Train/Val/Test sizes:", TRAIN_N, VAL_N, TEST_N)

# FNO blocks

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spec(x) + self.w(x))

class PINOInterface(nn.Module):
    def __init__(self, in_channels=6, width=48, modes1=16, modes2=16, depth=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(depth)])
        self.proj1 = nn.Conv2d(width, width, 1)
        self.proj2 = nn.Conv2d(width, 1, 1)

    def forward(self, inp):
        z = self.lift(inp)
        for blk in self.blocks:
            z = blk(z)
        z = F.gelu(self.proj1(z))
        raw = self.proj2(z).squeeze(1)

        Xd = X.to(inp.device)
        Yd = Y.to(inp.device)

        gd = gD0(Xd).unsqueeze(0).repeat(inp.shape[0], 1, 1)
        boundary_mask = (Xd * (1.0 - Xd) * Yd * (1.0 - Yd)).unsqueeze(0).repeat(inp.shape[0], 1, 1)

        return gd + boundary_mask * raw

# Derivatives

def grad_x(u, dx_):
    ux = torch.zeros_like(u)
    ux[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dx_)
    ux[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dx_
    ux[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dx_
    return ux

def grad_y(u, dy_):
    uy = torch.zeros_like(u)
    uy[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dy_)
    uy[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dy_
    uy[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dy_
    return uy

# Weak test functions, precomputed on CPU

def build_test_functions():
    V = []
    VX = []
    VY = []
    VIF = []

    for p in range(1, P_MAX + 1):
        for q in range(1, Q_MAX + 1):
            v = torch.sin(p * math.pi * X) * torch.sin(q * math.pi * Y)
            vx = p * math.pi * torch.cos(p * math.pi * X) * torch.sin(q * math.pi * Y)
            vy = q * math.pi * torch.sin(p * math.pi * X) * torch.cos(q * math.pi * Y)
            vif = torch.sin(p * math.pi * torch.tensor(x_iface)) * torch.sin(q * math.pi * y)

            V.append(v)
            VX.append(vx)
            VY.append(vy)
            VIF.append(vif)

    V = torch.stack(V, dim=0)
    VX = torch.stack(VX, dim=0)
    VY = torch.stack(VY, dim=0)
    VIF = torch.stack(VIF, dim=0)
    return V, VX, VY, VIF

V_cpu, VX_cpu, VY_cpu, VIF_cpu = build_test_functions()
NUM_TEST_FUNCS = V_cpu.shape[0]
print("Number of weak test functions:", NUM_TEST_FUNCS)

# Losses and metrics

def weak_loss(u_pred, a):
    Xd = X.to(u_pred.device)
    Yd = Y.to(u_pred.device)
    yd = y.to(u_pred.device)

    V = V_cpu.to(u_pred.device)
    VX = VX_cpu.to(u_pred.device)
    VY = VY_cpu.to(u_pred.device)
    VIF = VIF_cpu.to(u_pred.device)

    dx_ = dx.to(u_pred.device)
    dy_ = dy.to(u_pred.device)

    ux = grad_x(u_pred, dx_)
    uy = grad_y(u_pred, dy_)
    f = forcing_field(a, Xd, Yd)
    gI = interface_jump(a, yd)

    bulk = (
        ux.unsqueeze(1) * VX.unsqueeze(0) +
        uy.unsqueeze(1) * VY.unsqueeze(0) -
        f.unsqueeze(1) * V.unsqueeze(0)
    ).sum(dim=(2, 3)) * dx_ * dy_

    iface = (
        gI.unsqueeze(1) * VIF.unsqueeze(0)
    ).sum(dim=2) * dy_

    res = bulk - iface
    return torch.mean(res ** 2)

def jump_loss(u_pred, a):
    dx_ = dx.to(u_pred.device)
    yd = y.to(u_pred.device)

    ux = grad_x(u_pred, dx_)
    jump_pred = ux[:, iface_idx - 1, :] - ux[:, iface_idx + 1, :]
    jump_true = interface_jump(a, yd)
    return torch.mean((jump_pred - jump_true) ** 2)

def data_loss(u_pred, u_true):
    return F.mse_loss(u_pred, u_true)

def relative_l2(u_pred, u_true):
    num = torch.norm((u_pred - u_true).reshape(u_pred.shape[0], -1), dim=1)
    den = torch.norm(u_true.reshape(u_true.shape[0], -1), dim=1)
    return torch.mean(num / den)

# Evaluation

@torch.no_grad()
def evaluate_full(model, a_eval, inp_eval, u_eval, batch_size=32):
    model.eval()

    n = a_eval.shape[0]
    total_rel = 0.0
    total_data = 0.0
    total_weak = 0.0
    total_jump = 0.0
    total_count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        a_b = a_eval[start:end].to(device, non_blocking=True)
        inp_b = inp_eval[start:end].to(device, non_blocking=True)
        u_b = u_eval[start:end].to(device, non_blocking=True)

        pred = model(inp_b)

        bsz = end - start
        total_rel += relative_l2(pred, u_b).item() * bsz
        total_data += data_loss(pred, u_b).item() * bsz
        total_weak += weak_loss(pred, a_b).item() * bsz
        total_jump += jump_loss(pred, a_b).item() * bsz
        total_count += bsz

    return {
        "rel_l2": total_rel / total_count,
        "data": total_data / total_count,
        "weak": total_weak / total_count,
        "jump": total_jump / total_count,
    }

# Model and optimizer

model = PINOInterface().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=180, gamma=0.5)

# Training

history = {
    "epoch": [],
    "train_total": [],
    "train_weak": [],
    "train_data": [],
    "train_jump": [],
    "val_rel_l2": [],
    "val_data": [],
    "val_weak": [],
    "val_jump": [],
}

best_val_rel = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_total = 0.0
    running_weak = 0.0
    running_data = 0.0
    running_jump = 0.0
    seen = 0

    for a_b_cpu, inp_b_cpu, u_b_cpu in train_loader:
        a_b = a_b_cpu.to(device, non_blocking=True)
        inp_b = inp_b_cpu.to(device, non_blocking=True)
        u_b = u_b_cpu.to(device, non_blocking=True)

        pred = model(inp_b)

        lw = weak_loss(pred, a_b)
        ld = data_loss(pred, u_b)
        lj = jump_loss(pred, a_b)

        loss = lambda_weak * lw + lambda_data * ld + lambda_jump * lj

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = a_b.shape[0]
        running_total += loss.item() * bsz
        running_weak += lw.item() * bsz
        running_data += ld.item() * bsz
        running_jump += lj.item() * bsz
        seen += bsz

    scheduler.step()

    if epoch % LOG_EVERY == 0 or epoch == 1:
        train_total = running_total / seen
        train_weak = running_weak / seen
        train_data = running_data / seen
        train_jump = running_jump / seen

        val_metrics = evaluate_full(model, a_val, inp_val, u_val, batch_size=32)

        history["epoch"].append(epoch)
        history["train_total"].append(train_total)
        history["train_weak"].append(train_weak)
        history["train_data"].append(train_data)
        history["train_jump"].append(train_jump)
        history["val_rel_l2"].append(val_metrics["rel_l2"])
        history["val_data"].append(val_metrics["data"])
        history["val_weak"].append(val_metrics["weak"])
        history["val_jump"].append(val_metrics["jump"])

        print(
            f"epoch={epoch:4d} "
            f"train_total={train_total:.4e} "
            f"train_weak={train_weak:.4e} "
            f"train_data={train_data:.4e} "
            f"train_jump={train_jump:.4e} "
            f"val_relL2={val_metrics['rel_l2']:.4e} "
            f"val_data={val_metrics['data']:.4e} "
            f"val_weak={val_metrics['weak']:.4e} "
            f"val_jump={val_metrics['jump']:.4e}"
        )

        if val_metrics["rel_l2"] < best_val_rel:
            best_val_rel = val_metrics["rel_l2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

print("\nBest fixed-val relative L2:", best_val_rel)

test_metrics = evaluate_full(model, a_test, inp_test, u_test, batch_size=32)
print("Family test metrics:", test_metrics)

# Plotting

def plot_history(history):
    ep = np.array(history["epoch"])

    plt.figure(figsize=(7, 5))
    plt.plot(ep, history["train_total"], label="train total")
    plt.plot(ep, history["train_weak"], label="train weak")
    plt.plot(ep, history["train_data"], label="train data")
    plt.plot(ep, history["train_jump"], label="train jump")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training losses")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(ep, history["val_rel_l2"], label="fixed validation relative L2")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2")
    plt.title("Fixed validation relative L2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(ep, history["val_data"], label="val data")
    plt.plot(ep, history["val_weak"], label="val weak")
    plt.plot(ep, history["val_jump"], label="val jump")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Fixed validation losses")
    plt.legend()
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def get_prediction_pair(model, a_sample):
    model.eval()
    a_sample = a_sample.to(device)
    inp = build_input(a_sample, X.to(device), Y.to(device))
    pred = model(inp)[0].detach().cpu()
    true = exact_solution(a_sample, X.to(device), Y.to(device))[0].detach().cpu()
    return pred, true

def show_field_comparison(model, a_sample, title_suffix=""):
    pred, true = get_prediction_pair(model, a_sample)

    pred_np = pred.numpy()
    true_np = true.numpy()
    diff_np = pred_np - true_np
    abs_np = np.abs(diff_np)

    extent = [0, 1, 0, 1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    im0 = axes[0].imshow(true_np.T, origin="lower", extent=extent, aspect="auto")
    axes[0].set_title(f"Ground truth{title_suffix}")
    axes[0].axvline(0.5, linestyle="--")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_np.T, origin="lower", extent=extent, aspect="auto")
    axes[1].set_title(f"Prediction{title_suffix}")
    axes[1].axvline(0.5, linestyle="--")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff_np.T, origin="lower", extent=extent, aspect="auto")
    axes[2].set_title(f"Signed error{title_suffix}")
    axes[2].axvline(0.5, linestyle="--")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(abs_np.T, origin="lower", extent=extent, aspect="auto")
    axes[3].set_title(f"Absolute error{title_suffix}")
    axes[3].axvline(0.5, linestyle="--")
    plt.colorbar(im3, ax=axes[3])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()

def splice_plots(model, a_sample, y_values=(0.25, 0.5, 0.75), x_values=(0.25, 0.5, 0.75), title_suffix=""):
    pred, true = get_prediction_pair(model, a_sample)

    x_cpu = x
    y_cpu = y

    plt.figure(figsize=(7, 5))
    for y0 in y_values:
        j = torch.argmin(torch.abs(y_cpu - y0)).item()
        plt.plot(x_cpu.numpy(), true[:, j].numpy(), label=f"true y={float(y_cpu[j]):.3f}")
        plt.plot(x_cpu.numpy(), pred[:, j].numpy(), linestyle="--", label=f"pred y={float(y_cpu[j]):.3f}")
    plt.axvline(0.5, linestyle=":")
    plt.xlabel("x")
    plt.ylabel("u(x,y0)")
    plt.title(f"Horizontal splices{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    for x0 in x_values:
        i = torch.argmin(torch.abs(x_cpu - x0)).item()
        plt.plot(y_cpu.numpy(), true[i, :].numpy(), label=f"true x={float(x_cpu[i]):.3f}")
        plt.plot(y_cpu.numpy(), pred[i, :].numpy(), linestyle="--", label=f"pred x={float(x_cpu[i]):.3f}")
    plt.xlabel("y")
    plt.ylabel("u(x0,y)")
    plt.title(f"Vertical splices{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def interface_jump_plot(model, a_sample, title_suffix=""):
    model.eval()
    a_sample = a_sample.to(device)
    u = model(build_input(a_sample, X.to(device), Y.to(device)))
    ux = grad_x(u, dx.to(device))[0]

    jump_pred = (ux[iface_idx - 1, :] - ux[iface_idx + 1, :]).detach().cpu()
    jump_true = interface_jump(a_sample, y.to(device))[0].detach().cpu()

    plt.figure(figsize=(7, 5))
    plt.plot(y.numpy(), jump_true.numpy(), label="true jump")
    plt.plot(y.numpy(), jump_pred.numpy(), linestyle="--", label="predicted jump")
    plt.xlabel("y")
    plt.ylabel("jump")
    plt.title(f"Interface jump diagnostic{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def report_case_metrics(model, a_sample, case_name="case"):
    model.eval()
    a_sample = a_sample.to(device)
    inp = build_input(a_sample, X.to(device), Y.to(device))
    pred = model(inp)
    true = exact_solution(a_sample, X.to(device), Y.to(device))

    rel = relative_l2(pred, true).item()
    mse = data_loss(pred, true).item()
    weak = weak_loss(pred, a_sample).item()
    jmp = jump_loss(pred, a_sample).item()

    print(f"\nMetrics for {case_name}")
    print("relative L2:", rel)
    print("MSE:", mse)
    print("weak loss:", weak)
    print("jump loss:", jmp)

# Final diagnostics
# 1) family test sample
# 2) exact tutorial case

plot_history(history)

family_case = a_test[0:1]
report_case_metrics(model, family_case, case_name="random family test case")
show_field_comparison(model, family_case, title_suffix=" (family case)")
splice_plots(model, family_case, title_suffix=" (family case)")
interface_jump_plot(model, family_case, title_suffix=" (family case)")

tutorial_case = tutorial_case_coeffs(device_="cpu")
report_case_metrics(model, tutorial_case, case_name="exact tutorial case")
show_field_comparison(model, tutorial_case, title_suffix=" (tutorial case)")
splice_plots(model, tutorial_case, title_suffix=" (tutorial case)")
interface_jump_plot(model, tutorial_case, title_suffix=" (tutorial case)")
