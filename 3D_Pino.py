"""
PINO for Conjugate Heat Transfer (CHT) - 3D Parametric Dataset
===============================================================
Trains a Physics-Informed Neural Operator (PINO) to map design
parameters and geometry to steady-state flow and temperature fields
over a finned heat sink.

Physics residuals enforced via finite differences:
  Fluid  : continuity  (div u = 0)
           fluid energy (u.grad(theta_f) - alpha_f * lap(theta_f) = 0)
  Solid  : Laplace     (lap(theta_s) = 0)
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

torch.manual_seed(42)
np.random.seed(42)


#  CLI + CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--data",        default="/kaggle/input/datasets/subhommahalik/dataset-combined-npz/dataset_combined.npz")
parser.add_argument("--out",         default="pino_cht_output")
parser.add_argument("--epochs",      type=int,   default=600)
parser.add_argument("--batch",       type=int,   default=4)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--lambda_max",  type=float, default=0.05,
                    help="Maximum physics loss weight")
parser.add_argument("--phys_warmup", type=int,   default=60,
                    help="Epochs with physics loss disabled")
parser.add_argument("--phys_ramp",   type=int,   default=120,
                    help="Epochs to ramp lambda: 0 -> lambda_max")
args, _ = parser.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

os.makedirs(args.out, exist_ok=True)

N_IN      = 10
N_OUT     = 6
N_MODES_X = 12
N_MODES_Y = 6
N_MODES_Z = 6
WIDTH     = 48
N_LAYERS  = 4

FIELD_NAMES   = ["u", "v", "w", "p", "theta_f", "theta_s"]
FIELD_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.5, 8.0])
ALPHA_F       = 0.02


#  DATASET
class CHTDataset(Dataset):
    """
    Parametric CHT dataset with domain-aware normalisation, Gaussian
    smoothing, and optional z-flip augmentation.
    """

    def __init__(self, data_path, indices, stats=None,
                 augment=False, smooth_sigma=0.7):
        raw        = np.load(data_path, allow_pickle=True)
        params_raw = raw["params"][indices].astype(np.float32)
        fields_raw = raw["fields"][indices].astype(np.float32)
        masks_raw  = raw["geo_masks"][indices].astype(np.float32)
        self.grid_x = raw["grid_x"].astype(np.float32)
        self.grid_y = raw["grid_y"].astype(np.float32)
        self.grid_z = raw["grid_z"].astype(np.float32)

        print(f"  Smoothing {len(params_raw)} samples (sigma={smooth_sigma}) ...", flush=True)
        for n in range(len(fields_raw)):
            fluid_m = (masks_raw[n] < 0.5)
            solid_m = ~fluid_m
            for c in range(6):
                f   = fields_raw[n, c]
                tmp = gaussian_filter(f, sigma=smooth_sigma)
                if c < 5:
                    tmp[solid_m] = 0.0
                else:
                    tmp[fluid_m] = 0.0
                fields_raw[n, c] = tmp

        if augment:
            params_flip       = params_raw.copy()
            fields_flip       = fields_raw[:, :, :, :, ::-1].copy()
            fields_flip[:, 2] *= -1.0
            masks_flip        = masks_raw[:, :, :, ::-1].copy()
            self.params = np.concatenate([params_raw, params_flip], axis=0)
            self.fields = np.concatenate([fields_raw, fields_flip], axis=0)
            self.masks  = np.concatenate([masks_raw,  masks_flip],  axis=0)
            print(f"  Z-flip augmentation: {len(params_raw)} -> {len(self.params)} samples")
        else:
            self.params = params_raw
            self.fields = fields_raw
            self.masks  = masks_raw

        N, C, NX, NY, NZ = self.fields.shape
        self.NX, self.NY, self.NZ = NX, NY, NZ
        self.dx = float(self.grid_x[1] - self.grid_x[0]) if NX > 1 else 1.0
        self.dy = float(self.grid_y[1] - self.grid_y[0]) if NY > 1 else 1.0
        self.dz = float(self.grid_z[1] - self.grid_z[0]) if NZ > 1 else 1.0

        if stats is None:
            self.p_min = self.params.min(axis=0)
            self.p_max = self.params.max(axis=0)

            fluid_mask = 1.0 - self.masks
            solid_mask = self.masks
            f_mean = np.zeros(C, dtype=np.float32)
            f_std  = np.ones(C,  dtype=np.float32)

            for i in range(C):
                field_i = self.fields[:, i, :, :, :]
                vals    = field_i[fluid_mask > 0.5] if i < 5 else field_i[solid_mask > 0.5]
                if len(vals) > 10:
                    f_mean[i] = vals.mean()
                    f_std[i]  = max(vals.std(), 1e-6)

            self.f_mean = f_mean
            self.f_std  = f_std
        else:
            self.p_min  = stats["p_min"]
            self.p_max  = stats["p_max"]
            self.f_mean = stats["f_mean"]
            self.f_std  = stats["f_std"]

    def get_stats(self):
        return dict(p_min=self.p_min, p_max=self.p_max,
                    f_mean=self.f_mean, f_std=self.f_std)

    def get_spacings(self):
        return self.dx, self.dy, self.dz

    def _norm_params(self, p):
        return (p - self.p_min) / (self.p_max - self.p_min + 1e-8)

    def _norm_fields(self, f):
        return (f - self.f_mean[:, None, None, None]) / self.f_std[:, None, None, None]

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        p  = self._norm_params(self.params[idx])
        f  = self._norm_fields(self.fields[idx])
        m  = self.masks[idx]
        NX, NY, NZ = self.NX, self.NY, self.NZ

        xs     = torch.linspace(0, 1, NX)[:, None, None].expand(NX, NY, NZ)
        ys     = torch.linspace(0, 1, NY)[None, :, None].expand(NX, NY, NZ)
        zs     = torch.linspace(0, 1, NZ)[None, None, :].expand(NX, NY, NZ)
        coords = torch.stack([xs, ys, zs], dim=0)

        p_grid = torch.tensor(p)[:, None, None, None].expand(6, NX, NY, NZ)
        m_grid = torch.tensor(m[None])
        x_in   = torch.cat([p_grid, m_grid, coords], dim=0).float()
        y_out  = torch.tensor(f).float()
        return x_in, y_out, torch.tensor(m).float()



#  FNO ARCHITECTURE
class SpectralConv3d(nn.Module):
    """3-D spectral convolution retaining the lowest Fourier modes."""

    def __init__(self, in_ch, out_ch, mx, my, mz):
        super().__init__()
        self.out_ch = out_ch
        self.mx, self.my, self.mz = mx, my, mz
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))

    @staticmethod
    def cmul(x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x):
        B, C, NX, NY, NZ = x.shape
        mx, my, mz = self.mx, self.my, self.mz
        x_ft  = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        NZH   = NZ // 2 + 1
        out_ft = torch.zeros(B, self.out_ch, NX, NY, NZH,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx,  :my,  :mz] = self.cmul(x_ft[:, :, :mx,  :my,  :mz],  self.w1)
        out_ft[:, :, -mx:, :my,  :mz] = self.cmul(x_ft[:, :, -mx:, :my,  :mz],  self.w2)
        out_ft[:, :, :mx,  -my:, :mz] = self.cmul(x_ft[:, :, :mx,  -my:, :mz],  self.w3)
        out_ft[:, :, -mx:, -my:, :mz] = self.cmul(x_ft[:, :, -mx:, -my:, :mz],  self.w4)
        return torch.fft.irfftn(out_ft, s=(NX, NY, NZ), dim=(-3, -2, -1)).to(x.dtype)


class FNOLayer3D(nn.Module):
    """Single FNO layer: spectral conv + skip + InstanceNorm + Dropout."""

    def __init__(self, width, mx, my, mz, dropout=0.10):
        super().__init__()
        self.spec = SpectralConv3d(width, width, mx, my, mz)
        self.skip = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width, affine=True)
        self.act  = nn.GELU()
        self.drop = nn.Dropout3d(p=dropout)

    def forward(self, x):
        return self.drop(self.norm(self.act(self.spec(x) + self.skip(x))))


class FNO3D(nn.Module):
    """3-D Fourier Neural Operator with lifting and projection layers."""

    def __init__(self, n_in=N_IN, n_out=N_OUT, width=WIDTH,
                 n_layers=N_LAYERS, mx=N_MODES_X, my=N_MODES_Y, mz=N_MODES_Z):
        super().__init__()
        self.lift   = nn.Conv3d(n_in, width, 1)
        self.layers = nn.ModuleList([FNOLayer3D(width, mx, my, mz)
                                     for _ in range(n_layers)])
        self.proj   = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, n_out, 1),
        )

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        return self.proj(x)


#  FINITE DIFFERENCE UTILITIES
def fd_grad(f, spacing, dim):
    """Central finite differences (2nd order) along `dim`."""
    return torch.gradient(f, spacing=(spacing,), dim=(dim,))[0]


def fd_laplacian(f, dx, dy, dz):
    """Laplacian via second-order central finite differences."""
    return (fd_grad(fd_grad(f, dx, 1), dx, 1) +
            fd_grad(fd_grad(f, dy, 2), dy, 2) +
            fd_grad(fd_grad(f, dz, 3), dz, 3))



#  PHYSICS RESIDUALS
def physics_residuals(pred_norm, geo_mask, stats, dx, dy, dz, alpha_f=ALPHA_F):
    """
    Compute three PDE residuals on predicted (normalised) fields.

    Returns:
        loss_cont  : continuity residual (fluid)
        loss_fe    : fluid energy residual (fluid)
        loss_sl    : solid Laplace residual (solid)
    """
    f_mean = torch.tensor(stats["f_mean"], device=pred_norm.device, dtype=pred_norm.dtype)
    f_std  = torch.tensor(stats["f_std"],  device=pred_norm.device, dtype=pred_norm.dtype)
    pred_phys = pred_norm * f_std[None, :, None, None, None] \
              + f_mean[None, :, None, None, None]

    u    = pred_phys[:, 0]
    v    = pred_phys[:, 1]
    w    = pred_phys[:, 2]
    th_f = pred_phys[:, 4]
    th_s = pred_phys[:, 5]

    fluid = 1.0 - geo_mask
    solid = geo_mask

    div       = fd_grad(u, dx, 1) + fd_grad(v, dy, 2) + fd_grad(w, dz, 3)
    loss_cont = (div ** 2 * fluid).sum() / (fluid.sum() + 1e-8)

    adv      = (u * fd_grad(th_f, dx, 1) +
                v * fd_grad(th_f, dy, 2) +
                w * fd_grad(th_f, dz, 3))
    diff     = alpha_f * fd_laplacian(th_f, dx, dy, dz)
    loss_fe  = ((adv - diff) ** 2 * fluid).sum() / (fluid.sum() + 1e-8)

    lap_s   = fd_laplacian(th_s, dx, dy, dz)
    loss_sl = (lap_s ** 2 * solid).sum() / (solid.sum() + 1e-8)

    return loss_cont, loss_fe, loss_sl



#  DATA LOSS AND METRICS
def masked_mse(pred, target, mask):
    m    = (mask > 0.5).float()
    diff = (pred - target) ** 2
    return (diff * m).sum() / (m.sum() + 1e-8)


def data_loss(pred, target, geo_mask, weights=FIELD_WEIGHTS):
    """Per-field weighted masked MSE (fluid fields / solid theta_s)."""
    fluid = 1.0 - geo_mask
    solid = geo_mask
    w     = weights.to(pred.device)
    loss  = torch.tensor(0.0, device=pred.device)
    for i in range(4):
        loss = loss + w[i] * masked_mse(pred[:, i], target[:, i], fluid)
    loss = loss + w[4] * masked_mse(pred[:, 4], target[:, 4], fluid)
    loss = loss + w[5] * masked_mse(pred[:, 5], target[:, 5], solid)
    return loss


def compute_rel_errors(pred, target, geo_mask):
    """Relative L2 error (%) per field, masked to the correct domain."""
    fluid = (geo_mask < 0.5)
    solid = (geo_mask > 0.5)
    domain_masks = {
        "u": fluid, "v": fluid, "w": fluid, "p": fluid,
        "theta_f": fluid, "theta_s": solid,
    }
    errors = {n: [] for n in FIELD_NAMES}
    for i, name in enumerate(FIELD_NAMES):
        dm = domain_masks[name]
        for b in range(pred.shape[0]):
            p_vals = pred[b, i][dm[b]]
            t_vals = target[b, i][dm[b]]
            if len(t_vals) == 0:
                continue
            num = torch.norm(p_vals - t_vals)
            den = torch.norm(t_vals) + 1e-8
            errors[name].append((num / den * 100).item())
    return errors


#  PHYSICS WEIGHT SCHEDULE
def phys_lambda(epoch, warmup, ramp, lam_max):
    """Linear ramp from 0 to lam_max after a warm-up period."""
    if epoch <= warmup:
        return 0.0
    t = min(epoch - warmup, ramp) / ramp
    return lam_max * t


#  MAIN
def main():
    print(f"\nLoading {args.data} ...")
    raw = np.load(args.data, allow_pickle=True)
    N   = len(raw["params"])
    print(f"  Total samples : {N}")

    idx     = np.random.permutation(N)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    n_test  = N - n_train - n_val
    i_train = idx[:n_train]
    i_val   = idx[n_train:n_train + n_val]
    i_test  = idx[n_train + n_val:]
    print(f"  Train/Val/Test: {n_train}/{n_val}/{n_test}")

    train_ds   = CHTDataset(args.data, i_train, augment=True, smooth_sigma=0.7)
    stats      = train_ds.get_stats()
    dx, dy, dz = train_ds.get_spacings()

    print("\n  Field normalisation stats (domain-aware):")
    for i, name in enumerate(FIELD_NAMES):
        print(f"    {name:<10}  mean={stats['f_mean'][i]:+.4f}  std={stats['f_std'][i]:.4f}")

    val_ds  = CHTDataset(args.data, i_val,  stats, augment=False, smooth_sigma=0.7)
    test_ds = CHTDataset(args.data, i_test, stats, augment=False, smooth_sigma=0.7)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"\n  Grid spacings : dx={dx:.4f}  dy={dy:.4f}  dz={dz:.4f}")
    print(f"  Phys warmup   : {args.phys_warmup} ep  |  ramp: {args.phys_ramp} ep  "
          f"|  lambda_max: {args.lambda_max}")
    print(f"  Batches/epoch : {len(train_loader)} train, {len(val_loader)} val")

    INPUT_NOISE_STD = 0.02

    model = FNO3D().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\nPINO(FNO3D)  |  width={WIDTH}  layers={N_LAYERS}  "
          f"modes=({N_MODES_X},{N_MODES_Y},{N_MODES_Z})  params={n_par:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    def lr_lambda(ep):
        if ep < 20:
            return ep / 20.0
        t = (ep - 20) / max(1, args.epochs - 20)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    x0, y0, m0 = next(iter(train_loader))
    with torch.no_grad():
        p0 = model(x0.to(device))
        c0, f0, s0 = physics_residuals(p0, m0.to(device), stats, dx, dy, dz)
    scale_cont = max(float(c0), 1e-8)
    scale_fe   = max(float(f0), 1e-8)
    scale_sl   = max(float(s0), 1e-8)
    print(f"  Physics scales: cont={scale_cont:.3e}  fe={scale_fe:.3e}  sl={scale_sl:.3e}\n")

    train_data_hist = []
    train_phys_hist = []
    val_hist        = []
    best_val        = float("inf")
    t0              = time.time()

    for epoch in range(1, args.epochs + 1):
        lam = phys_lambda(epoch, args.phys_warmup, args.phys_ramp, args.lambda_max)
        model.train()
        ep_data = ep_phys = 0.0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            noise = torch.randn_like(x[:, :6]) * INPUT_NOISE_STD
            x     = torch.cat([x[:, :6] + noise, x[:, 6:]], dim=1)
            optimizer.zero_grad()
            pred = model(x)
            ld   = data_loss(pred, y, mask)

            if lam > 0:
                lc, lfe, lsl = physics_residuals(pred, mask, stats, dx, dy, dz)
                lp   = lc / scale_cont + lfe / scale_fe + lsl / scale_sl
                loss = ld + lam * lp
                ep_phys += lp.item()
            else:
                loss = ld

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_data += ld.item()

        scheduler.step()
        ep_data /= len(train_loader)
        ep_phys /= len(train_loader)
        train_data_hist.append(ep_data)
        train_phys_hist.append(ep_phys)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                v_loss += data_loss(model(x), y, mask).item()
        v_loss /= len(val_loader)
        val_hist.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save({"model": model.state_dict(),
                        "stats": stats,
                        "epoch": epoch,
                        "lambda_used": lam,
                        "config": dict(width=WIDTH, n_layers=N_LAYERS,
                                       mx=N_MODES_X, my=N_MODES_Y, mz=N_MODES_Z)},
                       f"{args.out}/best_pino.pt")

        if epoch % 50 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            eta     = elapsed / epoch * (args.epochs - epoch)
            lr_now  = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"data={ep_data:.3e}  phys={ep_phys:.3e}  "
                  f"lambda={lam:.4f}  val={v_loss:.3e}  "
                  f"best={best_val:.3e}  lr={lr_now:.2e}  "
                  f"t={elapsed:.1f}m  ETA={eta:.1f}m")

    print(f"\nTraining complete in {(time.time() - t0) / 60:.1f} min")

    ckpt = torch.load(f"{args.out}/best_pino.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loss        = 0.0
    all_errors       = {n: [] for n in FIELD_NAMES}
    phys_cont_errors = []
    phys_fe_errors   = []
    phys_sl_errors   = []

    with torch.no_grad():
        for x, y, mask in test_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            test_loss += data_loss(pred, y, mask).item()
            lc, lfe, lsl = physics_residuals(pred, mask, stats, dx, dy, dz)
            phys_cont_errors.append(lc.item())
            phys_fe_errors.append(lfe.item())
            phys_sl_errors.append(lsl.item())
            errs = compute_rel_errors(pred.cpu(), y.cpu(), mask.cpu())
            for n in FIELD_NAMES:
                all_errors[n].extend(errs[n])

    test_loss /= len(test_loader)
    print(f"\nTest weighted MSE (data): {test_loss:.4e}")
    print(f"{'Field':<12}  Rel L2 error (masked, %)")
    print("-" * 40)
    for name in FIELD_NAMES:
        e = np.array(all_errors[name])
        print(f"  {name:<10}  {e.mean():.2f}% +/- {e.std():.2f}%")

    print(f"\nPhysics residuals on test set:")
    print(f"  Continuity   : {np.mean(phys_cont_errors):.4e}")
    print(f"  Fluid energy : {np.mean(phys_fe_errors):.4e}")
    print(f"  Solid Laplace: {np.mean(phys_sl_errors):.4e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(train_data_hist, label="train data loss")
    ax1.semilogy(val_hist,        label="val data loss")
    ax1.axvline(args.phys_warmup, color="k", ls=":", label="physics ON")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("weighted masked MSE")
    ax1.set_title("Data loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.semilogy([max(l, 1e-12) for l in train_phys_hist],
                 label="physics loss (normalised)", color="orange")
    ax2.axvline(args.phys_warmup, color="k", ls=":")
    ax2.set_xlabel("epoch"); ax2.set_title("Physics loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("PINO - CHT training", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{args.out}/pino_curves.png", dpi=150)
    plt.close()

    x_batch, y_batch, mask_batch = next(iter(test_loader))
    with torch.no_grad():
        pred_batch = model(x_batch.to(device)).cpu()

    f_mean_t = torch.tensor(stats["f_mean"]).float()
    f_std_t  = torch.tensor(stats["f_std"]).float()
    gt_dn    = y_batch    * f_std_t[None, :, None, None, None] + f_mean_t[None, :, None, None, None]
    pr_dn    = pred_batch * f_std_t[None, :, None, None, None] + f_mean_t[None, :, None, None, None]

    solid_per_x = mask_batch[0].sum(dim=(1, 2))
    x_mid       = int(solid_per_x.argmax().item())
    print(f"  y-z slice at x_idx={x_mid} (max solid voxels = {int(solid_per_x[x_mid])})")

    try:
        gy = test_ds.grid_y
        gz = test_ds.grid_z
    except AttributeError:
        gy = np.linspace(-0.5, 0.5, gt_dn.shape[3])
        gz = np.linspace(-0.5, 0.5, gt_dn.shape[4])
    extent_yz = [gz[0], gz[-1], gy[0], gy[-1]]

    mask_yz = mask_batch[0, x_mid, :, :].numpy()

    def apply_mask(arr2d, field_idx):
        out = arr2d.copy().astype(np.float64)
        if field_idx < 5:
            out[mask_yz > 0.5] = np.nan
        else:
            out[mask_yz < 0.5] = np.nan
        return out

    cmap_field = plt.get_cmap("RdYlBu_r").copy(); cmap_field.set_bad("white", 1.0)
    cmap_diff  = plt.get_cmap("RdBu_r").copy();   cmap_diff.set_bad("white",  1.0)

    fig, axes = plt.subplots(len(FIELD_NAMES), 3,
                             figsize=(13, 2.8 * len(FIELD_NAMES)),
                             constrained_layout=True)
    fig.suptitle("PINO - y-z plane at x=mid\n"
                 "Columns: PINO prediction | Ground Truth | Difference",
                 fontsize=11, y=1.01)

    for i, name in enumerate(FIELD_NAMES):
        gt_sl  = gt_dn[0, i, x_mid, :, :].numpy()
        pr_sl  = pr_dn[0, i, x_mid, :, :].numpy()
        gt_m   = apply_mask(gt_sl,         i)
        pr_m   = apply_mask(pr_sl,         i)
        diff_m = apply_mask(pr_sl - gt_sl, i)

        vmin = np.nanmin(gt_m); vmax = np.nanmax(gt_m)
        if vmin == vmax: vmin -= 0.1; vmax += 0.1
        d_all = np.abs(diff_m[~np.isnan(diff_m)])
        dabs  = max(d_all.max() if len(d_all) else 0.1, 1e-4)

        for col_j, (data, cmap, vlo, vhi, title) in enumerate([
            (pr_m,   cmap_field,  vmin,  vmax,  f"PINO: {name}"),
            (gt_m,   cmap_field,  vmin,  vmax,  f"GT: {name}"),
            (diff_m, cmap_diff,  -dabs,  dabs,  f"Difference: {name}"),
        ]):
            ax = axes[i, col_j]
            im = ax.imshow(data, origin="lower", extent=extent_yz,
                           aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi,
                           interpolation="bilinear")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("z  (span)", fontsize=8)
            ax.set_ylabel("y  (height)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, format="%.3g", aspect=20)

    plt.savefig(f"{args.out}/pino_prediction_yz.png", dpi=150, bbox_inches="tight")
    plt.close()

    labels = ["Continuity", "Fluid energy", "Solid Laplace"]
    values = [np.mean(phys_cont_errors), np.mean(phys_fe_errors), np.mean(phys_sl_errors)]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["steelblue", "coral", "seagreen"])
    plt.yscale("log")
    plt.ylabel("Mean residual MSE")
    plt.title("PINO - physics residuals on test set")
    plt.tight_layout()
    plt.savefig(f"{args.out}/pino_physics_residuals.png", dpi=150)
    plt.close()

    torch.save({
        "model":       model.state_dict(),
        "stats":       stats,
        "config":      dict(width=WIDTH, n_layers=N_LAYERS,
                            mx=N_MODES_X, my=N_MODES_Y, mz=N_MODES_Z),
        "test_errors": {n: np.array(all_errors[n]).mean() for n in FIELD_NAMES},
        "epochs":      args.epochs,
    }, f"{args.out}/final_pino.pt")

    print(f"\nAll outputs saved to {args.out}/")


if __name__ == "__main__":
    main()