#kolmogorovflow

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import gc
from torch.utils.data import RandomSampler, Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
torch.manual_seed(42)


# HYPERPARAMETERS

Nx, Ny    = 64, 64
Nx_train  = 32
Ny_train  = 32
Cin       = 1
Cout      = 1
N1, N2    = 64, 64
b         = 16
num_layers= 4
k_max     = 12
Re        = 500
nu        = 1.0 / Re
lam       = 0.01
L         = 2.0 * np.pi

EXPERIMENT   = 'superres'
T_STEP       = 1
dt           = float(T_STEP) * 1.0
TRAIN_T_END  = 400
TEST_T_START = 400

path = '/kaggle/input/datasets/rajj93/re-40-n200/KFvorticity_Re500_N1000_T500.npy'


# SPECTRAL UTILITIES

def get_wavenumbers(Nx, Ny, device):
    kx = torch.fft.fftfreq(Nx,  d=1.0/Nx).to(device)
    ky = torch.fft.rfftfreq(Ny, d=1.0/Ny).to(device)
    return torch.meshgrid(kx, ky, indexing='ij')


def get_forcing(Nx, Ny, device):
    y = torch.linspace(0.0, L, Ny+1, device=device)[:-1]
    f = -4.0 * torch.cos(4.0 * y)
    return f.unsqueeze(0).expand(Nx, -1).contiguous()


def physics_residual(w_pred, w_in, KX, KY, Nx_r, Ny_r, forcing, stats):
    wp  = w_pred[:, 0].float() * stats['w_std'] + stats['w_mean']
    w0p = w_in[:,  0].float()  * stats['w_std'] + stats['w_mean']
    w_hat = torch.fft.rfft2(wp, dim=(-2,-1))
    s   = 2.0 * np.pi / L
    ikx = 1j * s * KX.unsqueeze(0)
    iky = 1j * s * KY.unsqueeze(0)
    k2  = (s*KX)**2 + (s*KY)**2
    k2_safe = k2.clone().unsqueeze(0)
    k2_safe[:, 0, 0] = 1.0
    psi_hat = -w_hat / k2_safe
    u   = torch.fft.irfft2( iky*psi_hat,            s=(Nx_r,Ny_r), dim=(-2,-1))
    v   = torch.fft.irfft2(-ikx*psi_hat,            s=(Nx_r,Ny_r), dim=(-2,-1))
    wx  = torch.fft.irfft2( ikx*w_hat,              s=(Nx_r,Ny_r), dim=(-2,-1))
    wy  = torch.fft.irfft2( iky*w_hat,              s=(Nx_r,Ny_r), dim=(-2,-1))
    lap = torch.fft.irfft2(-k2.unsqueeze(0)*w_hat, s=(Nx_r,Ny_r), dim=(-2,-1))
    dw_dt = (wp - w0p) / dt
    return dw_dt + u*wx + v*wy - nu*lap - forcing.unsqueeze(0)


# MODEL DEFINITIONS
class SpectralConv2d(nn.Module):
    def __init__(self, N1, N2, k_max):
        super().__init__()
        self.k_max    = k_max
        self.N2       = N2
        self.spec_wt1 = nn.Parameter(
            torch.randn(N1, N2, k_max, k_max, dtype=torch.cfloat) / (N1*N2))
        self.spec_wt2 = nn.Parameter(
            torch.randn(N1, N2, k_max, k_max, dtype=torch.cfloat) / (N1*N2))

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k_max
        x_ft  = torch.fft.rfft2(x.float(), dim=(-2,-1))
        out_ft = torch.zeros(B, self.N2, H, W//2+1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:,:,  :k, :k] = torch.einsum(
            'bimn,iomn->bomn', x_ft[:,:,  :k, :k], self.spec_wt1)
        out_ft[:,:, -k:, :k] = torch.einsum(
            'bimn,iomn->bomn', x_ft[:,:, -k:, :k], self.spec_wt2)
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2,-1)).to(x.dtype)


class FNO_Layer(nn.Module):
    def __init__(self, N1, N2, k_max):
        super().__init__()
        self.spec   = SpectralConv2d(N1, N2, k_max)
        self.skip   = nn.Conv2d(N1, N2, 1)
        self.act    = nn.GELU()

    def forward(self, x):
        return self.act(self.spec(x) + self.skip(x))


class PINO_Layer(nn.Module):
    def __init__(self, N1, N2, k_max):
        super().__init__()
        self.spec        = SpectralConv2d(N1, N2, k_max)
        self.channel_mix = nn.Conv2d(N2, N2, 1)
        self.skip1       = nn.Conv2d(N1, N2, 1)
        self.skip2       = nn.Conv2d(N1, N2, 1)
        self.act         = nn.Tanh()

    def forward(self, x):
        y = x
        x = self.act(self.spec(x) + self.skip1(y))
        x = self.act(self.channel_mix(x) + self.skip2(y))
        return x


class FNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift    = nn.Conv2d(Cin, N1, 1)
        self.project = nn.Conv2d(N2, Cout, 1)
        self.layers  = nn.ModuleList([
            FNO_Layer(N1, N2, k_max) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        return self.project(x)


class PINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift    = nn.Conv2d(Cin, N1, 1)
        self.project = nn.Conv2d(N2, Cout, 1)
        self.layers  = nn.ModuleList([
            PINO_Layer(N1, N2, k_max) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        return self.project(x)



# DATASET

class KFDataset(Dataset):
    def __init__(self, path, traj_indices, T_STEP=1,
                 t_start=0, t_end=499, out_res=64):
        self.raw     = np.load(path, mmap_mode='r')
        self.T_STEP  = T_STEP
        self.step    = 64 // out_res
        self.pairs   = [
            (traj, t)
            for traj in traj_indices
            for t in range(t_start, t_end, T_STEP)
        ]
        sample       = self.raw[list(traj_indices)[:50], 0, :, :].astype(np.float32)
        self.w_mean  = float(sample.mean())
        self.w_std   = float(sample.std())
        print(f"  [{out_res}x{out_res}] t={t_start}..{t_end}  "
              f"pairs={len(self.pairs):,}  "
              f"mean={self.w_mean:.4f}  std={self.w_std:.4f}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        traj, t = self.pairs[idx]
        st = self.step
        w_in  = self.raw[traj, t,             ::st, ::st].astype(np.float32)
        w_out = self.raw[traj, t+self.T_STEP, ::st, ::st].astype(np.float32)
        w_in  = (w_in  - self.w_mean) / self.w_std
        w_out = (w_out - self.w_mean) / self.w_std
        return (torch.from_numpy(w_in).unsqueeze(0),
                torch.from_numpy(w_out).unsqueeze(0))


# DATA LOADERS

print(f"\nEXPERIMENT: {EXPERIMENT}")
print("=" * 60)
ALL_TRAJ = range(0, 1000)

if EXPERIMENT == 'temporal':
    print("Temporal generalisation: train t<400, test t>=400")
    TRAIN_RES = 64; TEST_RES = 64
    train_ds = KFDataset(path, ALL_TRAJ, T_STEP, 0,            TRAIN_T_END, 64)
    val_ds   = KFDataset(path, ALL_TRAJ, T_STEP, TRAIN_T_END-50, TRAIN_T_END, 64)
    test_ds  = KFDataset(path, ALL_TRAJ, T_STEP, TEST_T_START, 499,         64)

elif EXPERIMENT == 'superres':
    print("Zero-shot super-resolution: train 32x32, test 64x64")
    TRAIN_RES = 32; TEST_RES = 64
    train_ds = KFDataset(path, range(0,   800), T_STEP, 0, 499, out_res=32)
    val_ds   = KFDataset(path, range(800, 900), T_STEP, 0, 499, out_res=32)
    test_ds  = KFDataset(path, range(900,1000), T_STEP, 0, 499, out_res=64)

elif EXPERIMENT == 'both':
    print("Both: train 32x32 t<400, test 64x64 t>=400")
    TRAIN_RES = 32; TEST_RES = 64
    train_ds = KFDataset(path, ALL_TRAJ, T_STEP, 0,              TRAIN_T_END, 32)
    val_ds   = KFDataset(path, ALL_TRAJ, T_STEP, TRAIN_T_END-50, TRAIN_T_END, 32)
    test_ds  = KFDataset(path, ALL_TRAJ, T_STEP, TEST_T_START,   499,         64)


val_ds.w_mean  = train_ds.w_mean; val_ds.w_std  = train_ds.w_std
test_ds.w_mean = train_ds.w_mean; test_ds.w_std = train_ds.w_std
stats = dict(w_mean=train_ds.w_mean, w_std=train_ds.w_std,
             w0_mean=train_ds.w_mean, w0_std=train_ds.w_std,
             wT_mean=train_ds.w_mean, wT_std=train_ds.w_std)

PAIRS_PER_EPOCH = 4000
train_sampler   = RandomSampler(train_ds, replacement=True,
                                num_samples=PAIRS_PER_EPOCH)
val_sampler     = RandomSampler(val_ds, replacement=False,
                                num_samples=min(1000, len(val_ds)))

train_loader = DataLoader(train_ds, batch_size=b, sampler=train_sampler,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=b, sampler=val_sampler,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=b, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"\ntrain batches/epoch : {len(train_loader)}")
print(f"val   batches/epoch : {len(val_loader)}")

xb, yb = next(iter(train_loader))
print(f"train batch — input: {list(xb.shape)}  target: {list(yb.shape)}")
xbt, ybt = next(iter(test_loader))
print(f"test  batch — input: {list(xbt.shape)}  target: {list(ybt.shape)}")

# DEVICE + PHYSICS GRIDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

KX_tr, KY_tr = get_wavenumbers(TRAIN_RES, TRAIN_RES, device)
forcing_tr   = get_forcing(TRAIN_RES, TRAIN_RES, device)
KX_te, KY_te = get_wavenumbers(TEST_RES,  TEST_RES,  device)
forcing_te   = get_forcing(TEST_RES,  TEST_RES,  device)

print("Estimating physics residual scale ...")
phys_scale = 0.0
_tmp = PINO().to(device)
with torch.no_grad():
    for i, (bx, by) in enumerate(train_loader):
        if i >= 5: break
        res = physics_residual(by.to(device), bx.to(device),
                               KX_tr, KY_tr, TRAIN_RES, TRAIN_RES,
                               forcing_tr, stats)
        phys_scale += float(res.pow(2).mean())
del _tmp
phys_scale = max(phys_scale / 5, 1e-8)
print(f"  phys_scale = {phys_scale:.4e}")


# GENERIC TRAINING FUNCTION
def train_model(model, model_name, use_physics=False,
                epochs=500, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"Training {model_name}  |  physics={'ON' if use_physics else 'OFF'}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup    = 100
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: (ep+1)/warmup if ep < warmup
                   else 0.5*(1 + np.cos(np.pi*(ep-warmup)/max(1,epochs-warmup)))
    )
    criterion   = nn.MSELoss()
    phys_warmup = 100
    phys_tau    = 20
    train_losses, val_losses = [], []

    try:
        for epoch in range(epochs):
            model.train()
            lam_ep = (0.0 if (not use_physics or epoch < phys_warmup)
                      else lam*(1 - np.exp(-(epoch-phys_warmup)/phys_tau)))

            ep_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred      = model(batch_x)
                loss_data = criterion(pred, batch_y)

                if use_physics and lam_ep > 0:
                    res       = physics_residual(pred, batch_x,
                                                 KX_tr, KY_tr,
                                                 TRAIN_RES, TRAIN_RES,
                                                 forcing_tr, stats)
                    loss_phys = torch.mean(res**2) / phys_scale
                    loss      = loss_data + lam_ep * loss_phys
                else:
                    loss_phys = torch.tensor(0.0)
                    loss      = loss_data

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss_data.item()

            scheduler.step()
            ep_loss /= len(train_loader)
            train_losses.append(ep_loss)

            # Validation
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    vl += criterion(model(bx.to(device)), by.to(device)).item()
            vl /= len(val_loader)
            val_losses.append(vl)

            if epoch % 50 == 0 or epoch == epochs-1:
                print(f"  [{model_name}] Ep {epoch:4d} | "
                      f"train={ep_loss:.5f} | val={vl:.5f} | "
                      f"phys={loss_phys.item():.4f} | lam={lam_ep:.4f}")

    except KeyboardInterrupt:
        print(f"[{model_name}] interrupted at epoch {epoch}")
    except RuntimeError as e:
        print(f"[{model_name}] RuntimeError: {e}")

    print(f"[{model_name}] Training complete.")
    model.eval()
    return train_losses, val_losses

# GENERIC EVALUATION FUNCTION

def evaluate_model(model, model_name):
    print(f"\n[{model_name}] Testing at {TEST_RES}x{TEST_RES} ...")
    criterion  = nn.MSELoss()
    test_mse   = 0.0
    rel_errors = []
    phys_res   = []

    pred_ex = gt_ex = inp_ex = None  

    model.eval()
    with torch.no_grad():
        for i, (bx, by) in enumerate(test_loader):
            bx = bx.to(device); by = by.to(device)
            pred = model(bx)
            test_mse += criterion(pred, by).item()
            for j in range(pred.shape[0]):
                rel_errors.append(
                    (torch.norm(pred[j]-by[j]) /
                     (torch.norm(by[j])+1e-8)).item() * 100.0)

            if i < 10:
                res_te = physics_residual(pred, bx,
                                          KX_te, KY_te,
                                          TEST_RES, TEST_RES,
                                          forcing_te, stats)
                phys_res.append(float(res_te.pow(2).mean()))

            
            if pred_ex is None:
                pred_ex = pred.cpu()
                gt_ex   = by.cpu()
                inp_ex  = bx.cpu()

    test_mse /= len(test_loader)
    print(f"  [{model_name}] test MSE      : {test_mse:.6f}")
    print(f"  [{model_name}] rel L2 error : "
          f"{np.mean(rel_errors):.2f}% ± {np.std(rel_errors):.2f}%")
    print(f"  [{model_name}] PDE resid @{TEST_RES}x{TEST_RES} : "
          f"{np.mean(phys_res):.4e}  (scale={phys_scale:.4e}  "
          f"ratio={np.mean(phys_res)/phys_scale:.2f}x)")

    return dict(
        test_mse=test_mse,
        rel_errors=rel_errors,
        phys_residuals=phys_res,
        pred=pred_ex, gt=gt_ex, inp=inp_ex,
    )



EPOCHS = 500

fno_model  = FNO().to(device)
pino_model = PINO().to(device)

print(f"\nFNO  parameters : {sum(p.numel() for p in fno_model.parameters()):,}")
print(f"PINO parameters : {sum(p.numel() for p in pino_model.parameters()):,}")
print(f"Training resolution : {TRAIN_RES}x{TRAIN_RES}")
print(f"Test     resolution : {TEST_RES}x{TEST_RES}")

# Train FNO 
fno_train_losses, fno_val_losses = train_model(
    fno_model,  "FNO",  use_physics=False, epochs=EPOCHS)

gc.collect(); torch.cuda.empty_cache()

# Train PINO 
pino_train_losses, pino_val_losses = train_model(
    pino_model, "PINO", use_physics=True,  epochs=EPOCHS)

gc.collect(); torch.cuda.empty_cache()


fno_results  = evaluate_model(fno_model,  "FNO")
pino_results = evaluate_model(pino_model, "PINO")


print("\n" + "="*60)
print(f"{'METRIC':<35} {'FNO':>10} {'PINO':>10}")
print("="*60)
print(f"{'Test MSE':<35} "
      f"{fno_results['test_mse']:>10.6f} {pino_results['test_mse']:>10.6f}")
print(f"{'Relative L2 Error (%)':<35} "
      f"{np.mean(fno_results['rel_errors']):>10.2f} "
      f"{np.mean(pino_results['rel_errors']):>10.2f}")
print(f"{'PDE Residual @{TEST_RES}x{TEST_RES}':<35} "
      f"{np.mean(fno_results['phys_residuals']):>10.4e} "
      f"{np.mean(pino_results['phys_residuals']):>10.4e}")
print(f"{'PDE Residual / phys_scale':<35} "
      f"{np.mean(fno_results['phys_residuals'])/phys_scale:>10.3f}x "
      f"{np.mean(pino_results['phys_residuals'])/phys_scale:>10.3f}x")
print("="*60)


# PLOT 1: TRAINING CURVES — FNO vs PINO

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train loss
axes[0].semilogy(fno_train_losses,  label='FNO train',  color='steelblue', lw=2)
axes[0].semilogy(pino_train_losses, label='PINO train', color='darkorange', lw=2)
axes[0].axvline(100, color='k', ls=':', lw=1.5, label='physics warmup (PINO)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE (log)')
axes[0].set_title(f'Training Loss — {EXPERIMENT}')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Val loss
axes[1].semilogy(fno_val_losses,  label='FNO val',  color='steelblue', lw=2)
axes[1].semilogy(pino_val_losses, label='PINO val', color='darkorange', lw=2)
axes[1].axvline(100, color='k', ls=':', lw=1.5, label='physics warmup (PINO)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE (log)')
axes[1].set_title(f'Validation Loss — {EXPERIMENT}')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Training Curves: FNO vs PINO  |  EXPERIMENT: {EXPERIMENT}',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# PLOT 2: 6-PANEL PREDICTION COMPARISON

fno_pred  = fno_results['pred']
pino_pred = pino_results['pred']
gt        = fno_results['gt']       
inp       = fno_results['inp']


vmin = float(gt[0,0].min())
vmax = float(gt[0,0].max())
fno_err  = (fno_pred[0,0]  - gt[0,0]).abs().numpy()
pino_err = (pino_pred[0,0] - gt[0,0]).abs().numpy()
err_max  = max(float(fno_err.max()), float(pino_err.max()))

fig, axes = plt.subplots(1, 6, figsize=(28, 4))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes:
    ax.set_facecolor('#0f0f1a')

ims = [
    axes[0].imshow(inp[0,0].numpy(),       cmap='RdBu_r'),
    axes[1].imshow(gt[0,0].numpy(),        cmap='RdBu_r', vmin=vmin, vmax=vmax),
    axes[2].imshow(fno_pred[0,0].numpy(),  cmap='RdBu_r', vmin=vmin, vmax=vmax),
    axes[3].imshow(fno_err,                cmap='hot',    vmin=0,    vmax=err_max),
    axes[4].imshow(pino_pred[0,0].numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax),
    axes[5].imshow(pino_err,               cmap='hot',    vmin=0,    vmax=err_max),
]
titles = [
    f'Input\n({TEST_RES}×{TEST_RES})',
    f'Ground Truth\n({TEST_RES}×{TEST_RES})',
    f'FNO Prediction\n(trained {TRAIN_RES}×{TRAIN_RES})',
    f'|FNO Error|\nrel-L2={np.mean(fno_results["rel_errors"]):.1f}%',
    f'PINO Prediction\n(trained {TRAIN_RES}×{TRAIN_RES})',
    f'|PINO Error|\nrel-L2={np.mean(pino_results["rel_errors"]):.1f}%',
]
for ax, im, title in zip(axes, ims, titles):
    ax.set_title(title, color='white', fontsize=10, pad=6)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.ax.tick_params(colors='white', labelsize=7)

for ax in [axes[3], axes[5]]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#ff6b35'); spine.set_linewidth(2)

plt.suptitle(
    f'FNO vs PINO — Prediction Comparison  |  {EXPERIMENT.upper()}'
    f'\nTrained {TRAIN_RES}×{TRAIN_RES} → Tested {TEST_RES}×{TEST_RES}  '
    f'(zero-shot super-resolution)',
    fontsize=12, fontweight='bold', color='white', y=1.02
)
plt.tight_layout()
fig.savefig('fno_pino_comparison.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

# PLOT 3: 1-D VORTICITY SLICE COMPARISON

x_ax = np.linspace(0, 2*np.pi, TEST_RES)
mid  = TEST_RES // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, t_idx, label in [(axes[0], 0, 'Sample 1'), (axes[1], 4, 'Sample 5')]:
    ax.plot(x_ax, gt[t_idx,0,:,mid].numpy(),
            label=f'GT ({TEST_RES}×{TEST_RES})',
            color='black', lw=2.5)
    ax.plot(x_ax, fno_pred[t_idx,0,:,mid].numpy(),
            label=f'FNO (train {TRAIN_RES}×{TRAIN_RES})',
            color='steelblue', lw=2, ls='--')
    ax.plot(x_ax, pino_pred[t_idx,0,:,mid].numpy(),
            label=f'PINO (train {TRAIN_RES}×{TRAIN_RES})',
            color='darkorange', lw=2, ls='-.')
    ax.set_xlabel('x'); ax.set_ylabel('Vorticity ω')
    ax.set_title(f'Vorticity Slice — {label}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle(
    f'1-D Vorticity Slices: FNO vs PINO\n'
    f'Trained {TRAIN_RES}×{TRAIN_RES}, Tested {TEST_RES}×{TEST_RES}',
    fontsize=12, fontweight='bold'
)
plt.tight_layout(); plt.show()

# PLOT 4: ERROR DISTRIBUTION  (violin / box plot)

fig, ax = plt.subplots(figsize=(7, 5))
vp = ax.violinplot([fno_results['rel_errors'], pino_results['rel_errors']],
                   positions=[1, 2], showmedians=True, showextrema=True)

colors = ['steelblue', 'darkorange']
for patch, color in zip(vp['bodies'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for part in ['cmedians','cmins','cmaxes','cbars']:
    vp[part].set_color('black'); vp[part].set_linewidth(1.5)

ax.set_xticks([1, 2])
ax.set_xticklabels(['FNO\n(no physics)', 'PINO\n(+NS residual)'], fontsize=12)
ax.set_ylabel('Relative L2 Error (%)', fontsize=11)
ax.set_title('Error Distribution: FNO vs PINO\n'
             f'Test {TEST_RES}×{TEST_RES}  (trained {TRAIN_RES}×{TRAIN_RES})',
             fontsize=12)

for x_pos, errs, color in zip([1, 2],
                               [fno_results['rel_errors'],
                                pino_results['rel_errors']],
                               colors):
    med = np.median(errs)
    ax.text(x_pos, med + 0.5, f'{med:.1f}%',
            ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=color)

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()


# PLOT 5: ENERGY SPECTRUM — key superres diagnostic

def energy_spectrum_1d(w_np):
    wh    = np.fft.fft2(w_np)
    power = np.abs(wh)**2
    kx    = np.fft.fftfreq(w_np.shape[0], 1.0/w_np.shape[0]).astype(int)
    ky    = np.fft.fftfreq(w_np.shape[1], 1.0/w_np.shape[1]).astype(int)
    KXn,KYn = np.meshgrid(kx, ky, indexing='ij')
    K     = np.round(np.sqrt(KXn**2+KYn**2)).astype(int)
    kmax  = min(w_np.shape) // 2
    return np.array([power[K==ki].sum() for ki in range(kmax)])

w_std  = train_ds.w_std
w_mean = train_ds.w_mean

n_spec = min(16, gt.shape[0])
Et_all = np.zeros(TEST_RES//2)
Ef_all = np.zeros(TEST_RES//2)
Ep_all = np.zeros(TEST_RES//2)
for i in range(n_spec):
    yt = gt[i,0].numpy()       * w_std + w_mean
    ft = fno_pred[i,0].numpy() * w_std + w_mean
    pt = pino_pred[i,0].numpy()* w_std + w_mean
    Et_all += energy_spectrum_1d(yt)
    Ef_all += energy_spectrum_1d(ft)
    Ep_all += energy_spectrum_1d(pt)
Et_all /= n_spec; Ef_all /= n_spec; Ep_all /= n_spec

ka       = np.arange(1, len(Et_all))
k_cutoff = TRAIN_RES // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].loglog(ka, Et_all[1:], color='black',      lw=2.5, label=f'GT ({TEST_RES}×{TEST_RES})')
axes[0].loglog(ka, Ef_all[1:], color='steelblue',  lw=2,   label=f'FNO (train {TRAIN_RES}×{TRAIN_RES})', ls='--')
axes[0].loglog(ka, Ep_all[1:], color='darkorange',  lw=2,   label=f'PINO (train {TRAIN_RES}×{TRAIN_RES})', ls='-.')
axes[0].axvline(k_cutoff, color='red', ls=':', lw=2,
                label=f'Train cutoff k={k_cutoff}')
k_ref = np.array([5, k_cutoff])
axes[0].loglog(k_ref, Et_all[5] * (k_ref/5)**(-5/3), color='gray',
               ls=':', lw=1.5, label='k⁻⁵/³ ref')
axes[0].set_xlabel('Wavenumber k', fontsize=11)
axes[0].set_ylabel('E(k)', fontsize=11)
axes[0].set_title('Energy Spectrum (log-log)\nModes right of red = never seen at train', fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.2)

Ef_rel = np.abs(Ef_all[1:] - Et_all[1:]) / (Et_all[1:] + 1e-30) * 100
Ep_rel = np.abs(Ep_all[1:] - Et_all[1:]) / (Et_all[1:] + 1e-30) * 100
axes[1].semilogx(ka, Ef_rel, color='steelblue',  lw=2, label='FNO',  ls='--')
axes[1].semilogx(ka, Ep_rel, color='darkorange',  lw=2, label='PINO', ls='-.')
axes[1].axvline(k_cutoff, color='red', ls=':', lw=2,
                label=f'Train cutoff k={k_cutoff}')
axes[1].set_xlabel('Wavenumber k', fontsize=11)
axes[1].set_ylabel('Relative Error in E(k) (%)', fontsize=11)
axes[1].set_title('Spectral Error vs GT\n(high-k = zero-shot extrapolation)', fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.suptitle('Energy Spectrum: FNO vs PINO',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()


# PLOT 6: PDE RESIDUAL COMPARISON BAR CHART

fno_phys_mean  = np.mean(fno_results['phys_residuals'])
pino_phys_mean = np.mean(pino_results['phys_residuals'])
fno_phys_std   = np.std(fno_results['phys_residuals'])
pino_phys_std  = np.std(pino_results['phys_residuals'])

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(['FNO\n(no physics)', 'PINO\n(+NS residual)'],
               [fno_phys_mean, pino_phys_mean],
               yerr=[fno_phys_std, pino_phys_std],
               color=['steelblue', 'darkorange'], alpha=0.8,
               capsize=8, edgecolor='black', linewidth=1.2)

ax.axhline(phys_scale, color='red', ls='--', lw=1.5,
           label=f'phys_scale (training level) = {phys_scale:.2e}')
ax.set_ylabel('PDE Residual MSE', fontsize=11)
ax.set_title(f'NS Equation Residual at {TEST_RES}×{TEST_RES}\n'
             '(lower = more physics-consistent prediction)', fontsize=11)
ax.legend(fontsize=9)

for bar, val, name in zip(bars,
                           [fno_phys_mean, pino_phys_mean],
                           ['FNO', 'PINO']):
    ratio = val / phys_scale
    ax.text(bar.get_x() + bar.get_width()/2, val + fno_phys_std*0.2,
            f'{ratio:.2f}×\nphys_scale',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("\nAll done.")
print(f"\nFINAL SUMMARY")
print(f"{'='*50}")
print(f"                     FNO          PINO")
print(f"Test MSE         {fno_results['test_mse']:>10.5f}    "
      f"{pino_results['test_mse']:>10.5f}")
print(f"Rel L2 Error %   {np.mean(fno_results['rel_errors']):>10.2f}    "
      f"{np.mean(pino_results['rel_errors']):>10.2f}")
print(f"PDE Residual     {fno_phys_mean:>10.4e}    {pino_phys_mean:>10.4e}")
print(f"Ratio/phys_scale {fno_phys_mean/phys_scale:>10.3f}x    "
      f"{pino_phys_mean/phys_scale:>10.3f}x")
print(f"{'='*50}")

# SAVE MODEL FOR TRANSFER LEARNING
save_path = f'pino_Re{Re}_source.pt'
torch.save(pino_model.state_dict(), save_path)
print(f"\n[SAVE] PINO model successfully saved to: {save_path}")

##################################################################
##################################################################
##################################################################
#for transfer learning we saved model from above and use it for other reynolds number
#here we are saving the model for Re=100 and then train re=40 and re=500, from here


import torch, torch.nn as nn, numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import RandomSampler, DataLoader
import matplotlib.pyplot as plt, gc, copy

# 1. Physics residual with EXPLICIT nu-
def physics_residual_nu(w_pred, w_in, KX, KY, Nx_r, Ny_r,
                        forcing, stats, nu_val, dt_val=1.0):
    L_domain = 2.0 * np.pi

    wp  = w_pred[:, 0].float() * stats['w_std'] + stats['w_mean']
    w0p = w_in[:,  0].float()  * stats['w_std'] + stats['w_mean']

    w_hat = torch.fft.rfft2(wp, dim=(-2, -1))

    s   = 2.0 * np.pi / L_domain
    ikx = 1j * s * KX.unsqueeze(0)
    iky = 1j * s * KY.unsqueeze(0)
    k2  = (s * KX) ** 2 + (s * KY) ** 2

    k2_safe = k2.clone().unsqueeze(0)
    k2_safe[:, 0, 0] = 1.0
    psi_hat = -w_hat / k2_safe

    u   = torch.fft.irfft2( iky * psi_hat,          s=(Nx_r, Ny_r), dim=(-2, -1))
    v   = torch.fft.irfft2(-ikx * psi_hat,          s=(Nx_r, Ny_r), dim=(-2, -1))
    wx  = torch.fft.irfft2( ikx * w_hat,            s=(Nx_r, Ny_r), dim=(-2, -1))
    wy  = torch.fft.irfft2( iky * w_hat,            s=(Nx_r, Ny_r), dim=(-2, -1))
    lap = torch.fft.irfft2(-k2.unsqueeze(0) * w_hat, s=(Nx_r, Ny_r), dim=(-2, -1))

    dw_dt = (wp - w0p) / dt_val
    return dw_dt + u * wx + v * wy - nu_val * lap - forcing.unsqueeze(0)


# Hyper-parameters
ALPHA_ANCHOR  = 0.5   
EPOCHS_TL     = 500    
LR_TL         = 3e-4  
N_PHYS_SCALE_BATCHES = 10 

criterion = nn.MSELoss()

# Target experiments
target_experiments = [
    {
        'Re'        : 40,
        'path'      : '/kaggle/input/datasets/rajj93/re-40-n200/KFvorticity_Re40_N200_T500.npy',
        'traj_count': 200,
    },
    {
        'Re'        : 500,
        'path'      : '/kaggle/input/datasets/rajj93/re-40-n200/KFvorticity_Re500_N1000_T500.npy', # <-- UPDATE THIS
        'traj_count': 1000,
    },
]

def energy_spectrum_1d(field):
    nx, ny = field.shape
    # 2D Fast Fourier Transform
    uhat = np.fft.fftn(field)
    # Energy in spectral space (amplitude squared)
    tkh = np.abs(uhat)**2 / (nx * ny)**2
    
    # Create grid of wavenumbers
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Calculate radial wavenumbers
    K_abs = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    
    # Fast binning into a 1D array
    E = np.bincount(K_abs.flatten(), weights=tkh.flatten())
    return E


# Main loop
for exp in target_experiments:
    Target_Re  = exp['Re']
    target_path = exp['path']
    n_traj     = exp['traj_count']
    nu_target  = 1.0 / Target_Re  

    print("\n" + "=" * 65)
    print(f" TRANSFER LEARNING  Re=100  →  Re={Target_Re}")
    print("=" * 65)

    target_train_ds = KFDataset(
        target_path, range(0, n_traj), T_STEP,
        t_start=0, t_end=TRAIN_T_END, out_res=TRAIN_RES
    )

    stats_tl = dict(
        w_mean  = target_train_ds.w_mean,
        w_std   = target_train_ds.w_std,
        w0_mean = target_train_ds.w_mean,
        w0_std  = target_train_ds.w_std,
        wT_mean = target_train_ds.w_mean,
        wT_std  = target_train_ds.w_std,
    )

    target_train_loader = DataLoader(
        target_train_ds, batch_size=b,
        sampler=RandomSampler(target_train_ds, replacement=True, num_samples=PAIRS_PER_EPOCH),
        num_workers=2, pin_memory=True
    )

    print(f"Estimating phys_scale for Re={Target_Re} ...")
    _tmp_model = PINO().to(device)
    _tmp_model.load_state_dict(torch.load('/kaggle/input/datasets/sherbettowers/pino-source/pino_Re100_source_final.pt')) # <-- CHECK THIS PATH
    _tmp_model.eval()
    phys_scale_target = 0.0
    with torch.no_grad():
        for _i, (_bx, _by) in enumerate(target_train_loader):
            if _i >= N_PHYS_SCALE_BATCHES:
                break
            _res = physics_residual_nu(
                _by.to(device), _bx.to(device),
                KX_tr, KY_tr, TRAIN_RES, TRAIN_RES,
                forcing_tr, stats_tl, nu_target, dt
            )
            phys_scale_target += float(_res.pow(2).mean())
    phys_scale_target = max(phys_scale_target / N_PHYS_SCALE_BATCHES, 1e-8)
    del _tmp_model
    print(f"phys_scale (Re={Target_Re}) = {phys_scale_target:.4e}")

    # 4. copy of source model + frozen anchor
    tl_model = PINO().to(device)
    tl_model.load_state_dict(torch.load('/kaggle/input/datasets/sherbettowers/pino-source/pino_Re100_source_final.pt')) # <-- CHECK THIS PATH
    tl_model.train()

    anchor_model = PINO().to(device)
    anchor_model.load_state_dict(torch.load('/kaggle/input/datasets/sherbettowers/pino-source/pino_Re100_source_final.pt')) # <-- CHECK THIS PATH
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad_(False)

    optimizer_tl = optim.AdamW(tl_model.parameters(), lr=LR_TL, weight_decay=1e-4)
    scheduler_tl = lr_scheduler.CosineAnnealingLR(optimizer_tl, T_max=EPOCHS_TL, eta_min=1e-5)

  
    # 5. Transfer-learning loop 
    history = {'phys': [], 'anchor': []}

    for epoch in range(EPOCHS_TL):
        tl_model.train()
        ep_phys = ep_anchor = 0.0

        for batch_x, _ in target_train_loader:  
            batch_x = batch_x.to(device)
            optimizer_tl.zero_grad()

            pred = tl_model(batch_x)

            residual  = physics_residual_nu(
                pred, batch_x,
                KX_tr, KY_tr, TRAIN_RES, TRAIN_RES,
                forcing_tr, stats_tl, nu_target, dt
            )
            loss_phys = torch.mean(residual ** 2) / phys_scale_target

            #anchor los
            with torch.no_grad():
                anchor_pred = anchor_model(batch_x)
            loss_anchor = criterion(pred, anchor_pred)

            loss = loss_phys + ALPHA_ANCHOR * loss_anchor

            loss.backward()
            torch.nn.utils.clip_grad_norm_(tl_model.parameters(), 1.0)
            optimizer_tl.step()

            ep_phys   += loss_phys.item()
            ep_anchor += loss_anchor.item()

        scheduler_tl.step()
        nb = len(target_train_loader)
        ep_phys /= nb; ep_anchor /= nb

        history['phys'].append(ep_phys)
        history['anchor'].append(ep_anchor)

        if epoch % 20 == 0 or epoch == EPOCHS_TL - 1:
            print(f"Re={Target_Re} | Ep {epoch:3d} | "
                  f"phys={ep_phys:.4f} | anchor={ep_anchor:.5f} | "
                  f"lr={scheduler_tl.get_last_lr()[-1]:.2e}")

  
    # 6. Evaluation on held-out TEST time window
    target_test_ds = KFDataset(
        target_path, range(0, n_traj), T_STEP,
        t_start=TEST_T_START, t_end=499, out_res=TEST_RES
    )
    target_test_ds.w_mean = stats_tl['w_mean']
    target_test_ds.w_std  = stats_tl['w_std']
    target_test_loader = DataLoader(
        target_test_ds, batch_size=b, shuffle=False,
        num_workers=2, pin_memory=True
    )

    tl_model.eval()
    test_loss  = 0.0
    rel_errors = []
    with torch.no_grad():
        for bx, by in target_test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = tl_model(bx)
            test_loss += criterion(pred, by).item()
            for i in range(pred.shape[0]):
                rel_errors.append(
                    (torch.norm(pred[i] - by[i]) /
                     (torch.norm(by[i]) + 1e-8)).item() * 100.0
                )
    test_loss /= len(target_test_loader)
    print(f"\n Test MSE       : {test_loss:.6f}")
    print(f" Relative L2    : {np.mean(rel_errors):.2f}% ± {np.std(rel_errors):.2f}%")

    # 7. Plots
    # Plot 0: training curves
    plt.figure(figsize=(8, 4))
    plt.semilogy(history['phys'],   label='physics loss')
    plt.semilogy(history['anchor'], label='anchor loss')
    plt.xlabel('epoch'); plt.ylabel('Loss')
    plt.legend(); plt.title(f'Training Curves — Re=100 → Re={Target_Re}')
    plt.tight_layout(); plt.show()

    # Plot 1:
    bx, by = next(iter(target_test_loader))
    with torch.no_grad():
        pred = tl_model(bx.to(device)).cpu()
    diff = (pred[0, 0] - by[0, 0]).abs()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    vmin = float(by[0, 0].min()); vmax = float(by[0, 0].max())
    im0 = axes[0].imshow(bx[0, 0],   cmap='RdBu_r')
    im1 = axes[1].imshow(by[0, 0],   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    im2 = axes[2].imshow(pred[0, 0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    im3 = axes[3].imshow(diff,        cmap='hot')
    for ax, im, t in zip(
        axes, [im0, im1, im2, im3],
        [f'Input (Re={Target_Re})', f'GT (Re={Target_Re})',
         'Fine-Tuned PINO', '|GT − PINO|']
    ):
        ax.set_title(t); ax.axis('off'); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(f'Transfer Learning: Re=100 → Re={Target_Re}', fontsize=14)
    plt.tight_layout(); plt.show()

    # Plot B: 1-D slice
    x_ax = np.linspace(0, 2 * np.pi, TEST_RES)
    plt.figure(figsize=(10, 4))
    plt.plot(x_ax, by[0, 0, :, TEST_RES // 2].numpy(),
             label=f'GT (Re={Target_Re})', lw=2)
    plt.plot(x_ax, pred[0, 0, :, TEST_RES // 2].numpy(),
             label='Fine-Tuned PINO', ls='--', lw=2)
    plt.xlabel('x'); plt.ylabel('vorticity w'); plt.legend()
    plt.title(f'Vorticity slice at y=π  (Re={Target_Re})')
    plt.tight_layout(); plt.show()

    # Plot C: energy spectrum
    yt = by[0, 0].numpy()   * stats_tl['w_std'] + stats_tl['w_mean']
    pt = pred[0, 0].numpy() * stats_tl['w_std'] + stats_tl['w_mean']
    Et = energy_spectrum_1d(yt)
    Ep = energy_spectrum_1d(pt)
    ka = np.arange(1, len(Et))
    plt.figure(figsize=(7, 5))
    plt.loglog(ka, Et[1:], label=f'GT (Re={Target_Re})', color='steelblue')
    plt.loglog(ka, Ep[1:], label='Fine-Tuned PINO',      color='orange', ls='--')
    plt.axvline(TRAIN_RES // 2, color='red', ls=':', lw=1.5,
                label=f'training k_max={TRAIN_RES // 2}')
    plt.xlabel('wavenumber k'); plt.ylabel('E(k)'); plt.legend()
    plt.title(f'Energy Spectrum  (Re={Target_Re})')
    plt.tight_layout(); plt.show()

    # 8. fine-tuned model
    save_path = f'pino_Re{Target_Re}_finetuned.pt'
    torch.save(tl_model.state_dict(), save_path)
    print(f"Saved: {save_path}")

    # cleanup
    del (tl_model, anchor_model, optimizer_tl, scheduler_tl,
         target_train_ds, target_test_ds,
         target_train_loader, target_test_loader)
    gc.collect()
    torch.cuda.empty_cache()

print("\n ALL EXPERIMENTS FINISHED!")