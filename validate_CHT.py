"""
validate_nvidia.py - Validate PINN / FNO / PINO against NVIDIA OpenFOAM
=========================================================================
Loads the NVIDIA CHT reference CSV files (threeFin_extend_*) and
compares all three models on the same reference point-cloud.

NVIDIA reference geometry (laminar case):
  Channel   : (-2.5, -0.5, -0.5) to (2.5, 0.5, 0.5)
  Heat-sink : base (-1, -0.5, -0.3) -> (0, -0.3, 0.3), fins h=0.4 t=0.1 l=1.0
  Inlet     : u = 1 m/s,  T = 293.15 K
  Source    : dT/dn = 360 / 273.15 K/m
  Fluid     : nu = 0.02,  alpha_f = 0.02,  k_f = 1.0
  Solid     : alpha_s = 0.0625,  k_s = 5.0

Note: Our models use nu = 0.01 (Re differs by 2x), so velocity errors
will be larger but temperature distributions should be qualitatively correct.

Outputs:
  validate_fields_yz.png  - GT | PINN | FNO | PINO   y-z cross-section
  validate_errors.png     - per-field Rel-L2 bar chart
  validate_scatter.png    - scatter: predicted vs NVIDIA for each field
  validate_summary.txt    - human-readable error table

Example usage:
  python validate_nvidia.py \\
    --fluid_csv /path/to/threeFin_extend_fluid0.csv \\
    --solid_csv /path/to/threeFin_extend_solid0.csv \\
    --dataset   /path/to/dataset_combined.npz \\
    --fno_ckpt  /path/to/best_fno.pt \\
    --pino_ckpt /path/to/best_pino.pt \\
    --case laminar \\
    --out validate_output
"""

import os, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


#  CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--nvidia_dir", default=None,
                    help="Directory containing all NVIDIA CSVs")
parser.add_argument("--fluid_csv",  default=None,
                    help="Direct path to fluid CSV (overrides --nvidia_dir)")
parser.add_argument("--solid_csv",  default=None,
                    help="Direct path to solid CSV (overrides --nvidia_dir)")
parser.add_argument("--dataset",    default=(
    "/kaggle/input/datasets/subhommahalik/dataset-combined-npz/dataset_combined.npz"))
parser.add_argument("--fno_ckpt",   default=(
    "/kaggle/input/datasets/pravega/best-fno-pt/best_fno.pt"))
parser.add_argument("--pino_ckpt",  default=(
    "/kaggle/input/datasets/pravega/best-pino-pt/best_pino (1).pt"))
parser.add_argument("--out",        default="validate_output")
parser.add_argument("--sample",     type=int, default=-1,
                    help="Dataset sample index (-1 = auto-pick closest to NVIDIA params)")
parser.add_argument("--case",       default="laminar",
                    choices=["laminar", "turbulent"])
args, _ = parser.parse_known_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

CHANNEL_ORIGIN = np.array([-2.5, -0.5, -0.5])
NVIDIA_PARAMS  = dict(fin_height=0.4, fin_length=1.0, fin_thick=0.10,
                      grad_t=1.318, u_in=1.0, k_s=5.0)

FIELD_NAMES = ["u", "v", "w", "p", "theta_f", "theta_s"]
T_REF       = 273.15


#  LOAD NVIDIA CSV DATA
def load_csv_safe(path):
    """Load CSV with auto-detected delimiter, skipping comment rows."""
    for delim in [",", None, "\t", " "]:
        try:
            data = np.genfromtxt(path, delimiter=delim, names=True,
                                 dtype=np.float64, invalid_raise=False)
            if len(data) > 10:
                return data
        except Exception:
            continue
    raise RuntimeError(f"Cannot parse CSV: {path}")


def find_nvidia_csvs(nvidia_dir, case="laminar"):
    """Locate the fluid and solid CSV files."""
    if args.fluid_csv and args.solid_csv:
        return args.fluid_csv, args.solid_csv

    if args.fluid_csv:
        fluid_csv = args.fluid_csv
        solid_csv = None
        if nvidia_dir and os.path.isdir(nvidia_dir):
            prefix_s = ("threeFin_extend_solid0" if case == "laminar"
                        else "threeFin_extend_zeroEq_re500_solid")
            for f in os.listdir(nvidia_dir):
                if prefix_s.lower().replace("_", "") in f.lower().replace("_", ""):
                    solid_csv = os.path.join(nvidia_dir, f)
        return fluid_csv, solid_csv

    if args.solid_csv:
        solid_csv = args.solid_csv
        fluid_csv = None
        if nvidia_dir and os.path.isdir(nvidia_dir):
            prefix_f = ("threeFin_extend_fluid0" if case == "laminar"
                        else "threeFin_extend_zeroEq_re500_fluid")
            for f in os.listdir(nvidia_dir):
                if prefix_f.lower().replace("_", "") in f.lower().replace("_", ""):
                    fluid_csv = os.path.join(nvidia_dir, f)
        return fluid_csv, solid_csv

    if nvidia_dir is None:
        warnings.warn("No --nvidia_dir, --fluid_csv, or --solid_csv provided.")
        return None, None

    prefix_f = ("threeFin_extend_fluid0" if case == "laminar"
                else "threeFin_extend_zeroEq_re500_fluid")
    prefix_s = ("threeFin_extend_solid0" if case == "laminar"
                else "threeFin_extend_zeroEq_re500_solid")

    fluid_csv = solid_csv = None
    for f in os.listdir(nvidia_dir):
        fl = f.lower()
        if prefix_f.lower().replace("_", "") in fl.replace("_", ""):
            fluid_csv = os.path.join(nvidia_dir, f)
        if prefix_s.lower().replace("_", "") in fl.replace("_", ""):
            solid_csv = os.path.join(nvidia_dir, f)

    for ext in [".csv", ""]:
        if fluid_csv is None:
            p = os.path.join(nvidia_dir, prefix_f + ext)
            if os.path.exists(p):
                fluid_csv = p
        if solid_csv is None:
            p = os.path.join(nvidia_dir, prefix_s + ext)
            if os.path.exists(p):
                solid_csv = p

    return fluid_csv, solid_csv


def load_nvidia_reference(nvidia_dir, case="laminar"):
    """
    Load NVIDIA OpenFOAM reference data.

    Returns a dict with keys:
      x, y, z, u, v, w, p, theta_f  (fluid point cloud)
      xs, ys, zs, theta_s            (solid point cloud)
    Coordinates are shifted to the simulation frame.
    Temperature is non-dimensionalised: theta = T / 273.15 - 1.
    """
    fluid_csv, solid_csv = find_nvidia_csvs(nvidia_dir, case)
    ref = {}

    if fluid_csv and os.path.exists(fluid_csv):
        print(f"  Loading fluid CSV: {fluid_csv}")
        data = load_csv_safe(fluid_csv)
        cols = list(data.dtype.names)
        print(f"    Columns: {cols}")
        print(f"    Points : {len(data)}")

        def find_col(candidates, required=True):
            for c in candidates:
                for col in cols:
                    if (col.lower().replace(":", "").replace("_", "") ==
                            c.lower().replace(":", "").replace("_", "")):
                        return col
            if required:
                raise KeyError(f"Cannot find column matching {candidates} in {cols}")
            return None

        cx = find_col(["Points:0", "Points0", "x", "X"])
        cy = find_col(["Points:1", "Points1", "y", "Y"])
        cz = find_col(["Points:2", "Points2", "z", "Z"])
        cu = find_col(["U:0", "U0", "u", "Ux"])
        cv = find_col(["U:1", "U1", "v", "Uy"])
        cw = find_col(["U:2", "U2", "w", "Uz"])
        cp = find_col(["p_rgh", "p", "P", "prgh"])
        ct = find_col(["T", "Temperature", "theta_f"], required=False)

        ref["x"] = data[cx].astype(np.float32) + CHANNEL_ORIGIN[0]
        ref["y"] = data[cy].astype(np.float32) + CHANNEL_ORIGIN[1]
        ref["z"] = data[cz].astype(np.float32) + CHANNEL_ORIGIN[2]
        ref["u"] = data[cu].astype(np.float32)
        ref["v"] = data[cv].astype(np.float32)
        ref["w"] = data[cw].astype(np.float32)
        ref["p"] = data[cp].astype(np.float32)

        if ct:
            T_raw = data[ct].astype(np.float32)
            if T_raw.mean() > 100:
                ref["theta_f"] = T_raw / T_REF - 1.0
                print(f"    T_fluid: {T_raw.min():.1f}-{T_raw.max():.1f} K -> "
                      f"theta_f: {ref['theta_f'].min():.4f}-{ref['theta_f'].max():.4f}")
            else:
                ref["theta_f"] = T_raw
                print(f"    theta_f: {T_raw.min():.4f}-{T_raw.max():.4f} (already nondim)")

        n_orig = len(ref["x"])
        for k in list(ref.keys()):
            ref[k] = ref[k][::4]
        print(f"    Downsampled: {n_orig} -> {len(ref['x'])} points")
    else:
        warnings.warn(f"Fluid CSV not found in {nvidia_dir}")

    if solid_csv and os.path.exists(solid_csv):
        print(f"  Loading solid CSV: {solid_csv}")
        data = load_csv_safe(solid_csv)
        cols = list(data.dtype.names)
        print(f"    Columns: {cols}")

        def find_col_s(candidates, required=True):
            for c in candidates:
                for col in cols:
                    if (col.lower().replace(":", "").replace("_", "") ==
                            c.lower().replace(":", "").replace("_", "")):
                        return col
            if required:
                raise KeyError(f"Cannot find column {candidates} in {cols}")
            return None

        cx = find_col_s(["Points:0", "Points0", "x"])
        cy = find_col_s(["Points:1", "Points1", "y"])
        cz = find_col_s(["Points:2", "Points2", "z"])
        ct = find_col_s(["T", "Temperature", "theta_s"])

        ref["xs"] = data[cx].astype(np.float32) + CHANNEL_ORIGIN[0]
        ref["ys"] = data[cy].astype(np.float32) + CHANNEL_ORIGIN[1]
        ref["zs"] = data[cz].astype(np.float32) + CHANNEL_ORIGIN[2]

        T_raw = data[ct].astype(np.float32)
        if T_raw.mean() > 100:
            ref["theta_s"] = T_raw / T_REF - 1.0
            print(f"    T_solid: {T_raw.min():.1f}-{T_raw.max():.1f} K -> "
                  f"theta_s: {ref['theta_s'].min():.4f}-{ref['theta_s'].max():.4f}")
        else:
            ref["theta_s"] = T_raw

        n_orig = len(ref["xs"])
        for k in ["xs", "ys", "zs", "theta_s"]:
            ref[k] = ref[k][::4]
        print(f"    Downsampled: {n_orig} -> {len(ref['xs'])} points")
    else:
        warnings.warn(f"Solid CSV not found in {nvidia_dir}")

    return ref


#  FNO ARCHITECTURE  (must match fno.py / Pino.py)
WIDTH     = 48
N_LAYERS  = 4
N_MODES_X = 12
N_MODES_Y = 6
N_MODES_Z = 6
N_IN      = 10
N_OUT     = 6


class SpectralConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, mx, my, mz):
        super().__init__()
        self.out_ch = out_ch
        self.mx, self.my, self.mz = mx, my, mz
        s = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))
        self.w3 = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))
        self.w4 = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, my, mz, dtype=torch.cfloat))

    @staticmethod
    def cmul(x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x):
        B, C, NX, NY, NZ = x.shape
        mx, my, mz = self.mx, self.my, self.mz
        xf  = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        NZH = NZ // 2 + 1
        out = torch.zeros(B, self.out_ch, NX, NY, NZH,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :mx,  :my,  :mz] = self.cmul(xf[:, :, :mx,  :my,  :mz], self.w1)
        out[:, :, -mx:, :my,  :mz] = self.cmul(xf[:, :, -mx:, :my,  :mz], self.w2)
        out[:, :, :mx,  -my:, :mz] = self.cmul(xf[:, :, :mx,  -my:, :mz], self.w3)
        out[:, :, -mx:, -my:, :mz] = self.cmul(xf[:, :, -mx:, -my:, :mz], self.w4)
        return torch.fft.irfftn(out, s=(NX, NY, NZ), dim=(-3, -2, -1)).to(x.dtype)


class FNOLayer3D(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.lift   = nn.Conv3d(N_IN, WIDTH, 1)
        self.layers = nn.ModuleList([
            FNOLayer3D(WIDTH, N_MODES_X, N_MODES_Y, N_MODES_Z)
            for _ in range(N_LAYERS)])
        self.proj   = nn.Sequential(
            nn.Conv3d(WIDTH, WIDTH * 2, 1), nn.GELU(),
            nn.Conv3d(WIDTH * 2, N_OUT, 1))

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        return self.proj(x)


def build_model_input(params_vec, mask, grid_x, grid_y, grid_z, stats):
    """Build the 10-channel input tensor for FNO/PINO inference."""
    p_min  = stats["p_min"]
    p_max  = stats["p_max"]
    p_norm = (params_vec - p_min) / (p_max - p_min + 1e-8)

    NX, NY, NZ = len(grid_x), len(grid_y), len(grid_z)
    p_grid = np.zeros((6, NX, NY, NZ), dtype=np.float32)
    for i in range(6):
        p_grid[i] = p_norm[i]

    m_grid = mask[np.newaxis]
    xs     = np.linspace(0, 1, NX).astype(np.float32)[:, None, None] * np.ones((1, NY, NZ), np.float32)
    ys     = np.linspace(0, 1, NY).astype(np.float32)[None, :, None] * np.ones((NX, 1, NZ), np.float32)
    zs     = np.linspace(0, 1, NZ).astype(np.float32)[None, None, :] * np.ones((NX, NY, 1), np.float32)
    coords = np.stack([xs, ys, zs], axis=0)

    x_in = np.concatenate([p_grid, m_grid, coords], axis=0)
    return torch.tensor(x_in[np.newaxis]).float()


def denorm_fields(pred_norm, stats):
    """Denormalise (B, 6, NX, NY, NZ) predictions back to physical units."""
    mu  = torch.tensor(stats["f_mean"]).float()
    sig = torch.tensor(stats["f_std"]).float()
    return pred_norm * sig[None, :, None, None, None] + mu[None, :, None, None, None]


#  INTERPOLATION
def interp_grid_to_points(field_3d, grid_x, grid_y, grid_z, px, py, pz):
    """Trilinear interpolation from structured grid to scattered points."""
    interp = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), field_3d,
        method="linear", bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([px, py, pz])).astype(np.float32)


#  METRICS
def rel_l2(pred, true):
    """Relative L2 error (%), ignoring NaN."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 10:
        return np.nan
    p, t = pred[mask], true[mask]
    return np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-12) * 100.0


def mae(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 10:
        return np.nan
    return np.mean(np.abs(pred[mask] - true[mask]))


#  MAIN
def main():
    print("=" * 60)
    print("VALIDATE PINN / FNO / PINO vs NVIDIA OpenFOAM Reference")
    print("=" * 60)

    print(f"\n[1] Loading NVIDIA reference ({args.case}) from {args.nvidia_dir}")
    ref = load_nvidia_reference(args.nvidia_dir, args.case)

    has_fluid   = "u" in ref
    has_theta_f = "theta_f" in ref
    has_solid   = "theta_s" in ref
    print(f"  Has fluid fields: {has_fluid}  "
          f"Has theta_f: {has_theta_f}  Has theta_s: {has_solid}")

    if not has_fluid:
        print("ERROR: No fluid data loaded. Check --nvidia_dir / --fluid_csv path.")
        return

    print(f"\n[2] Loading dataset: {args.dataset}")
    raw        = np.load(args.dataset, allow_pickle=True)
    all_params = raw["params"]
    all_fields = raw["fields"]
    all_masks  = raw["geo_masks"]
    grid_x     = raw["grid_x"].astype(np.float32)
    grid_y     = raw["grid_y"].astype(np.float32)
    grid_z     = raw["grid_z"].astype(np.float32)
    NX, NY, NZ = len(grid_x), len(grid_y), len(grid_z)
    print(f"  Samples: {len(all_params)}, Grid: {NX}x{NY}x{NZ}")

    if args.sample >= 0:
        si = args.sample
    else:
        nv    = np.array([NVIDIA_PARAMS[k] for k in
                          ["fin_height", "fin_length", "fin_thick",
                           "grad_t", "u_in", "k_s"]], dtype=np.float32)
        dists = np.linalg.norm(all_params - nv[None, :], axis=1)
        si    = int(np.argmin(dists))

    pinn_params = all_params[si]
    pinn_fields = all_fields[si]
    pinn_mask   = all_masks[si]
    print(f"  Using sample {si}: params={pinn_params}")
    print(f"  NVIDIA target   : {list(NVIDIA_PARAMS.values())}")
    print(f"  Param distance  : {np.linalg.norm(pinn_params - np.array(list(NVIDIA_PARAMS.values()))):.4f}")

    models = {}
    for label, ckpt_path in [("FNO", args.fno_ckpt), ("PINO", args.pino_ckpt)]:
        if not os.path.exists(ckpt_path):
            print(f"\n  [{label}] checkpoint not found: {ckpt_path} - skipping")
            continue
        print(f"\n[3] Loading {label}: {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        stats = ckpt.get("stats", None)

        if stats is None:
            print(f"  WARNING: no 'stats' in checkpoint, computing from dataset")
            f_mean = np.zeros(6, np.float32)
            f_std  = np.ones(6,  np.float32)
            for c in range(6):
                vals = all_fields[:, c][(all_masks < 0.5) if c < 5 else (all_masks > 0.5)]
                if len(vals) > 10:
                    f_mean[c] = vals.mean()
                    f_std[c]  = max(vals.std(), 1e-6)
            stats = {"p_min": all_params.min(axis=0), "p_max": all_params.max(axis=0),
                     "f_mean": f_mean, "f_std": f_std}

        model = FNO3D().to(device)
        sd    = ckpt.get("model", ckpt)
        if isinstance(sd, dict) and "model" not in sd and "lift.weight" not in sd:
            sd = ckpt
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model.load_state_dict(sd)
        model.eval()
        print(f"  {label} loaded successfully")

        x_in = build_model_input(pinn_params, pinn_mask, grid_x, grid_y, grid_z, stats)
        with torch.no_grad():
            pred_norm = model(x_in.to(device)).cpu()
        models[label] = denorm_fields(pred_norm, stats)[0].numpy()

    print(f"\n[4] Interpolating predictions to NVIDIA point cloud ...")

    fluid_fields = ["u", "v", "w", "p"]
    if has_theta_f:
        fluid_fields.append("theta_f")

    results = {}
    for model_name, fields in [("PINN", pinn_fields)] + list(models.items()):
        results[model_name] = {}
        for fi, fname in enumerate(FIELD_NAMES):
            if fname == "theta_s":
                if has_solid:
                    results[model_name][fname] = interp_grid_to_points(
                        fields[fi], grid_x, grid_y, grid_z,
                        ref["xs"], ref["ys"], ref["zs"])
            else:
                if has_fluid:
                    results[model_name][fname] = interp_grid_to_points(
                        fields[fi], grid_x, grid_y, grid_z,
                        ref["x"], ref["y"], ref["z"])

    print(f"\n[5] Computing errors against NVIDIA reference ...\n")

    nvidia_vals = {f: ref[f] for f in fluid_fields}
    if has_solid:
        nvidia_vals["theta_s"] = ref["theta_s"]

    avail_fields = list(nvidia_vals.keys())
    all_errors   = {}
    all_maes     = {}

    print(f"  {'Model':<8}  {'Field':<10}  {'Rel-L2 %':>10}  {'MAE':>10}  {'Points':>8}")
    print(f"  {'-' * 52}")

    for mname in results:
        all_errors[mname] = {}
        all_maes[mname]   = {}
        for fname in avail_fields:
            if fname not in results[mname]:
                continue
            pred  = results[mname][fname]
            truth = nvidia_vals[fname]
            err   = rel_l2(pred, truth)
            m_    = mae(pred, truth)
            npts  = np.sum(np.isfinite(pred) & np.isfinite(truth))
            all_errors[mname][fname] = err
            all_maes[mname][fname]   = m_
            print(f"  {mname:<8}  {fname:<10}  {err:>9.2f}%  {m_:>10.4f}  {npts:>8d}")
        print()

    print("[6] Generating plots ...")

    model_names = list(results.keys())
    colors      = {"PINN": "mediumseagreen", "FNO": "steelblue", "PINO": "darkorange"}
    n_models    = len(model_names)
    n_fields    = len(avail_fields)
    x_pos       = np.arange(n_fields)
    w           = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mi, mname in enumerate(model_names):
        vals = [all_errors[mname].get(f, 0) for f in avail_fields]
        bars = ax.bar(x_pos + mi * w - (n_models - 1) * w / 2, vals, w,
                      label=mname, color=colors.get(mname, f"C{mi}"),
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(avail_fields, fontsize=11)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=11)
    ax.set_title("PINN vs FNO vs PINO - Error against NVIDIA OpenFOAM Reference",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{args.out}/validate_errors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {args.out}/validate_errors.png")

    from scipy.interpolate import griddata

    solid_per_x = pinn_mask.sum(axis=(1, 2))
    x_mid       = int(np.argmax(solid_per_x))
    x_val       = grid_x[x_mid]
    print(f"  y-z slice at x_idx={x_mid}, x={x_val:.3f}")

    YY, ZZ  = np.meshgrid(grid_y, grid_z, indexing="ij")
    x_tol   = (grid_x[1] - grid_x[0]) * 2
    nvidia_yz = {}

    if has_fluid:
        near = np.abs(ref["x"] - x_val) < x_tol
        if near.sum() > 50:
            pts_2d = np.column_stack([ref["y"][near], ref["z"][near]])
            for fname in ["u", "v", "w", "p"]:
                nvidia_yz[fname] = griddata(pts_2d, ref[fname][near], (YY, ZZ),
                                            method="linear", fill_value=np.nan).astype(np.float32)
            if has_theta_f:
                nvidia_yz["theta_f"] = griddata(pts_2d, ref["theta_f"][near], (YY, ZZ),
                                                method="linear", fill_value=np.nan).astype(np.float32)
        else:
            print(f"  WARNING: only {near.sum()} NVIDIA fluid pts near x={x_val:.2f}")

    if has_solid:
        near_s = np.abs(ref["xs"] - x_val) < x_tol
        if near_s.sum() > 10:
            pts_2d_s = np.column_stack([ref["ys"][near_s], ref["zs"][near_s]])
            nvidia_yz["theta_s"] = griddata(pts_2d_s, ref["theta_s"][near_s], (YY, ZZ),
                                            method="linear", fill_value=np.nan).astype(np.float32)

    plot_fields = [f for f in FIELD_NAMES if f in nvidia_yz]
    n_pf        = len(plot_fields)

    if n_pf > 0:
        col_labels = ["OpenFOAM"] + model_names
        n_cols     = len(col_labels)
        mask_yz    = pinn_mask[x_mid]
        extent     = [grid_z[0], grid_z[-1], grid_y[0], grid_y[-1]]

        def apply_mask(arr, fi):
            out = arr.copy().astype(np.float64)
            if fi < 5:
                out[mask_yz > 0.5] = np.nan
            else:
                out[mask_yz < 0.5] = np.nan
            return out

        cmap_f = plt.get_cmap("RdYlBu_r").copy()
        cmap_f.set_bad("white", 1.0)

        fig, axes = plt.subplots(n_pf, n_cols,
                                 figsize=(4 * n_cols, 2.8 * n_pf),
                                 constrained_layout=True)
        if n_pf == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Validation against NVIDIA OpenFOAM - y-z plane at x={x_val:.2f}\n"
            f"Note: our models use nu=0.01, NVIDIA uses nu=0.02 (Re differs by 2x)",
            fontsize=11, y=1.02)

        for ri, fname in enumerate(plot_fields):
            fi        = FIELD_NAMES.index(fname)
            nvd_slice = apply_mask(nvidia_yz[fname], fi)
            all_slices = [nvd_slice]
            model_slices = {}
            for mname, mfields in [("PINN", pinn_fields)] + list(models.items()):
                sl = apply_mask(mfields[fi, x_mid], fi)
                model_slices[mname] = sl
                all_slices.append(sl)

            all_vals = np.concatenate([s[np.isfinite(s)] for s in all_slices
                                       if np.any(np.isfinite(s))])
            if len(all_vals) == 0:
                continue
            vmin, vmax = np.nanpercentile(all_vals, [2, 98])
            if vmin == vmax:
                vmin -= 0.1; vmax += 0.1

            ax = axes[ri, 0]
            im = ax.imshow(nvd_slice, origin="lower", extent=extent,
                           aspect="auto", cmap=cmap_f, vmin=vmin, vmax=vmax,
                           interpolation="bilinear")
            ax.set_title(f"OpenFOAM: {fname}", fontsize=9, fontweight="bold")
            ax.set_xlabel("z (span)", fontsize=7)
            ax.set_ylabel("y (height)", fontsize=7)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, format="%.3g")

            for ci, mname in enumerate(model_names):
                ax = axes[ri, ci + 1]
                sl = model_slices[mname]
                im = ax.imshow(sl, origin="lower", extent=extent,
                               aspect="auto", cmap=cmap_f, vmin=vmin, vmax=vmax,
                               interpolation="bilinear")
                ax.set_title(f"{mname}: {fname}", fontsize=9)
                ax.set_xlabel("z (span)", fontsize=7)
                ax.set_ylabel("y (height)", fontsize=7)
                ax.tick_params(labelsize=6)
                plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, format="%.3g")

        plt.savefig(f"{args.out}/validate_fields_yz.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {args.out}/validate_fields_yz.png")

    scatter_fields = [f for f in avail_fields if f in results.get("PINN", {})]
    n_sf           = len(scatter_fields)

    if n_sf > 0 and len(model_names) > 0:
        fig, axes = plt.subplots(n_sf, len(model_names),
                                 figsize=(4.5 * len(model_names), 4 * n_sf),
                                 constrained_layout=True)
        if n_sf == 1:
            axes = axes[np.newaxis, :]
        if len(model_names) == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle("Predicted vs NVIDIA OpenFOAM (each dot = one reference point)",
                     fontsize=12, fontweight="bold", y=1.01)

        for ri, fname in enumerate(scatter_fields):
            truth = nvidia_vals[fname]
            for ci, mname in enumerate(model_names):
                if fname not in results[mname]:
                    continue
                pred = results[mname][fname]
                mask = np.isfinite(pred) & np.isfinite(truth)
                ax   = axes[ri, ci]
                idx  = np.where(mask)[0]
                if len(idx) > 5000:
                    idx = np.random.choice(idx, 5000, replace=False)
                ax.scatter(truth[idx], pred[idx], s=1, alpha=0.3,
                           c=colors.get(mname, "C0"))
                lims = [min(truth[idx].min(), pred[idx].min()),
                        max(truth[idx].max(), pred[idx].max())]
                ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
                ax.set_xlabel(f"NVIDIA {fname}", fontsize=8)
                ax.set_ylabel(f"{mname} {fname}", fontsize=8)
                err = all_errors[mname].get(fname, np.nan)
                ax.set_title(f"{mname} - {fname}  (L2={err:.1f}%)", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)
                ax.set_aspect("equal", adjustable="box")

        plt.savefig(f"{args.out}/validate_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {args.out}/validate_scatter.png")

    lines = []
    lines.append("=" * 62)
    lines.append(" PINN / FNO / PINO  vs  NVIDIA OpenFOAM ({})".format(args.case))
    lines.append("=" * 62)
    lines.append("")
    lines.append("NOTE: Our models use nu=0.01, NVIDIA uses nu=0.02.")
    lines.append("      This causes ~2x Reynolds number difference,")
    lines.append("      so velocity errors are expected to be large.")
    lines.append("      Temperature fields are more comparable.")
    lines.append("")
    lines.append(f"  Dataset sample   : {si}  params={list(pinn_params)}")
    lines.append(f"  NVIDIA target    : {list(NVIDIA_PARAMS.values())}")
    lines.append(f"  Grid             : {NX}x{NY}x{NZ}")
    lines.append("")

    header = f"  {'Field':<10}"
    for mn in model_names:
        header += f"  {mn + ' L2%':>12}  {mn + ' MAE':>10}"
    lines.append(header)
    lines.append(f"  {'-' * (10 + 24 * len(model_names))}")

    for fname in avail_fields:
        row = f"  {fname:<10}"
        for mn in model_names:
            e = all_errors[mn].get(fname, np.nan)
            m = all_maes[mn].get(fname, np.nan)
            row += f"  {e:>11.2f}%  {m:>10.4f}"
        lines.append(row)

    lines.append("")
    lines.append("  Winner per field (lowest Rel-L2):")
    for fname in avail_fields:
        errs   = {mn: all_errors[mn].get(fname, np.inf) for mn in model_names}
        winner = min(errs, key=errs.get)
        lines.append(f"    {fname:<10}  -> {winner} ({errs[winner]:.2f}%)")

    lines.append("")
    lines.append("=" * 62)

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(f"{args.out}/validate_summary.txt", "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved: {args.out}/validate_summary.txt")

    print(f"\n{'=' * 62}")
    print(f"All outputs in: {args.out}/")
    print("  validate_errors.png    - per-field error bar chart")
    print("  validate_fields_yz.png - y-z slice comparison")
    print("  validate_scatter.png   - scatter: predicted vs reference")
    print("  validate_summary.txt   - human-readable summary")


if __name__ == "__main__":
    main()
