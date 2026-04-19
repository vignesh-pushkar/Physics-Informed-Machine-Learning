"""
CHT PINN Dataset Generator
===========================
Generates a parametric dataset for the Conjugate Heat Transfer (CHT)
problem using Physics-Informed Neural Networks (PINNs).

Pipeline:
  Stage 1 — Train a base PINN on the nominal geometry configuration.
  Stage 2 — Fine-tune the base PINN for each LHS sample and evaluate
             on a structured grid, saving results to dataset.npz.

Parameters varied (Latin Hypercube Sampling, 125 samples):
  fin_height, fin_length, fin_thick, grad_t, u_in, k_s

Usage:
  python pinn_data.py [--skip-base] [--fast]
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn

#  CLI
parser = argparse.ArgumentParser()
parser.add_argument("--skip-base", action="store_true",
                    help="Skip base PINN training and load existing checkpoint")
parser.add_argument("--fast", action="store_true",
                    help="Ultra-fast mode: 250+500 epochs per sample")
args, _ = parser.parse_known_args()

FAST_MODE = args.fast or True

if FAST_MODE:
    BASE_EPOCHS_FLOW = 3000
    BASE_EPOCHS_TEMP = 5000
    FT_EPOCHS_FLOW   = 250
    FT_EPOCHS_TEMP   = 500
    N_COLLOC         = 150
    print("Fast mode: 250+500 fine-tune epochs per sample")
else:
    BASE_EPOCHS_FLOW = 4000
    BASE_EPOCHS_TEMP = 6000
    FT_EPOCHS_FLOW   = 500
    FT_EPOCHS_TEMP   = 800
    N_COLLOC         = 200
    print("Standard mode: 500+800 fine-tune epochs per sample")

#  DEVICE CHECK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("No GPU detected — this will be extremely slow. Aborting.")
    sys.exit(1)
print(f"Device: {device}  ({torch.cuda.get_device_name(0)})\n")


#  PATHS
if os.path.isdir("/kaggle/working"):
    OUT_DIR = "/kaggle/working/cht_output"
else:
    OUT_DIR = "cht_output"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_CKPT    = os.path.join(OUT_DIR, "base_checkpoint.pt")
DATASET_PATH = os.path.join(OUT_DIR, "dataset.npz")
RESUME_FILE  = os.path.join(OUT_DIR, "resume.json")

torch.manual_seed(0)
np.random.seed(0)


#  PARAMETERS AND LHS SAMPLING
PARAM_RANGES = {
    "fin_height": (0.05, 0.6,  0.4  ),
    "fin_length":  (0.5,  1.0,  1.0  ),
    "fin_thick":   (0.05, 0.15, 0.10 ),
    "grad_t":      (0.5,  2.0,  1.318),
    "u_in":        (0.5,  2.0,  1.0  ),
    "k_s":         (2.0,  10.0, 5.0  ),
}
PARAM_NAMES = list(PARAM_RANGES.keys())
N_PARAMS    = len(PARAM_NAMES)
BASE_PARAMS = {k: v[2] for k, v in PARAM_RANGES.items()}


def lhs_sample(n=125, seed=42):
    """Latin Hypercube Sampling over the parameter space."""
    rng = np.random.default_rng(seed)
    S   = np.zeros((n, N_PARAMS))
    for j in range(N_PARAMS):
        perm    = rng.permutation(n)
        S[:, j] = (perm + rng.random(n)) / n
    out = np.zeros_like(S)
    for j, (lo, hi, _) in enumerate(PARAM_RANGES.values()):
        out[:, j] = lo + (hi - lo) * S[:, j]
    return out.astype(np.float32)


ALL_PARAMS = lhs_sample(125)
print(f"LHS grid: {ALL_PARAMS.shape}")


#  GEOMETRY
CH_X0, CH_X1 = -2.5,  2.5
CH_Y0, CH_Y1 = -0.5,  0.5
CH_Z0, CH_Z1 = -0.5,  0.5


def build_geometry(p):
    fl, fh, ft = p["fin_length"], p["fin_height"], p["fin_thick"] / 2.0
    bx0, bx1   = -fl, 0.0
    by0, by1   = -0.5, -0.3
    bz0, bz1   = -0.3,  0.3
    fy0, fy1   = by1, by1 + max(fh, 0.01)
    fins        = [(zc - ft, zc + ft) for zc in [-0.25, 0.0, 0.25]]
    src_x0      = bx0 + 0.3 * fl
    src_x1      = bx0 + 0.7 * fl
    return dict(bx0=bx0, bx1=bx1, by0=by0, by1=by1, bz0=bz0, bz1=bz1,
                fx0=bx0, fx1=bx1, fy0=fy0, fy1=fy1, fins=fins,
                src_x0=src_x0, src_x1=src_x1, src_z0=-0.1, src_z1=0.1, src_y=by0)


def in_fin_np(x, y, z, g):
    m = np.zeros(len(x), bool)
    for z0, z1 in g["fins"]:
        m |= ((x >= g["fx0"]) & (x <= g["fx1"]) & (y >= g["fy0"]) & (y <= g["fy1"])
              & (z >= z0) & (z <= z1))
    return m


def in_base_np(x, y, z, g):
    return ((x >= g["bx0"]) & (x <= g["bx1"]) & (y >= g["by0"]) & (y <= g["by1"])
            & (z >= g["bz0"]) & (z <= g["bz1"]))


def in_solid_np(x, y, z, g):
    return in_base_np(x, y, z, g) | in_fin_np(x, y, z, g)


def in_fluid_np(x, y, z, g):
    ch = ((x >= CH_X0) & (x <= CH_X1) & (y >= CH_Y0) & (y <= CH_Y1)
          & (z >= CH_Z0) & (z <= CH_Z1))
    return ch & ~in_solid_np(x, y, z, g)


#  NETWORKS
class MLP(nn.Module):
    """Fully-connected network with Tanh activations and Xavier initialisation."""

    def __init__(self, in_d, out_d, h=256, nl=6):
        super().__init__()
        L = [nn.Linear(in_d, h), nn.Tanh()]
        for _ in range(nl - 1):
            L += [nn.Linear(h, h), nn.Tanh()]
        L.append(nn.Linear(h, out_d))
        self.net = nn.Sequential(*L)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def make_nets():
    return (MLP(3, 4).to(device),
            MLP(3, 1).to(device),
            MLP(3, 1).to(device))


def G(out, coords):
    return torch.autograd.grad(out, coords,
                               grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]


def t(arr):
    return torch.tensor(np.asarray(arr, np.float32), device=device, requires_grad=True)

#  COLLOCATION POINT SAMPLERS
def _smp(fn, bbox, N, ov=6):
    x0, x1, y0, y1, z0, z1 = bbox
    pts = []
    while len(pts) < N:
        x = np.random.uniform(x0, x1, N * ov).astype(np.float32)
        y = np.random.uniform(y0, y1, N * ov).astype(np.float32)
        z = np.random.uniform(z0, z1, N * ov).astype(np.float32)
        m = fn(x, y, z)
        for p in zip(x[m], y[m], z[m]):
            pts.append(p)
            if len(pts) >= N:
                break
    return t(pts[:N])


def sf(N, g):
    return _smp(lambda x, y, z: in_fluid_np(x, y, z, g),
                (CH_X0, CH_X1, CH_Y0, CH_Y1, CH_Z0, CH_Z1), N)


def sfh(N, g):
    return _smp(lambda x, y, z: in_fluid_np(x, y, z, g) & (x >= g["bx0"] - 0.1) & (x <= 0.5),
                (g["bx0"] - 0.1, 0.5, CH_Y0, CH_Y1, CH_Z0, CH_Z1), N)


def ss(N, g):
    fy1 = max(g["fy1"], g["by1"] + 0.01)
    return _smp(lambda x, y, z: in_solid_np(x, y, z, g),
                (g["bx0"], g["bx1"], g["by0"], fy1, g["bz0"], g["bz1"]), N)


def sbt(N, g):
    pts = []
    while len(pts) < N:
        x  = np.random.uniform(g["bx0"], g["bx1"], N * 4).astype(np.float32)
        z  = np.random.uniform(g["bz0"], g["bz1"], N * 4).astype(np.float32)
        y  = np.full(len(x), g["by1"], np.float32)
        ok = ~in_fin_np(x, y, z, g)
        for p in zip(x[ok], z[ok]):
            pts.append(p)
            if len(pts) >= N:
                break
    a = np.array(pts[:N], np.float32)
    return t(np.column_stack([a[:, 0], np.full(N, g["by1"], np.float32), a[:, 1]]))


def sfi(M, g):
    sides, tips = [], []
    for z0, z1 in g["fins"]:
        xf = np.random.uniform(g["fx0"], g["fx1"], M).astype(np.float32)
        yf = np.random.uniform(g["fy0"], g["fy1"], M).astype(np.float32)
        sides += [np.column_stack([xf, yf, np.full(M, z0)]),
                  np.column_stack([xf, yf, np.full(M, z1)])]
        xt = np.random.uniform(g["fx0"], g["fx1"], M).astype(np.float32)
        zt = np.random.uniform(z0, z1, M).astype(np.float32)
        tips.append(np.column_stack([xt, np.full(M, g["fy1"]), zt]))
    return t(np.vstack(sides)), t(np.vstack(tips))


def ssrc(N, g):
    x = np.random.uniform(g["src_x0"], g["src_x1"], N).astype(np.float32)
    z = np.random.uniform(g["src_z0"], g["src_z1"], N).astype(np.float32)
    return t(np.column_stack([x, np.full(N, g["src_y"], np.float32), z]))


def spl(N, xp, g):
    pts = []
    while len(pts) < N:
        y = np.random.uniform(CH_Y0, CH_Y1, N * 4).astype(np.float32)
        z = np.random.uniform(CH_Z0, CH_Z1, N * 4).astype(np.float32)
        x = np.full(len(y), xp, np.float32)
        m = in_fluid_np(x, y, z, g)
        for p in zip(x[m], y[m], z[m]):
            pts.append(p)
            if len(pts) >= N:
                break
    return t(pts[:N])


#  PDE RESIDUALS
def ns(xyz, fn, nu=0.01):
    """Incompressible Navier-Stokes residuals (continuity + 3 momentum)."""
    o = fn(xyz)
    u, v, w, p = o[:, 0:1], o[:, 1:2], o[:, 2:3], o[:, 3:4]
    Du = G(u, xyz); Dv = G(v, xyz); Dw = G(w, xyz); Dp = G(p, xyz)
    ux, uy, uz = Du[:, 0:1], Du[:, 1:2], Du[:, 2:3]
    vx, vy, vz = Dv[:, 0:1], Dv[:, 1:2], Dv[:, 2:3]
    wx, wy, wz = Dw[:, 0:1], Dw[:, 1:2], Dw[:, 2:3]
    lap = lambda qx, qy, qz: (G(qx, xyz)[:, 0:1] + G(qy, xyz)[:, 1:2] + G(qz, xyz)[:, 2:3])
    cont = ux + vy + wz
    mx_  = u * ux + v * uy + w * uz + Dp[:, 0:1] - nu * lap(ux, uy, uz)
    my_  = u * vx + v * vy + w * vz + Dp[:, 1:2] - nu * lap(vx, vy, vz)
    mz_  = u * wx + v * wy + w * wz + Dp[:, 2:3] - nu * lap(wx, wy, wz)
    return cont, mx_, my_, mz_


def fen(xyz, fn_net, ft_net, af=0.02):
    """Fluid energy equation residual: u.grad(theta_f) - alpha_f * lap(theta_f)."""
    with torch.no_grad():
        fo = fn_net(xyz)
        u, v, w = fo[:, 0:1], fo[:, 1:2], fo[:, 2:3]
    th = ft_net(xyz)
    Dt = G(th, xyz)
    tx, ty, tz = Dt[:, 0:1], Dt[:, 1:2], Dt[:, 2:3]
    lap = G(tx, xyz)[:, 0:1] + G(ty, xyz)[:, 1:2] + G(tz, xyz)[:, 2:3]
    return u * tx + v * ty + w * tz - af * lap


def sen(xyz, st_net):
    """Solid Laplace equation residual: lap(theta_s) = 0."""
    th = st_net(xyz)
    Dt = G(th, xyz)
    tx, ty, tz = Dt[:, 0:1], Dt[:, 1:2], Dt[:, 2:3]
    return G(tx, xyz)[:, 0:1] + G(ty, xyz)[:, 1:2] + G(tz, xyz)[:, 2:3]


#  BOUNDARY CONDITIONS
INLET_T = 293.15 / 273.15 - 1.0


def bc_flow(fn_net, g, p):
    u_in = p["u_in"]
    N    = 200
    L    = []

    def noslip(arr):
        o = fn_net(t(arr))
        return torch.mean(o[:, 0:1] ** 2 + o[:, 1:2] ** 2 + o[:, 2:3] ** 2)

    yi = np.random.uniform(CH_Y0, CH_Y1, N * 2).astype(np.float32)
    zi = np.random.uniform(CH_Z0, CH_Z1, N * 2).astype(np.float32)
    xi = np.full(len(yi), CH_X0, np.float32)
    m  = in_fluid_np(xi, yi, zi, g)
    ti_ = t(np.column_stack([xi[m][:N], yi[m][:N], zi[m][:N]]))
    oi  = fn_net(ti_)
    L.append(torch.mean((oi[:, 0:1] - u_in) ** 2 + oi[:, 1:2] ** 2 + oi[:, 2:3] ** 2))

    yo = np.random.uniform(CH_Y0, CH_Y1, N * 2).astype(np.float32)
    zo = np.random.uniform(CH_Z0, CH_Z1, N * 2).astype(np.float32)
    xo = np.full(len(yo), CH_X1, np.float32)
    mo = in_fluid_np(xo, yo, zo, g)
    to_ = t(np.column_stack([xo[mo][:N], yo[mo][:N], zo[mo][:N]]))
    L.append(torch.mean(fn_net(to_)[:, 3:4] ** 2))

    xw = np.random.uniform(CH_X0, CH_X1, N).astype(np.float32)
    zw = np.random.uniform(CH_Z0, CH_Z1, N).astype(np.float32)
    yw = np.random.uniform(CH_Y0, CH_Y1, N).astype(np.float32)
    L += [noslip(np.column_stack([xw, np.full(N, CH_Y1), zw])),
          noslip(np.column_stack([xw, yw, np.full(N, CH_Z0)])),
          noslip(np.column_stack([xw, yw, np.full(N, CH_Z1)]))]

    xb = np.random.uniform(CH_X0, CH_X1, N * 4).astype(np.float32)
    zb = np.random.uniform(CH_Z0, CH_Z1, N * 4).astype(np.float32)
    yb = np.full(len(xb), CH_Y0, np.float32)
    mb = ~in_solid_np(xb, yb, zb, g)
    L.append(noslip(np.column_stack([xb[mb][:N], yb[mb][:N], zb[mb][:N]])))

    for xv in [g["bx0"], g["bx1"]]:
        yv = np.random.uniform(g["by0"], g["by1"], N // 2).astype(np.float32)
        zv = np.random.uniform(g["bz0"], g["bz1"], N // 2).astype(np.float32)
        L.append(noslip(np.column_stack([np.full(N // 2, xv), yv, zv])))
    for zv in [g["bz0"], g["bz1"]]:
        xv = np.random.uniform(g["bx0"], g["bx1"], N // 2).astype(np.float32)
        yv = np.random.uniform(g["by0"], g["by1"], N // 2).astype(np.float32)
        L.append(noslip(np.column_stack([xv, yv, np.full(N // 2, zv)])))

    if g["fy1"] > g["fy0"] + 0.01:
        for z0, z1 in g["fins"]:
            xf = np.random.uniform(g["fx0"], g["fx1"], N // 3).astype(np.float32)
            yf = np.random.uniform(g["fy0"], g["fy1"], N // 3).astype(np.float32)
            L += [noslip(np.column_stack([xf, yf, np.full(N // 3, z0)])),
                  noslip(np.column_stack([xf, yf, np.full(N // 3, z1)]))]

    for xp in [CH_X0 + 0.5, 0.0, CH_X1 - 0.5]:
        try:
            xp_ = spl(80, xp, g)
            L.append(40.0 * (torch.mean(fn_net(xp_)[:, 0:1]) - u_in) ** 2)
        except Exception:
            pass

    return sum(L)


def bc_temp(fn_net, ft_net, st_net, g, p):
    gT = p["grad_t"]
    kf = p.get("k_f", 1.0)
    ks = p["k_s"]
    N  = 150
    L  = []

    yi = np.random.uniform(CH_Y0, CH_Y1, N * 2).astype(np.float32)
    zi = np.random.uniform(CH_Z0, CH_Z1, N * 2).astype(np.float32)
    xi = np.full(len(yi), CH_X0, np.float32)
    m  = in_fluid_np(xi, yi, zi, g)
    L.append(torch.mean((ft_net(t(np.column_stack([xi[m][:N], yi[m][:N], zi[m][:N]]))) - INLET_T) ** 2))

    yo = np.random.uniform(CH_Y0, CH_Y1, N * 2).astype(np.float32)
    zo = np.random.uniform(CH_Z0, CH_Z1, N * 2).astype(np.float32)
    xo = np.full(len(yo), CH_X1, np.float32)
    mo = in_fluid_np(xo, yo, zo, g)
    to_ = t(np.column_stack([xo[mo][:N], yo[mo][:N], zo[mo][:N]]))
    L.append(torch.mean(G(ft_net(to_), to_)[:, 0:1] ** 2))

    def af(arr, ni):
        tt = t(arr)
        return torch.mean(G(ft_net(tt), tt)[:, ni:ni + 1] ** 2)

    xw = np.random.uniform(CH_X0, CH_X1, N).astype(np.float32)
    zw = np.random.uniform(CH_Z0, CH_Z1, N).astype(np.float32)
    yw = np.random.uniform(CH_Y0, CH_Y1, N).astype(np.float32)
    L += [af(np.column_stack([xw, np.full(N, CH_Y1), zw]), 1),
          af(np.column_stack([xw, yw, np.full(N, CH_Z0)]), 2),
          af(np.column_stack([xw, yw, np.full(N, CH_Z1)]), 2)]

    xs = ssrc(N, g)
    L.append(torch.mean((G(st_net(xs), xs)[:, 1:2] - (-gT)) ** 2))

    xab = np.random.uniform(g["bx0"], g["bx1"], N * 4).astype(np.float32)
    zab = np.random.uniform(g["bz0"], g["bz1"], N * 4).astype(np.float32)
    ins = ((xab >= g["src_x0"]) & (xab <= g["src_x1"])
           & (zab >= g["src_z0"]) & (zab <= g["src_z1"]))
    xab, zab = xab[~ins][:N], zab[~ins][:N]
    if len(xab) > 10:
        tab = t(np.column_stack([xab, np.full(len(xab), g["src_y"]), zab]))
        L.append(torch.mean(G(st_net(tab), tab)[:, 1:2] ** 2))

    def afs(arr, ni):
        tt = t(arr)
        return torch.mean(G(st_net(tt), tt)[:, ni:ni + 1] ** 2)

    for xv in [g["bx0"], g["bx1"]]:
        yv = np.random.uniform(g["by0"], g["by1"], N // 2).astype(np.float32)
        zv = np.random.uniform(g["bz0"], g["bz1"], N // 2).astype(np.float32)
        L.append(afs(np.column_stack([np.full(N // 2, xv), yv, zv]), 0))
    for zv in [g["bz0"], g["bz1"]]:
        xv = np.random.uniform(g["bx0"], g["bx1"], N // 2).astype(np.float32)
        yv = np.random.uniform(g["by0"], g["by1"], N // 2).astype(np.float32)
        L.append(afs(np.column_stack([xv, yv, np.full(N // 2, zv)]), 2))

    return sum(L)


def bc_intf(ft_net, st_net, g, p):
    kf = p.get("k_f", 1.0)
    ks = p["k_s"]

    def pair(pts, ni):
        xf = pts.clone().requires_grad_(True)
        xs = pts.clone().requires_grad_(True)
        tf = ft_net(xf)
        ts = st_net(xs)
        return (torch.mean((tf - ts) ** 2),
                torch.mean((kf * G(tf, xf)[:, ni:ni + 1] - ks * G(ts, xs)[:, ni:ni + 1]) ** 2))

    lTb, lQb = pair(sbt(100, g), 1)
    if g["fy1"] > g["fy0"] + 0.01:
        sp, tp       = sfi(30, g)
        lTs, lQs     = pair(sp, 2)
        lTt, lQt     = pair(tp, 1)
        return (lTb + lQb), (lTs + lQs + lTt + lQt)
    else:
        return (lTb + lQb), torch.tensor(0.0, device=device)


#  TRAINING
def train(fn_net, ft_net, st_net, g, p, efl, ete, verbose=False, tag=""):
    nu_f = p.get("nu_f", 0.01)
    af_  = p.get("alpha_f", 0.02)
    NC   = N_COLLOC

    opt = torch.optim.Adam(fn_net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9995)
    if verbose:
        print(f"  [{tag}] Flow {efl} epochs")
    for ep in range(efl):
        opt.zero_grad()
        xyz        = torch.cat([sf(NC, g), sfh(NC, g)])
        rc, rx, ry, rz = ns(xyz, fn_net, nu_f)
        lp         = (rc ** 2 + rx ** 2 + ry ** 2 + rz ** 2).mean()
        lb         = bc_flow(fn_net, g, p)
        (lp + 4.0 * lb).backward()
        torch.nn.utils.clip_grad_norm_(fn_net.parameters(), 1.0)
        opt.step(); sch.step()
        if verbose and ep % (efl // 4) == 0:
            print(f"    [{ep:5d}] pde={lp.item():.2e}  bc={lb.item():.2e}")
    for param in fn_net.parameters():
        param.requires_grad_(False)

    opt2 = torch.optim.Adam(list(ft_net.parameters()) + list(st_net.parameters()), lr=1e-3)
    sch2 = torch.optim.lr_scheduler.ExponentialLR(opt2, 0.9995)
    if verbose:
        print(f"  [{tag}] Temp {ete} epochs")
    for ep in range(ete):
        opt2.zero_grad()
        xyz_f = torch.cat([sf(NC, g), sfh(NC, g)])
        xyz_s = ss(NC, g)
        lf    = torch.mean(fen(xyz_f, fn_net, ft_net, af_) ** 2)
        ls    = torch.mean(sen(xyz_s, st_net) ** 2)
        lb    = bc_temp(fn_net, ft_net, st_net, g, p)
        li_b, li_f = bc_intf(ft_net, st_net, g, p)
        loss  = lf + ls + 8.0 * lb + 4.0 * li_b + 2.0 * li_f
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(ft_net.parameters()) + list(st_net.parameters()), 1.0)
        opt2.step(); sch2.step()
        if verbose and ep % (ete // 4) == 0:
            print(f"    [{ep:5d}] f={lf.item():.2e}  s={ls.item():.2e}  bc={lb.item():.2e}")

    for param in fn_net.parameters():
        param.requires_grad_(True)


#  GRID EVALUATION
NX, NY, NZ = 50, 20, 20


def eval_grid(fn_net, ft_net, st_net, g):
    """Evaluate the trained PINN on a structured grid and return field arrays."""
    xl = np.linspace(CH_X0, CH_X1, NX, dtype=np.float32)
    yl = np.linspace(CH_Y0, CH_Y1, NY, dtype=np.float32)
    zl = np.linspace(CH_Z0, CH_Z1, NZ, dtype=np.float32)
    XX, YY, ZZ = np.meshgrid(xl, yl, zl, indexing="ij")
    flat = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    ms   = in_solid_np(flat[:, 0], flat[:, 1], flat[:, 2], g)
    mfl  = ~ms
    xt   = torch.tensor(flat, device=device)
    with torch.no_grad():
        uvwp = np.zeros((len(flat), 4), np.float32)
        if mfl.any():
            uvwp[mfl] = fn_net(xt[mfl]).cpu().numpy()
        tf_ = np.zeros(len(flat), np.float32)
        ts_ = np.zeros(len(flat), np.float32)
        if mfl.any():
            tf_[mfl] = ft_net(xt[mfl]).cpu().numpy().squeeze()
        if ms.any():
            ts_[ms]  = st_net(xt[ms]).cpu().numpy().squeeze()
    fields = np.stack([
        uvwp[:, 0].reshape(NX, NY, NZ), uvwp[:, 1].reshape(NX, NY, NZ),
        uvwp[:, 2].reshape(NX, NY, NZ), uvwp[:, 3].reshape(NX, NY, NZ),
        tf_.reshape(NX, NY, NZ),        ts_.reshape(NX, NY, NZ),
    ], axis=0)
    return fields, ms.reshape(NX, NY, NZ).astype(np.float32), xl, yl, zl


#  STAGE 1: BASE PINN
if not args.skip_base:
    print("=" * 60)
    print("STAGE 1: Base PINN training")
    print("=" * 60)
    bp = dict(fin_height=BASE_PARAMS["fin_height"],
              fin_length=BASE_PARAMS["fin_length"],
              fin_thick =BASE_PARAMS["fin_thick"],
              grad_t    =BASE_PARAMS["grad_t"],
              u_in      =BASE_PARAMS["u_in"],
              k_s       =BASE_PARAMS["k_s"],
              k_f=1.0, alpha_f=0.02, nu_f=0.01)
    bg = build_geometry(bp)
    fn, ft, st = make_nets()
    t0b = time.time()
    train(fn, ft, st, bg, bp, BASE_EPOCHS_FLOW, BASE_EPOCHS_TEMP, verbose=True, tag="BASE")
    print(f"Base PINN done in {(time.time() - t0b) / 60:.1f} min")
    torch.save({"flow": fn.state_dict(), "fluid_temp": ft.state_dict(),
                "solid_temp": st.state_dict()}, BASE_CKPT)
    print(f"Saved -> {BASE_CKPT}")
else:
    print(f"[SKIP BASE] loading {BASE_CKPT}")


#  STAGE 2: DATASET GENERATION
print("\n" + "=" * 60)
print("STAGE 2: Dataset generation")
print("=" * 60)

ckpt_data = torch.load(BASE_CKPT, map_location=device)

start_idx       = 0
all_params_done = []
all_fields_done = []
all_masks_done  = []

if os.path.exists(RESUME_FILE):
    with open(RESUME_FILE) as f:
        rs = json.load(f)
    start_idx = rs["next_idx"]
    print(f"Resuming from sample {start_idx}/125")

if start_idx > 0 and os.path.exists(DATASET_PATH):
    partial = np.load(DATASET_PATH)
    if "fields" in partial:
        all_params_done = list(partial["params"][:start_idx])
        all_fields_done = list(partial["fields"][:start_idx])
        all_masks_done  = list(partial["geo_masks"][:start_idx])
        print(f"  Loaded {len(all_fields_done)} previously completed samples")

t_total          = time.time()
times_per_sample = []

for idx in range(start_idx, 125):
    row   = ALL_PARAMS[idx]
    p_smp = dict(fin_height=float(row[0]), fin_length=float(row[1]),
                 fin_thick =float(row[2]), grad_t    =float(row[3]),
                 u_in      =float(row[4]), k_s       =float(row[5]),
                 k_f=1.0, alpha_f=0.02, nu_f=0.01)
    g_smp = build_geometry(p_smp)

    fn, ft, st = make_nets()
    fn.load_state_dict(ckpt_data["flow"])
    ft.load_state_dict(ckpt_data["fluid_temp"])
    st.load_state_dict(ckpt_data["solid_temp"])

    t1 = time.time()
    train(fn, ft, st, g_smp, p_smp, FT_EPOCHS_FLOW, FT_EPOCHS_TEMP,
          verbose=False, tag=f"s{idx:03d}")
    fields, mask, xl, yl, zl = eval_grid(fn, ft, st, g_smp)
    elapsed = time.time() - t1
    times_per_sample.append(elapsed)

    all_params_done.append(row)
    all_fields_done.append(fields)
    all_masks_done.append(mask)

    done = idx + 1
    avg  = np.mean(times_per_sample[-10:])
    eta  = avg * (125 - done) / 3600
    print(f"  {done:3d}/125  {elapsed:.0f}s  avg={avg:.0f}s  ETA~{eta:.1f}h  "
          f"| gt={p_smp['grad_t']:.2f}  u={p_smp['u_in']:.2f}  "
          f"ks={p_smp['k_s']:.1f}  fh={p_smp['fin_height']:.2f}")

    if done % 10 == 0 or done == 125:
        np.savez_compressed(
            DATASET_PATH,
            params      = np.array(all_params_done, np.float32),
            fields      = np.array(all_fields_done, np.float32),
            geo_masks   = np.array(all_masks_done,  np.float32),
            grid_x      = xl, grid_y=yl, grid_z=zl,
            param_names = np.array(PARAM_NAMES),
            field_names = np.array(["u", "v", "w", "p", "theta_f", "theta_s"]),
        )
        with open(RESUME_FILE, "w") as f:
            json.dump({"next_idx": done}, f)
        sz = os.path.getsize(DATASET_PATH) / 1e6
        print(f"  Checkpoint: {done} samples -> dataset.npz ({sz:.0f} MB)")

if os.path.exists(RESUME_FILE):
    os.remove(RESUME_FILE)

total_h = (time.time() - t_total) / 3600
print(f"\n{'=' * 60}")
print(f"Dataset complete: {DATASET_PATH}")
print(f"  Shape: params={np.array(all_params_done).shape}  "
      f"fields={np.array(all_fields_done).shape}")
print(f"  Total time: {total_h:.1f} hrs")
print(f"  File size:  {os.path.getsize(DATASET_PATH) / 1e6:.1f} MB")
print("=" * 60)