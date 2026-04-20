###########vpinns-1

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def exact_solution_np(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.5, x**2, (x - 1.0)**2)


def generate_data(N_interior: int = 2000,
                  N_interface: int = 300,
                  N_boundary:  int = 500,
                  seed:        int = 42) -> dict:

    rng = np.random.default_rng(seed)

    # --- Interior: Omega_1 ---
    x1 = rng.uniform(0.0, 0.5, N_interior)
    y1 = rng.uniform(0.0, 1.0, N_interior)
    interior_1 = torch.tensor(np.stack([x1, y1], axis=1), dtype=torch.float32)

    # --- Interior: Omega_2 ---
    x2 = rng.uniform(0.5, 1.0, N_interior)
    y2 = rng.uniform(0.0, 1.0, N_interior)
    interior_2 = torch.tensor(np.stack([x2, y2], axis=1), dtype=torch.float32)

    # Interface: Gamma (x = 0.5)
    y_int = rng.uniform(0.0, 1.0, N_interface)
    x_int = np.full_like(y_int, 0.5)
    interface_pts = torch.tensor(np.stack([x_int, y_int], axis=1), dtype=torch.float32)

    # Dirichlet Boundary (4 edges)
    n_each = N_boundary // 4
    b_x0  = np.stack([np.zeros(n_each),                         rng.uniform(0, 1, n_each)], axis=1)  
    b_x1  = np.stack([np.ones(n_each),                          rng.uniform(0, 1, n_each)], axis=1)  
    b_y0  = np.stack([rng.uniform(0, 1, n_each),                np.zeros(n_each)         ], axis=1)  
    b_y1  = np.stack([rng.uniform(0, 1, n_each),                np.ones(n_each)          ], axis=1)  
    bndry_xy  = np.concatenate([b_x0, b_x1, b_y0, b_y1], axis=0)
    bndry_u   = exact_solution_np(bndry_xy[:, 0])              

    boundary_pts = torch.tensor(bndry_xy, dtype=torch.float32)
    boundary_u   = torch.tensor(bndry_u,  dtype=torch.float32).unsqueeze(1)  

    return {
        "interior_1":    interior_1,
        "interior_2":    interior_2,
        "interface_pts": interface_pts,
        "boundary_pts":  boundary_pts,
        "boundary_u":    boundary_u,
    }


# 2. Network Architecture

class VPINN(nn.Module):
    def __init__(self, layers: list = [2, 50, 50, 50, 50, 1]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.linears[:-1]:
            x = self.activation(lin(x))
        return self.linears[-1](x)


# 3. Test Function Generators


class TestFunctions:

    #Legendre helpers

    @staticmethod
    def _legendre(t: torch.Tensor, k: int) -> torch.Tensor:
        """P_k(t) on t in [-1, 1]"""
        if k == 0: return torch.ones_like(t)
        if k == 1: return t
        if k == 2: return 0.5  * (3*t**2 - 1)
        if k == 3: return 0.5  * (5*t**3 - 3*t)
        if k == 4: return 0.125 * (35*t**4 - 30*t**2 + 3)
        raise ValueError(f"Legendre order {k} not supported (max=4)")

    @staticmethod
    def _dlegendre(t: torch.Tensor, k: int) -> torch.Tensor:
        """dP_k/dt on t in [-1, 1]"""
        if k == 0: return torch.zeros_like(t)
        if k == 1: return torch.ones_like(t)
        if k == 2: return 3*t
        if k == 3: return 0.5  * (15*t**2 - 3)
        if k == 4: return 0.125 * (140*t**3 - 60*t)
        raise ValueError(f"Legendre order {k} not supported (max=4)")

    #Main interface

    @staticmethod
    def get_test_fn(coords: torch.Tensor,
                    m: int, n: int,
                    fn_type: str = "trig"):
        """
        Returns (v, dv/dx, dv/dy) for mode (m, n).

        trig     : v = sin(m*pi*x) * sin(n*pi*y)
                   Vanishes on all four edges for m, n >= 1.

        legendre : v = [x(1-x) * P_m(2x-1)] * [y(1-y) * P_n(2y-1)]
                   Bubble-Legendre — vanishes on all four edges for any m, n.
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]

        if fn_type == "trig":
            v   = torch.sin(m * torch.pi * x)  * torch.sin(n * torch.pi * y)
            v_x = m * torch.pi * torch.cos(m * torch.pi * x) * torch.sin(n * torch.pi * y)
            v_y = n * torch.pi * torch.sin(m * torch.pi * x) * torch.cos(n * torch.pi * y)

        elif fn_type == "legendre":
            tx, ty = 2*x - 1,  2*y - 1
            Pm, Pn   = TestFunctions._legendre(tx, m),  TestFunctions._legendre(ty, n)
            dPm, dPn = TestFunctions._dlegendre(tx, m), TestFunctions._dlegendre(ty, n)

            wx,  wy  = x*(1 - x),   y*(1 - y)  
            dwx, dwy = 1 - 2*x,     1 - 2*y    

            v   = wx * Pm * wy * Pn
            v_x = (dwx * Pm + wx * dPm * 2) * (wy * Pn)
            v_y = (wx * Pm) * (dwy * Pn + wy * dPn * 2)

        else:
            raise ValueError(f"Unknown fn_type '{fn_type}'. Choose 'trig' or 'legendre'.")

        return v, v_x, v_y


# 4. Gradient Helper

def get_gradients(u: torch.Tensor, coords: torch.Tensor):
    """du/dx and du/dy via autograd (create_graph=True for higher-order)."""
    grad = torch.autograd.grad(
        u, coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    return grad[:, 0:1], grad[:, 1:2]


# 5. Variational (Physics) Loss

def compute_variational_loss(model:            nn.Module,
                              interior_coords:  torch.Tensor,
                              interface_coords: torch.Tensor,
                              f_val:   float = -2.0,
                              gI_val:  float =  2.0,
                              test_modes: list = None,
                              fn_types:   list = None) -> torch.Tensor:

    if test_modes is None:
        test_modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]   # 16 modes

    if fn_types is None:
        fn_types = ["trig", "legendre"]

    u_int       = model(interior_coords)
    u_x, u_y   = get_gradients(u_int, interior_coords)

    residuals = []

    for fn_type in fn_types:
        for m, n in test_modes:
            v, vx, vy   = TestFunctions.get_test_fn(interior_coords, m, n, fn_type)
            domain_int  = torch.mean(u_x * vx + u_y * vy - f_val * v)
            v_face, _, _ = TestFunctions.get_test_fn(interface_coords, m, n, fn_type)
            iface_int    = torch.mean(gI_val * v_face)
            residuals.append((domain_int - iface_int) ** 2)

    return sum(residuals)


# 6. Visualization
def plot_results(model: nn.Module,
                 loss_history: dict,
                 epoch: int,
                 save_path: str = "vpinn_results.png"):
    model.eval()

    # 2-D evaluation grid
    Ng    = 200
    x_lin = np.linspace(0, 1, Ng)
    y_lin = np.linspace(0, 1, Ng)
    X, Y  = np.meshgrid(x_lin, y_lin)
    xy_flat = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        u_pred = model(xy_flat).numpy().reshape(X.shape)

    u_exact = exact_solution_np(X)          # shape (Ng, Ng)
    u_diff  = np.abs(u_pred - u_exact)
    final_mse = np.mean(u_diff ** 2)
    relative_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    
    print("\n" + "="*40)
    print(" FINAL EVALUATION METRICS")
    print("="*40)
    print(f" Global MSE        : {final_mse:.4e}")
    print(f" Relative L2 Error : {relative_l2:.4f}%")
    print("="*40 + "\n")
  
    #1-D cross-section at y = 0.5
    x_1d  = np.linspace(0, 1, 500)
    y_1d  = np.full_like(x_1d, 0.5)
    xy_1d = torch.tensor(np.stack([x_1d, y_1d], axis=1), dtype=torch.float32)
    with torch.no_grad():
        u_pred_1d = model(xy_1d).numpy().flatten()
    u_exact_1d = exact_solution_np(x_1d)

    # Loss history
    ep_arr = np.arange(len(loss_history["total"]))

    # Layout: 3 rows
    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_pred  = fig.add_subplot(gs[0, 0])
    ax_exact = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[0, 2])
    ax_loss  = fig.add_subplot(gs[1, :])
    ax_kink  = fig.add_subplot(gs[2, :])

    #Row 1: 2-D maps 
    vmin, vmax = 0.0, 0.25

    im0 = ax_pred.imshow(u_pred,  origin='lower', extent=[0,1,0,1],
                         cmap='jet', vmin=vmin, vmax=vmax)
    ax_pred.set_title("VPINN Prediction", fontsize=12)
    ax_pred.set_xlabel('x'); ax_pred.set_ylabel('y')
    fig.colorbar(im0, ax=ax_pred, fraction=0.046, pad=0.04)

    im1 = ax_exact.imshow(u_exact, origin='lower', extent=[0,1,0,1],
                          cmap='jet', vmin=vmin, vmax=vmax)
    ax_exact.set_title("Analytical Solution", fontsize=12)
    ax_exact.set_xlabel('x'); ax_exact.set_ylabel('y')
    fig.colorbar(im1, ax=ax_exact, fraction=0.046, pad=0.04)

    im2 = ax_err.imshow(u_diff, origin='lower', extent=[0,1,0,1], cmap='hot_r')
    ax_err.set_title("Absolute Error  |VPINN − Exact|", fontsize=12)
    ax_err.set_xlabel('x'); ax_err.set_ylabel('y')
    fig.colorbar(im2, ax=ax_err, fraction=0.046, pad=0.04)

    # Row 2: Loss convergence
    ax_loss.semilogy(ep_arr, loss_history["total"],   label="Total Loss",
                     linewidth=1.8,  color='navy')
    ax_loss.semilogy(ep_arr, loss_history["bndry"],   label="Boundary Loss  (×λ)",
                     linewidth=1.4,  color='crimson',  linestyle='--')
    ax_loss.semilogy(ep_arr, loss_history["physics"],  label="Variational Physics Loss",
                     linewidth=1.4,  color='darkorange', linestyle=':')
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss  (log scale)", fontsize=11)
    ax_loss.set_title("Loss Convergence", fontsize=12)
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, which="both", alpha=0.35)

    # Row 3: 1-D kink plot
    ax_kink.plot(x_1d, u_exact_1d, 'k-',  linewidth=2.5, label="Exact solution")
    ax_kink.plot(x_1d, u_pred_1d,  'r--', linewidth=2.0, label="VPINN prediction")
    ax_kink.axvline(0.5, color='steelblue', linestyle=':', linewidth=1.5,
                    label="Interface  x = 0.5")
    ax_kink.annotate("gradient\nkink", xy=(0.5, exact_solution_np(np.array([0.5]))[0]),
                     xytext=(0.55, 0.18),
                     arrowprops=dict(arrowstyle='->', color='steelblue'),
                     fontsize=10, color='steelblue')
    ax_kink.set_xlabel("x", fontsize=11)
    ax_kink.set_ylabel("u(x,  y=0.5)", fontsize=11)
    ax_kink.set_title("1-D Cross-Section at y = 0.5  (Kink / Gradient-Jump Plot)", fontsize=12)
    ax_kink.legend(fontsize=10)
    ax_kink.grid(True, alpha=0.35)

    fig.suptitle(f"VPINN — Interface Problem via Variational Method   (Epoch {epoch})",
                 fontsize=14, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved → {save_path}")



# 7. Training
def train(epochs:        int   = 5000,
          lr:            float = 1e-3,
          lambda_bndry:  float = 100.0,
          fn_types:      list  = None,
          seed:          int   = 42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if fn_types is None:
        fn_types = ["trig", "legendre"]

    #Data 
    print("Generating collocation data...")
    data = generate_data(N_interior=2000, N_interface=300, N_boundary=500, seed=seed)

    
    interior_coords  = torch.cat([data["interior_1"], data["interior_2"]], dim=0)
    interior_coords.requires_grad_(True)         

    interface_coords = data["interface_pts"]
    bndry_coords     = data["boundary_pts"]
    bndry_u_true     = data["boundary_u"]        

    # Test modes: 4×4 grid
    test_modes = [(m, n) for m in range(1, 5) for n in range(1, 5)]   

    # Model
    model     = VPINN(layers=[2, 50, 50, 50, 50, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    loss_history = {"total": [], "bndry": [], "physics": []}

    print(f"Training: {epochs} epochs | λ_bndry={lambda_bndry} | "
          f"fn_types={fn_types} | test_modes=4×4={len(test_modes)}")
    print("-" * 72)

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Boundary loss (strong-form Dirichlet)
        u_bndry_pred = model(bndry_coords)
        loss_bndry   = nn.MSELoss()(u_bndry_pred, bndry_u_true)

        # Variational physics loss (weak form) 
        loss_physics = compute_variational_loss(
            model, interior_coords, interface_coords,
            f_val=-2.0, gI_val=2.0,
            test_modes=test_modes, fn_types=fn_types
        )

        loss = lambda_bndry * loss_bndry + loss_physics
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history["total"].append(loss.item())
        loss_history["bndry"].append(loss_bndry.item())
        loss_history["physics"].append(loss_physics.item())

        if epoch % 500 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:05d} | LR {lr_now:.2e} | "
                  f"Total {loss.item():.5f} | "
                  f"Bndry {loss_bndry.item():.5f} | "
                  f"Phys {loss_physics.item():.5f}")

    print("-" * 72)
    print("Training complete!")
    return model, loss_history



# 8. Entry Point
if __name__ == "__main__":
    model, loss_history = train(
        epochs       = 5000,
        lr           = 1e-3,
        lambda_bndry = 100.0,
        fn_types     = ["trig", "legendre"],
        seed         = 42,
    )
    plot_results(model, loss_history, epoch=5000)