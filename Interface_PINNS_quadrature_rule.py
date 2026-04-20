###########vipnns-2

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy.polynomial.legendre as leg


# 1. Exact Solution
def exact_solution_np(x):
    return np.where(x <= 0.5, x**2, (x - 1.0)**2)


# 2. Quadrature Data Generator (The Core Upgrade)
def generate_quadrature_data(nx=80, ny=80, n_interface=100, N_boundary=500, seed=42):
    """Generates Gauss-Legendre Quadrature points and their integration weights."""
    
    def get_quad_2d(n_x, n_y, x_bounds, y_bounds):
        x, wx = leg.leggauss(n_x)
        y, wy = leg.leggauss(n_y)
        
        x_map = 0.5 * (x_bounds[1] - x_bounds[0]) * x + 0.5 * (x_bounds[1] + x_bounds[0])
        y_map = 0.5 * (y_bounds[1] - y_bounds[0]) * y + 0.5 * (y_bounds[1] + y_bounds[0])
        
        wx_map = 0.5 * (x_bounds[1] - x_bounds[0]) * wx
        wy_map = 0.5 * (y_bounds[1] - y_bounds[0]) * wy
        
        X, Y = np.meshgrid(x_map, y_map, indexing='ij')
        WX, WY = np.meshgrid(wx_map, wy_map, indexing='ij')
        
        pts = np.stack([X.flatten(), Y.flatten()], axis=1)
        weights = (WX * WY).flatten()
        return pts, weights

    def get_quad_1d(n_y, x_fixed, y_bounds):
        y, wy = leg.leggauss(n_y)
        y_map = 0.5 * (y_bounds[1] - y_bounds[0]) * y + 0.5 * (y_bounds[1] + y_bounds[0])
        wy_map = 0.5 * (y_bounds[1] - y_bounds[0]) * wy
        
        pts = np.stack([np.full_like(y_map, x_fixed), y_map], axis=1)
        return pts, wy_map

    # Quadrature: Omega_1 and Omega_2 
    pts1, w1 = get_quad_2d(nx, ny, [0.0, 0.5], [0.0, 1.0])
    pts2, w2 = get_quad_2d(nx, ny, [0.5, 1.0], [0.0, 1.0])
    
    #Quadrature: Interface (x = 0.5) 
    pts_int, w_int = get_quad_1d(n_interface, 0.5, [0.0, 1.0])

    # Random: Dirichlet Boundary
    rng = np.random.default_rng(seed)
    n_each = N_boundary // 4
    b_x0  = np.stack([np.zeros(n_each), rng.uniform(0, 1, n_each)], axis=1)
    b_x1  = np.stack([np.ones(n_each), rng.uniform(0, 1, n_each)], axis=1)
    b_y0  = np.stack([rng.uniform(0, 1, n_each), np.zeros(n_each)], axis=1)
    b_y1  = np.stack([rng.uniform(0, 1, n_each), np.ones(n_each)], axis=1)
    bndry_xy = np.concatenate([b_x0, b_x1, b_y0, b_y1], axis=0)
    bndry_u  = exact_solution_np(bndry_xy[:, 0])

    return {
        "interior_pts": torch.tensor(np.vstack([pts1, pts2]), dtype=torch.float32),
        "interior_w": torch.tensor(np.concatenate([w1, w2]), dtype=torch.float32).unsqueeze(1),
        "interface_pts": torch.tensor(pts_int, dtype=torch.float32),
        "interface_w": torch.tensor(w_int, dtype=torch.float32).unsqueeze(1),
        "boundary_pts": torch.tensor(bndry_xy, dtype=torch.float32),
        "boundary_u": torch.tensor(bndry_u, dtype=torch.float32).unsqueeze(1),
    }


# 3. Network Architecture
class VPINN(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 50, 1]):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        for lin in self.linears[:-1]:
            x = self.activation(lin(x))
        return self.linears[-1](x)

# 4. Test Functions & Gradients
class TestFunctions:
    @staticmethod
    def get_trig_test_fn(coords, m, n):
        """v = sin(m*pi*x) * sin(n*pi*y). Vanishes perfectly on [0,1] edges."""
        x, y = coords[:, 0:1], coords[:, 1:2]
        
        v   = torch.sin(m * torch.pi * x) * torch.sin(n * torch.pi * y)
        v_x = m * torch.pi * torch.cos(m * torch.pi * x) * torch.sin(n * torch.pi * y)
        v_y = n * torch.pi * torch.sin(m * torch.pi * x) * torch.cos(n * torch.pi * y)
        return v, v_x, v_y

def get_gradients(u, coords):
    grad = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return grad[:, 0:1], grad[:, 1:2]


# 5. Quadrature Variational Loss
def compute_variational_loss_quad(model, int_coords, int_w, iface_coords, iface_w,
                                  f_val=-2.0, gI_val=2.0, test_modes=None):
    if test_modes is None:
        test_modes = [(m, n) for m in range(1, 5) for n in range(1, 5)] 

    u_int = model(int_coords)
    u_x, u_y = get_gradients(u_int, int_coords)

    residuals = []
    
    for m, n in test_modes:
        v, vx, vy = TestFunctions.get_trig_test_fn(int_coords, m, n)
        integrand_domain = (u_x * vx) + (u_y * vy) - (f_val * v)
        domain_int = torch.sum(integrand_domain * int_w)

        v_face, _, _ = TestFunctions.get_trig_test_fn(iface_coords, m, n)
        integrand_iface = gI_val * v_face
        iface_int = torch.sum(integrand_iface * iface_w)

        residuals.append((domain_int - iface_int) ** 2)

    return torch.mean(torch.stack(residuals))

# 6. Evaluation and Plotting
def plot_results(model, loss_history, epoch):
    model.eval()

    # 1.grid
    Ng = 200
    X, Y = np.meshgrid(np.linspace(0, 1, Ng), np.linspace(0, 1, Ng))
    xy_flat = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        u_pred = model(xy_flat).numpy().reshape(X.shape)
    
    u_exact = exact_solution_np(X)
    u_diff  = np.abs(u_pred - u_exact)

    # 2. Print Metrics
    final_mse = np.mean(u_diff ** 2)
    relative_l2 = (np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)) * 100
    
    print("\n" + "="*40)
    print(" FINAL EVALUATION METRICS")
    print("="*40)
    print(f" Global MSE        : {final_mse:.4e}")
    print(f" Relative L2 Error : {relative_l2:.4f}%")
    print("="*40 + "\n")

    # 3. 1D Cross-section
    x_1d = np.linspace(0, 1, 500)
    xy_1d = torch.tensor(np.stack([x_1d, np.full_like(x_1d, 0.5)], axis=1), dtype=torch.float32)
    with torch.no_grad():
        u_pred_1d = model(xy_1d).numpy().flatten()
    u_exact_1d = exact_solution_np(x_1d)

    # 4. Plotting Dashboard
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_pred  = fig.add_subplot(gs[0, 0])
    ax_exact = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[0, 2])
    ax_loss  = fig.add_subplot(gs[1, :])
    ax_kink  = fig.add_subplot(gs[2, :])

    vmin, vmax = 0.0, 0.25

    im0 = ax_pred.imshow(u_pred, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=vmin, vmax=vmax)
    ax_pred.set_title("VPINN Prediction (Quadrature)")
    fig.colorbar(im0, ax=ax_pred)

    im1 = ax_exact.imshow(u_exact, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=vmin, vmax=vmax)
    ax_exact.set_title("Analytical Solution")
    fig.colorbar(im1, ax=ax_exact)

    im2 = ax_err.imshow(u_diff, origin='lower', extent=[0,1,0,1], cmap='hot_r')
    ax_err.set_title("Absolute Error |VPINN − Exact|")
    fig.colorbar(im2, ax=ax_err)

    ep_arr = np.arange(len(loss_history["total"]))
    ax_loss.semilogy(ep_arr, loss_history["total"], label="Total Loss", color='navy')
    ax_loss.semilogy(ep_arr, loss_history["bndry"], label="Boundary Loss (×λ)", color='crimson', linestyle='--')
    ax_loss.semilogy(ep_arr, loss_history["physics"], label="Variational Loss", color='darkorange', linestyle=':')
    ax_loss.set_title("Loss Convergence")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.35)

    ax_kink.plot(x_1d, u_exact_1d, 'k-', linewidth=2.5, label="Exact solution")
    ax_kink.plot(x_1d, u_pred_1d, 'r--', linewidth=2.0, label="VPINN prediction")
    ax_kink.axvline(0.5, color='steelblue', linestyle=':', label="Interface x = 0.5")
    ax_kink.set_title("1-D Cross-Section at y = 0.5")
    ax_kink.legend()
    ax_kink.grid(True, alpha=0.35)

    plt.suptitle(f"VPINN — Quadrature + Trig Functions (Epoch {epoch})", fontsize=16, fontweight='bold')
    plt.show()

# 7. Training Loop
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generating Quadrature Data
    print("Generating Gauss-Legendre Quadrature data...")
    data = generate_quadrature_data(nx=40, ny=40, n_interface=100)

    int_coords = data["interior_pts"].requires_grad_(True)
    int_w = data["interior_w"]
    
    iface_coords = data["interface_pts"]
    iface_w = data["interface_w"]
    
    bndry_coords = data["boundary_pts"]
    bndry_u_true = data["boundary_u"]

    # 2. Init Model
    model = VPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    lambda_bndry = 100.0
    epochs = 5000
    loss_history = {"total": [], "bndry": [], "physics": []}

    print("-" * 60)
    print("Starting Quadrature Training...")
    
    # 3. Train
    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        # Strong Boundary Loss
        u_bndry_pred = model(bndry_coords)
        loss_bndry = nn.MSELoss()(u_bndry_pred, bndry_u_true)

        # Weak Quadrature Physics Loss
        loss_physics = compute_variational_loss_quad(
            model, int_coords, int_w, iface_coords, iface_w
        )

        loss = (lambda_bndry * loss_bndry) + loss_physics
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history["total"].append(loss.item())
        loss_history["bndry"].append(loss_bndry.item())
        loss_history["physics"].append(loss_physics.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch:05d} | Total: {loss.item():.5f} | Phys: {loss_physics.item():.5f}")

    print("Training complete!")
    plot_results(model, loss_history, epochs)