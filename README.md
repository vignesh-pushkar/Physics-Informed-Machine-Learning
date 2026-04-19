# Physics-Informed Machine Learning: PINNs, FNO & PINO

> A comprehensive study of Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNO), and Physics-Informed Neural Operators (PINO) applied to benchmark PDE problems and 3-D conjugate heat transfer.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
  - [1. PINNs — 1D & 2D Poisson](#1-pinns--1d--2d-poisson)
  - [2. Fourier Neural Operator (FNO)](#2-fourier-neural-operator-fno)
  - [3. Physics-Informed Neural Operator (PINO)](#3-physics-informed-neural-operator-pino)
  - [4. Inverse Problem: Parameter Identification](#4-inverse-problem-parameter-identification)
  - [5. Interface Problem via VPINN](#5-interface-problem-via-vpinn)
  - [6. 1-D Burger's Flow](#6-1-d-burgers-flow)
  - [7. Navier–Stokes / Kolmogorov Flow](#7-navierstokes--kolmogorov-flow)
  - [8. Conjugate Heat Transfer (3-D)](#8-conjugate-heat-transfer-3-d)
- [Results Summary](#results-summary)
- [Setup & Installation](#setup--installation)
- [Running the Code](#running-the-code)
- [Adding Results / Plots](#adding-results--plots)
- [References](#references)

---

## Overview

This project explores how physical laws (expressed as PDEs) can be embedded into neural network training to produce physically consistent solutions with reduced data requirements.

**Three core paradigms are studied:**

| Paradigm | Key Idea |
|---|---|
| **PINN** | MLP trained with a PDE-residual loss via automatic differentiation |
| **FNO** | Operator learned in Fourier space; resolution-invariant, O(N log N) |
| **PINO** | FNO backbone + PDE constraints; merges operator learning with physics |

---

## Repository Structure

```
.
├── 3D_Fno.py                    # 3-D FNO for conjugate heat transfer (CHT)
├── 3D_Pino.py                   # 3-D PINO for CHT
├── Pinn_Data_Generation_CHT.py  # PINN-based dataset generation for CHT
├── validate_CHT.py              # CHT validation vs NVIDIA/OpenFOAM reference
├── FNO_1_1d_Burgers.py          # 1-D Burgers baseline (FNO)
├── FNO_2_1d_Burgers.py          # 1-D Burgers FNO experiments
├── PINO_1d_Burgers_equation.py  # 1-D Burgers PINO experiments
├── Interface_Problem_VPINNs     # Interface problem (VPINN)
├── Interface_Problem_PINO       # Interface problem (PINO)
└── README.md
```

> **Tip:** Keep the `results/` sub-folders organised by experiment so plots are easy to find and reference in the README (see [Adding Results / Plots](#adding-results--plots)).

---

## Experiments

### 1. PINNs — 1D & 2D Poisson

Physics-Informed Neural Networks approximate PDE solutions by minimising a combined loss:

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}$$

**1D Poisson — Results**

| Metric | Value |
|---|---|
| Final loss | $2 \times 10^{-6}$ |
| Agreement | Exact ≈ Predicted |

| Exact Solution | Predicted Solution |
|---|---|
| ![1D PINN Exact](results/pinn/pinn_1d_exact.png) | ![1D PINN Predicted](results/pinn/pinn_1d_prediction.png) |

**2D Poisson — Results**

| Metric | Value |
|---|---|
| $L_2$ error | $3.21 \times 10^{-2}$ |

![2D PINN](results/pinn/pinn_2d_exact.png)

**Learnable Loss Balancing**

A learnable parameter $k = \exp(\alpha)$ weights the boundary loss:
$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + k \cdot \mathcal{L}_{\text{BC}}$$

| Metric | Value |
|---|---|
| Final $k$ | 0.685833 |
| $L_2$ error | $4.75 \times 10^{-2}$ |

![PINN with learnable k](results/pinn/pinn_k_exact.png)

---

### 2. Fourier Neural Operator (FNO)

Each FNO layer applies:

$$v(x) = \sigma\!\left(\mathcal{F}^{-1}\!\left(R(k)\,\mathcal{F}(u)(k)\right) + W\,u(x)\right)$$

Only $|k| \le K$ low-frequency modes are retained, giving $\mathcal{O}(N \log N)$ complexity per layer. The model is **resolution-invariant**: trained at one grid resolution, it generalises to finer/coarser meshes.

---

### 3. Physics-Informed Neural Operator (PINO)

PINO augments the FNO backbone with a physics loss:

$$\mathcal{L} = \lambda_d\,\mathcal{L}_{\text{data}} + \lambda_p\,\mathcal{L}_{\text{physics}}, \qquad \mathcal{L}_{\text{physics}} = \|\mathcal{N}(v) - f\|^2$$

This reduces label requirements, improves generalisation in sparse-data regimes, and supports both forward and inverse problems.

---

### 4. Inverse Problem: Parameter Identification

**2D Poisson inverse problem** — identify unknown coefficients $k_1, k_2$ in:

$$k_1\,u_{xx} + k_2\,u_{yy} = f(x,y)$$

| Method | $k_1$ (true: 1.0) | $k_2$ (true: 1.0) |
|---|---|---|
| **PINN** | 0.9693 | 1.0747 |
| **PINO** | 0.6465 | 0.6176 |

PINN recovers parameters accurately via automatic differentiation. PINO suffers from finite-difference approximation errors.

| PINN Loss | $k_1$ Error | $k_2$ Error |
|---|---|---|
| ![PINN inv loss](results/pinn/pinn_inv_loss.png) | ![k1 error](results/pinn/pinn_k1_error.png) | ![k2 error](results/pinn/pinn_k2_error.png) |

| PINO Loss | PINO $k_1$ | PINO $k_2$ |
|---|---|---|
| ![PINO loss](results/pino/pino_loss.png) | ![PINO w](results/pino/w.png) | ![PINO download](results/pino/download.png) |

---

### 5. Interface Problem via VPINN

**Problem:** 2D Poisson on $(0,1)^2$ with interface at $x = 0.5$, where $\partial_x u$ is discontinuous.

**VPINN test spaces used:**
- Trigonometric: $v_{mn} = \sin(m\pi x)\sin(n\pi y)$
- Bubble-Legendre: $v_{mn} = x(1-x)y(1-y)\,P_m(2x-1)\,P_n(2y-1)$

| Variant | MSE | Relative $L^2$ |
|---|---|---|
| Trig + Legendre | $1.76 \times 10^{-5}$ | 3.76% |
| Quadrature | $1.56 \times 10^{-5}$ | 3.54% |
| Quadrature + RBF | $1.50 \times 10^{-5}$ | **3.47%** |

**PINO on the same problem** (64×64 grid, 4 Fourier layers, width 48):

| Metric | Family test set | Tutorial case |
|---|---|---|
| Relative $L^2$ | 3.41% | **2.43%** |
| MSE | $1.24 \times 10^{-5}$ | $7.23 \times 10^{-6}$ |
| Weak residual | $6.02 \times 10^{-6}$ | $3.76 \times 10^{-7}$ |
| Jump loss | $9.93 \times 10^{-3}$ | $9.93 \times 10^{-3}$ |

![VPINN Results 1](results/vpinn/vpinn_results_1.jpeg)
![VPINN Results 2](results/vpinn/vpinn_results_2.jpeg)
![VPINN Results 3](results/vpinn/vpinn_results_3.jpeg)

---

### 6. 1-D Burger's Flow

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}, \quad x \in [0,2\pi],\; t \in [0,1]$$

Operator learning from $u(x,0) \mapsto u(x,1)$ on a grid of resolution **8192**.

| | FNO | PINO |
|---|---|---|
| Fourier modes | 2048 | 12 |
| Regularisation | Data only | Data + PDE |
| Training samples | 512 | 512 |
| Test samples | 1024 | 1024 |

| FNO: Loss & Prediction | | PINO: Loss & Prediction | |
|---|---|---|---|
| ![Burgers FNO 1](results/burgers/1d_Burgers_FNO1.png) | ![Burgers FNO 2](results/burgers/1d_Burgers_fno2.png) | ![Burgers PINO 1](results/burgers/burger_pino1.png) | ![Burgers PINO 2](results/burgers/burger_pino2.png) |

---

### 7. Navier–Stokes / Kolmogorov Flow

**Vorticity formulation:**

$$\partial_t \omega + \mathbf{u}\cdot\nabla\omega = \nu\,\Delta\omega + f(x)$$

**Subtasks:**

| Subtask | Details |
|---|---|
| Chaotic Kolmogorov flow | $Re=500$, $t \in [t_0, t_0+1]$, $l=2\pi$, modes $k_{\max}=12$ |
| Transfer learning ($Re$) | Pre-trained at $Re=100$, fine-tuned to $Re=40$ and $Re=500$ |
| Long temporal flow | $T=50$, steps $\{0,5,\dots,50\}$, $64\times64$ grid |

**Transfer learning loss:**

$$\mathcal{L}_{\text{total}} = \beta\,\mathcal{L}_{\text{phys}}(\nu_{\text{target}}) + \alpha\,\mathcal{L}_{\text{anchor}}$$

---

### 8. Conjugate Heat Transfer (3-D)

3-D steady-state conjugate heat transfer over a **parametric finned heat sink** with 6 design parameters.

**Pipeline:**

```
PINN dataset generation (125 samples)
        ↓
3-D FNO / PINO training (50×20×20 grid)
        ↓
Validation vs. NVIDIA OpenFOAM reference
```

**FNO test errors (masked relative $\ell_2$, %):**

| Field | Mean | Std |
|---|---|---|
| $u$ | 34.19 | 8.19 |
| $v$ | 45.61 | 13.61 |
| $w$ | 72.27 | 16.63 |
| $p$ | 41.94 | 17.29 |
| $\theta_f$ | 44.09 | 20.91 |
| $\theta_s$ | 57.20 | 27.91 |

**PINO test errors:**

| Field | Mean | Std |
|---|---|---|
| $u$ | 36.69 | 9.36 |
| $v$ | 47.34 | 14.78 |
| $w$ | 73.67 | 16.96 |
| $p$ | 42.74 | 18.40 |
| $\theta_f$ | 55.93 | 28.96 |
| $\theta_s$ | **41.49** | 17.33 |

**Validation vs. NVIDIA OpenFOAM (Rel-$\ell_2$ % / MAE):**

| Field | PINN | FNO | PINO |
|---|---|---|---|
| $u$ | 44.90 / 0.397 | 46.20 / 0.397 | **42.17** / 0.374 |
| $\theta_f$ | 36.17 / 0.027 | **35.25** / 0.026 | 34.69 / 0.027 |
| $\theta_s$ | 51.09 / 0.061 | **33.59** / 0.039 | 42.44 / 0.053 |

> **Note:** Large velocity errors are primarily explained by a 2× viscosity mismatch ($\nu=0.01$ vs. $\nu=0.02$) between training and reference, not model failure.

#### CHT Plots

> Run the CHT commands in [Running the Code](#running-the-code) with `--out results/cht` to generate these files.

| FNO Curves | FNO Prediction (YZ) |
|---|---|
| ![CHT FNO curves](results/cht/fno_curves.png) | ![CHT FNO prediction](results/cht/fno_prediction_yz.png) |

| PINO Curves | PINO Prediction (YZ) |
|---|---|
| ![CHT PINO curves](results/cht/pino_curves.png) | ![CHT PINO prediction](results/cht/pino_prediction_yz.png) |

![CHT PINO physics residuals](results/cht/pino_physics_residuals.png)
![CHT validation errors](results/cht/validate_errors.png)
![CHT validation fields](results/cht/validate_fields_yz.png)
![CHT validation scatter](results/cht/validate_scatter.png)

---

## Results Summary

| Experiment | Best Method | Key Metric |
|---|---|---|
| 1-D Poisson PINN | PINN | Loss $2\times10^{-6}$ |
| 2-D Poisson PINN (learnable k) | PINN + learnable k | $L_2 = 4.75\times10^{-2}$ |
| Inverse problem | PINN | $k_1=0.97,\; k_2=1.07$ |
| Interface problem | PINO | $L_2 = 2.43\%$ |
| CHT solid temperature | FNO | Rel-$\ell_2 = 33.6\%$ (vs NVIDIA) |
| CHT fluid temperature | PINO (internal) | Rel-$\ell_2 = 55.93\%$ |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a conda environment
conda create -n piml python=3.10 -y
conda activate piml

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib scikit-learn tqdm
```

> **GPU:** All experiments were developed on a single NVIDIA T4 (Kaggle). A CUDA-enabled GPU is strongly recommended.

---

## Running the Code

| Script | Description | Command |
|---|---|---|
| `Pinn_Data_Generation_CHT.py` | Generate CHT training data | `python Pinn_Data_Generation_CHT.py` |
| `3D_Fno.py` | Train 3-D FNO on CHT | `python 3D_Fno.py --out results/cht` |
| `3D_Pino.py` | Train 3-D PINO on CHT | `python 3D_Pino.py --out results/cht` |
| `validate_CHT.py` | Validate against reference and save CHT plots | `python validate_CHT.py --out results/cht` |
| `FNO_1_1d_Burgers.py` | 1-D Burgers FNO baseline | `python FNO_1_1d_Burgers.py` |
| `FNO_2_1d_Burgers.py` | 1-D Burgers FNO experiments | `python FNO_2_1d_Burgers.py` |
| `PINO_1d_Burgers_equation.py` | 1-D Burgers PINO | `python PINO_1d_Burgers_equation.py` |
| `Interface_Problem_VPINNs` | Interface problem (VPINN) | `python Interface_Problem_VPINNs` |
| `Interface_Problem_PINO` | Interface problem (PINO) | `python Interface_Problem_PINO` |

---

## Adding Results / Plots

### Recommended folder structure inside `results/`

```
results/
├── pinn/
│   ├── pinn_1d_exact.png
│   ├── pinn_1d_prediction.png
│   ├── pinn_2d_exact.png
│   ├── pinn_2d_error.png
│   ├── pinn_k_exact.png
│   ├── pinn_inv_loss.png
│   ├── pinn_k1_error.png
│   └── pinn_k2_error.png
├── fno/
│   ├── 1d_Burgers_FNO1.png
│   └── 1d_Burgers_fno2.png
├── pino/
│   ├── pino_loss.png
│   ├── w.png
│   ├── download.png
│   ├── burger_pino1.png
│   └── burger_pino2.png
├── vpinn/
│   ├── vpinn_results_1.jpeg
│   ├── vpinn_results_2.jpeg
│   └── vpinn_results_3.jpeg
├── burgers/           # symlinked / copied from fno/ and pino/
├── kolmogorov/
└── cht/
    ├── fno_curves.png
    ├── fno_prediction_yz.png
    ├── pino_curves.png
    ├── pino_prediction_yz.png
    ├── pino_physics_residuals.png
    ├── validate_errors.png
    ├── validate_fields_yz.png
    └── validate_scatter.png
```

### How to save plots from your scripts

At the end of any `matplotlib` plotting block, save to the appropriate sub-folder:

```python
import os, matplotlib.pyplot as plt

os.makedirs("results/pinn", exist_ok=True)
plt.savefig("results/pinn/pinn_1d_exact.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Should you include plots in the README?

**Yes** — GitHub renders images inline if you use a relative path like:

```markdown
![Alt text](results/pinn/pinn_1d_exact.png)
```

**Best practices:**
- Keep image files **under ~500 KB** each (use PNG for line plots, JPEG for heatmaps).
- Use descriptive alt-text so the README is accessible.
- For large figures (e.g., multi-panel), consider linking to the file instead of embedding:
  ```markdown
  [View full CHT results](results/cht/cht_summary.png)
  ```
- Add a `.gitignore` rule for large checkpoint files so only plots are tracked:
  ```gitignore
  # Ignore model checkpoints but keep result images
  results/**/*.pth
  results/**/*.pt
  data/
  ```

---

## References

1. Li, Z. et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
2. Raissi, M., Perdikaris, P. & Karniadakis, G. E. *Physics-informed neural networks.* JCP 2019.
3. Li, Z. et al. *Physics-Informed Neural Operator for Learning Partial Differential Equations.* ACM / JML 2021.
4. NVIDIA PhysicsNeMo Conjugate Heat Transfer Reference Dataset.
