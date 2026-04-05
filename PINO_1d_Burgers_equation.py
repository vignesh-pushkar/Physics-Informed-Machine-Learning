import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import gc
torch.manual_seed(42)
 
###hyperparameters
Nx = 8192                ###spatial resolution
Cin, Cout = 2, 1         ###input channels (a, a_x), output channel (u)
N1, N2 = 64, 64         ###channel widths through fno layers
b = 64                  ###batch size
num_layers = 4           ###number of fno layers stacked
k_max = 12             ###max fourier modes retained per layer
nu = 0.01 / np.pi        ###viscosity coefficient for burgers equation
lam = 0.1                ###physics loss weight relative to data loss
dt = 1.0                 ###normalized time horizon, t in [0, 1]
L  = 2.0                 ###spatial domain length, x in [-1, 1]
 
###spectral derivative of order `order` using fft
###u: [batch, C, Nx] -> same shape. exact for periodic signals, natural for fno
def spectral_deriv(u, order=1):
    N  = u.shape[-1]
    k  = torch.fft.fftfreq(N, d=1.0/N).to(u.device)   ###integer wavenumbers [0,1,...,N/2,-N/2+1,...,-1]
    ik = (2j * np.pi / L * k) ** order                  ###(ik)^order spectral multiplier
    return torch.fft.ifft(torch.fft.fft(u, dim=-1) * ik, dim=-1).real
 
###burgers residual: du/dt + u*(du/dx) - nu*(d2u/dx2) = 0
###u_pred : fno output, solution at t=T  [batch, 1, Nx]
###batch_x: model input, first channel is a(x) at t=0  [batch, Cin, Nx]
def physics_residual(u_pred, batch_x):
    u_t  = (u_pred - batch_x[:, :1, :]) / dt   ###large-step time deriv: (u(T) - u(0)) / T
    u_x  = spectral_deriv(u_pred, order=1)
    u_xx = spectral_deriv(u_pred, order=2)
    return u_t + u_pred * u_x - nu * u_xx
 
###single fourier layer: spectral conv (low freq path) + two skip connections (high freq path)
class FNO_Layer(nn.Module):
    def __init__(self, N1, N2, Nx, k_max, mesh_dim, batch_size):
        super(FNO_Layer, self).__init__()
        self.Sigmoid        = nn.GELU()
        self.spec_wt_tensor = nn.Parameter(torch.randn(N1, N2, k_max, dtype=torch.cfloat)/(N1*N2))
        self.channel_mix    = nn.Conv1d(in_channels=N2, out_channels=N2, kernel_size=1)
        self.skip1          = nn.Conv1d(in_channels=N1, out_channels=N2, kernel_size=1)
        self.skip2          = nn.Conv1d(in_channels=N1, out_channels=N2, kernel_size=1)
        self.k_max          = k_max
        self.batch_size     = batch_size
        self.Nx             = Nx
        self.N2             = N2
 
    def forward(self, x):
        y = x.clone()
 
        x = torch.fft.fftn(x,  dim=-1)
        x = torch.fft.fftshift(x, dim=-1)
 
        center = self.Nx // 2
        start  = center - self.k_max // 2
        end    = center + self.k_max // 2
 
        x = x[:, :, start:end]
        x = torch.einsum('bcx,cdx->bdx', x, self.spec_wt_tensor)   ###[b,C_in,k],[C_in,C_out,k]->[b,C_out,k]
        out = torch.zeros(self.batch_size, self.N2, self.Nx, dtype=torch.cfloat, device=x.device)
        out[:, :, start:end] = x
 
        x = torch.fft.ifftshift(out, dim=-1)
        x = torch.fft.ifftn(x,  dim=-1)
        x = x.real + self.skip1(y)    ###first residual add then activate
        x = self.Sigmoid(x)
 
        x = self.channel_mix(x) + self.skip2(y)   ###second residual add then activate
        x = self.Sigmoid(x)
        return x
 
###physics-informed neural operator: fno backbone supervised by both data and burgers residual
class PINO(nn.Module):
    def __init__(self):
        super(PINO, self).__init__()
        self.Sigmoid = nn.GELU()
        self.lift    = nn.Conv1d(in_channels=Cin,  out_channels=N1,   kernel_size=1)
        self.project = nn.Conv1d(in_channels=N2,   out_channels=Cout, kernel_size=1)
        self.layers  = nn.ModuleList([FNO_Layer(k_max=k_max, mesh_dim=1, N1=N1, N2=N2, Nx=Nx, batch_size=b) for _ in range(num_layers)])
 
    def forward(self, x):
        x = self.lift(x)
        
        for layer in self.layers:
            x = layer(x)
 
        x = self.project(x)
        return x
 
from scipy.io import loadmat
 
data = loadmat('/kaggle/input/datasets/scaomath/pde-dataset/burgers_data_R10.mat')
print("Variables in file:")
for var in data:
    print(var)
for var in ['a', 'u', 'a_x']:
    print(f"{var} shape: {data[var].shape}")
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
model = PINO().to(device)
 
epochs    = 100
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion = nn.MSELoss()
losses    = []
vals=[]
print("started training")
 
from torch.utils.data import DataLoader, TensorDataset, Subset
 
inputs  = torch.from_numpy(np.stack([data['a'], np.column_stack((data['a_x'], np.array([0]*2048)))], axis=1)).float()
targets = torch.from_numpy(data['u']).float().unsqueeze(1)
dataset = TensorDataset(inputs, targets)
train_subset = Subset(dataset, range(512))
val_subset = Subset(dataset, range(512, 1024))
test_subset =Subset(dataset, range(1024, 2048))

train_loader  = DataLoader(train_subset, batch_size=b, shuffle=True)
val_loader  = DataLoader(val_subset, batch_size=b, shuffle=False)
test_loader  = DataLoader(test_subset, batch_size=b, shuffle=False)
 
try:
    for epoch in range(epochs):
        model.train()
        mean_train_loss=0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
 
            optimizer.zero_grad()
 
            ###forward: fno maps a(x) -> u_pred(x) across full spatial grid
            prediction = model(batch_x)
 
            ###data loss: supervised operator learning against ground truth u
            loss_data = criterion(prediction, batch_y)
 
            ###physics loss: enforce burgers eq on predicted solution spectrally
            residual  = physics_residual(prediction, batch_x)
            loss_phys = torch.mean(residual ** 2)
            mean_train_loss += loss_data.item()
            loss = loss_data + lam * loss_phys
            loss.backward()
            optimizer.step()
 
        scheduler.step()
        mean_train_loss/=len(train_loader)
        losses.append(mean_train_loss)

        model.eval()
        with torch.no_grad():
            mean_val_loss=0.0
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
          
                ###forward: fno maps a(x) -> u_pred(x) across full spatial grid
                prediction = model(batch_x)
     
                ###data loss: supervised operator learning against ground truth u
                loss_data = criterion(prediction, batch_y)
     
                loss = loss_data
                mean_val_loss+=loss.item()
            mean_val_loss/=len(val_loader)
            vals.append(mean_val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:5d} | Loss: {mean_train_loss:.6f} | Data: {loss_data.item():.6f} | Physics: {loss_phys.item():.6f} | Validation: {mean_val_loss:.6f}")

    
    print("training done")
    
    model.eval()
    del optimizer
    del train_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    print("testing...")
    loss=0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            ###forward: fno maps a(x) -> u_pred(x) across full spatial grid
            prediction = model(batch_x)
        
            ###data loss: supervised operator learning against ground truth u
            loss_data = criterion(prediction, batch_y)
        
            loss += loss_data.item()
    loss/=(1024/b)
    
    print(f"final test loss:{loss}")
except KeyboardInterrupt:
    print("Manual interruption detected. Cleaning up...")
except RuntimeError as e:
    print(f"{e}")

 
import matplotlib.pyplot as plt
plt.plot(np.linspace(1, epochs, num=len(losses)), losses)
plt.plot(np.linspace(1, epochs, num=len(vals)), vals)
plt.figure()
x, y = next(iter(test_loader))
plt.plot(np.linspace(0, 2*np.pi, Nx), y[0].cpu().numpy().squeeze())
with torch.no_grad():
    plt.plot(np.linspace(0, 2*np.pi, Nx), model(x.to(device))[0].cpu().numpy().squeeze())

torch.cuda.empty_cache()
torch.cuda.synchronize()
print("GPU Memory Cleared.")
for name in list(globals().keys()):
    if not name.startswith('_'):
        del globals()[name]
print("all variables deleted.")
