### This impelementation follows the second architecture given in fno a practical perspective'###############
#############Needs normalization 80% done.########################
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
torch.manual_seed(42)
Nx, Ny, C = 8192, 16, 3
Cin, Cout = 2, 1
N1, N2=64, 64
b=128
num_layers=4
###implementation of a single fourier layer. Hopefully the final version.
###considering a change in the tensor format, from batch, x, y, channel to batch, channel, x, y
class FNO_Layer(nn.Module):
    def __init__(self, N1, N2, Nx, k_max, mesh_dim, batch_size):
        super(FNO_Layer, self).__init__()
        self.Sigmoid=nn.GELU()
        self.spec_wt_tensor=nn.Parameter(torch.randn(N1, N2, k_max, dtype=torch.cfloat)/(N1*N2))
        self.channel_mix=nn.Conv1d(in_channels=N2, out_channels=N2, kernel_size=1)
        self.k_max=k_max
        self.batch_size=batch_size
        self.Nx=Nx
        self.N2=N2
        self.skip1=nn.Conv1d(in_channels=N1, out_channels=N2, kernel_size=1)
        self.skip2=nn.Conv1d(in_channels=N1, out_channels=N2, kernel_size=1)
        
    def forward(self, x):
        y=x.clone()
        
        x=torch.fft.fftn(x, dim=-1)
        x=torch.fft.fftshift(x, dim=-1)
        
        center= self.Nx//2
        start=center-self.k_max//2
        end=center+self.k_max//2
        
        x=x[:, :, start:end]
        x=torch.einsum('bcx,cdx->bdx', x, self.spec_wt_tensor)#[batch in_channel x][in_channel out_channel x]=[batch out_channel x]
        out=torch.zeros(self.batch_size, self.N2, self.Nx, dtype=torch.cfloat, device=x.device)
        out[:, :, start:end]=x
        
        x=torch.fft.ifftshift(out, dim=-1)
        x=torch.fft.ifftn(x, dim=-1)
        x=x.real+self.skip1(y)
        x=self.Sigmoid(x)
        
        x=self.channel_mix(x)+self.skip2(y)
                
        x=self.Sigmoid(x)
        return x

class FNO_1D(nn.Module):
    def __init__(self):
        super(FNO_1D, self).__init__()
        self.Sigmoid=nn.GELU()
        self.lift=nn.Conv1d(in_channels= Cin, out_channels= N1, kernel_size= 1)
        self.project=nn.Conv1d(in_channels= N2, out_channels= Cout, kernel_size= 1)
        self.layers=nn.ModuleList([FNO_Layer(k_max=128, mesh_dim=1, N1=N1, N2=N2, Nx=Nx, batch_size=b) for _ in range(num_layers)])

    def forward(self, x):
        x=self.lift(x)
        x=self.Sigmoid(x)
    
        for layer in self.layers:
            x=layer(x)
    
        x=self.project(x)
        x=self.Sigmoid(x)
        return x
    

from scipy.io import loadmat

# Load only metadata to see variable names and shapes without loading all data into RAM
data = loadmat('/kaggle/input/datasets/scaomath/pde-dataset/burgers_data_R10.mat')
print("Variables in file:")
for var in data:
    print(var)

for var in ['a', 'u', 'a_x']:
    print(f"{var} shape: {data[var].shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FNO_1D().to(device)
model.train()
epochs=100
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion = nn.MSELoss()
print("started training")


from torch.utils.data import DataLoader, TensorDataset, Subset

inputs = torch.from_numpy(np.stack([data['a'], np.column_stack(( data['a_x'], np.array([0]*2048) ))], axis=1)).float()
targets = torch.from_numpy(data['u']).float().unsqueeze(1)
dataset = TensorDataset(inputs, targets)
train_subset=Subset(dataset, range(512))
loader = DataLoader(train_subset, batch_size=b, shuffle=True)
losses=[]

try:
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
        #scheduler.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        for group in optimizer.param_groups:
            group['lr']/=1.1
except KeyboardInterrupt:
    print("Manual interruption detected. Cleaning up...")
finally:
    # This block runs regardless of how the script ends
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize() # Wait for all kernels to finish
    print("GPU Memory Cleared.")

import matplotlib.pyplot as plt
plt.plot(np.linspace(1, epochs, num=epochs), losses)
print(loss)
print("training done")
print("testing...")
