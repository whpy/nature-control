import torch
import Qstep
import numpy as np

p = Qstep.param(0.1, 0.00075, 0.2, 0.1,0.1, 64*np.pi, 64*np.pi, 0.002, 1, 768, 768 )
# int Nx = 512*3/2;
# int Ny = Nx;

Nx = 512*3/2
Ny = Nx

nsize = int(Nx*Ny)
print(nsize)
w = torch.rand(nsize,device="cuda:0")
r1 = torch.rand(nsize,device="cuda:0")
r2 = torch.rand(nsize,device="cuda:0")

w = w.double()
r1 = r1.double()
r2 = r2.double()

Qstep.torch_Qstep(w,r1,r2,400001,1000,True)