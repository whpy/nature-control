import torch
import Incstep
import numpy as np
import time

# description:
# create a Incparam object to record the parameters of domain and parameters
#
# arguments:
# Incparam(double Re, double Lx, double Ly, double step_length, int Nx, int Ny)
# Re: Reynolds number; 
# Lx: side length of x-axis; 
# Ly: side length of y-axis; 
# step_length: length of per step; 
# Nx: number of cells on x-axis; 
# Ny: number of cells on y-axis

Ip = Incstep.Incparam(40, 2*np.pi, 2*np.pi, 0.02, 768, 768)
Nx = Ip.Nx
Ny = Ip.Ny

nsize = int(Nx*Ny)
print(nsize)

# description:
# create a tensor to store vorticity. Noted that it should be on the GPU 
# and the data type should be double
w = torch.rand(nsize,device="cuda:0")
w = w.double()

u = torch.rand(nsize,device="cuda:0")
u = u.double()

v = torch.rand(nsize,device="cuda:0")
v = v.double()

w0 = torch.rand(nsize,device="cuda:0")
w0 = w0.double()

# create the solver for NS equation
sol = Incstep.IncSolver()

# description:
# Initialization before computation
#
# arguments:
# sol.init(Incparam param, int BSZ, torch::tensor w)
# param: Incparam recording the parameters; 
# BSZ: control the thread number in each block, 16 performs empirically good
# w: torch.tensor to record the result at each step
sol.init(Ip, 16, w)

Incstep.VelConvert(u,v,w,Ip,16)

np.savetxt(str(0)+"u.csv",u.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
np.savetxt(str(0)+"v.csv",v.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")

Incstep.VortConvert(w0,u,v,Ip,16)
np.savetxt(str(0)+"w0.csv",w0.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")

