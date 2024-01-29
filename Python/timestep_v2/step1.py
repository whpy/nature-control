import torch
import Qstep
import numpy as np

# int Nx = 512*3/2;
# int Ny = Nx;

Nx = int(512*3/2)
Ny = Nx

nsize = int(Nx*Ny)
print(nsize)
w = torch.rand(nsize,device="cuda:0")
r1 = torch.rand(nsize,device="cuda:0")
r2 = torch.rand(nsize,device="cuda:0")
phys = torch.rand(nsize,device="cuda:0")

w = w.double()
r1 = r1.double()
r2 = r2.double()

for i in range(150001):
    if (i==0):
        Qstep.torch_Qstep(w,r1,r2,1,1000,True,True)
    elif((i%1000 == 0) and (i!=0)):
        Qstep.torch_Qstep(w,r1,r2,1,1000,False,True)
        # phys = w.clone().reshape([Nx, Ny])
        np.savetxt(str(i)+"w.csv",w.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
        np.savetxt(str(i)+"r1.csv",r1.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
        np.savetxt(str(i)+"r2.csv",r2.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
    else:
        Qstep.torch_Qstep(w,r1,r2,1,1000,False,False)
    print("=============={}=============".format(i))

