import torch
import Qstep_v3 as Qstep
import numpy as np
import time

p = Qstep.param(0.1, 0.00075, 0.2, 0.1, 0.1, 66*np.pi, 66*np.pi, 0.002, 1, 768, 768 )

Nx = p.Nx
Ny = p.Ny

nsize = int(Nx*Ny)
print(nsize)
w_init = np.genfromtxt("./init/220000w.csv",delimiter=",")[:,:-1]
r1_init = np.genfromtxt("./init/220000w.csv",delimiter=",")[:,:-1]
r2_init = np.genfromtxt("./init/220000w.csv",delimiter=",")[:,:-1]
w = torch.rand(nsize,device="cuda:0")
r1 = torch.rand(nsize,device="cuda:0")
r2 = torch.rand(nsize,device="cuda:0")
phys = torch.rand(nsize,device="cuda:0")

w = w.double()
r1 = r1.double()
r2 = r2.double()

sol = Qstep.Qsolver()
sol.init(p,16,w,r1,r2)
# Qstep.torch_Qinit(sol, p, 16, w, r1, r2)
start = time.time()

for i in range(150001):
    sol.step()
    if (i==0):
        np.savetxt(str(i)+"w.csv",w.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
        np.savetxt(str(i)+"r1.csv",r1.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
        np.savetxt(str(i)+"r2.csv",r2.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
    elif((i%1000 == 0) and (i!=0)):
        # Qstep.torch_Qstep(w,r1,r2,1,1000,False,True)
        # phys = w.clone().reshape([Nx, Ny])
        np.savetxt(str(i)+"w.csv",w.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
        np.savetxt(str(i)+"r1.csv",r1.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
        np.savetxt(str(i)+"r2.csv",r2.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
    else:
        pass
    print("=============={}=============".format(i))

end = time.time()

print("time used: %f",format(end-start))