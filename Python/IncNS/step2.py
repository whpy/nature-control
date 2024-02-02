import torch
import Incstep
import numpy as np
import time

# p = Qstep.param(0.1, 0.00075, 0.2, 0.1, 0.1, 66*np.pi, 66*np.pi, 0.002, 1, 768, 768 )
Ip = Incstep.Incparam(40, 2*np.pi, 2*np.pi, 0.02, 768, 768)
Nx = Ip.Nx
Ny = Ip.Ny

nsize = int(Nx*Ny)
print(nsize)
w = torch.rand(nsize,device="cuda:0")

w = w.double()

sol = Incstep.IncSolver()
sol.init(Ip, 16, w.data_ptr(), w)

start = time.time()
for i in range(150001):
    sol.step()
    if (i==0):
        np.savetxt(str(i)+"w.csv",w.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
    elif((i%1000 == 0) and (i!=0)):
        np.savetxt(str(i)+"w.csv",w.clone().reshape([Nx, Ny]).cpu().numpy().reshape([Nx,Ny]),fmt="%.6f",delimiter=",")
    else:
        pass
    print("=============={}=============".format(i))

end = time.time()

print("time used: %f",format(end-start))