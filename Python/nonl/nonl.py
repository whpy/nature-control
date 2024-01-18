import torch
import Qnonl
import numpy as np

n = 512*512

nl0 = torch.rand(n,device="cuda:0")
nl1 = torch.rand(n,device="cuda:0")
nl2 = torch.rand(n,device="cuda:0")

nl0 = nl0.double()
nl1 = nl1.double()
nl2 = nl2.double()

Qnonl.torch_Qnonl(nl0,nl1,nl2)
torch.cuda.synchronize()
nl0 = nl0.reshape([512,512])
nl1 = nl1.reshape([512,512])
nl2 = nl2.reshape([512,512])

print(nl0)
print(nl1)
print(nl2)

np.savetxt("tensor_nl0.csv",nl0.cpu().numpy(),fmt="%.5f",delimiter=",")
np.savetxt("tensor_nl1.csv",nl1.cpu().numpy(),fmt="%.5f",delimiter=",")
np.savetxt("tensor_nl2.csv",nl2.cpu().numpy(),fmt="%.5f",delimiter=",")

