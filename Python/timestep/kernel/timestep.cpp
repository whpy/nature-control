#include <torch/extension.h>
#include <timestep.h>

// void nonl_ker(Qreal* nl0, Qreal *nl1, Qreal *nl2);

void torch_Qstep(torch::Tensor &w, torch::Tensor &r1, torch::Tensor &r2, 
int Nsteps, int cpN, bool initflag, bool cpflag){
    timestep((double*)w.data_ptr(), (double*)r1.data_ptr(), (double*)r2.data_ptr(), Nsteps, cpN, initflag, cpflag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_Qstep",
          &torch_Qstep,
          "step forward function of Q equation system");
}
