#include <torch/extension.h>
#include <timestep.h>

// void nonl_ker(Qreal* nl0, Qreal *nl1, Qreal *nl2);
namespace py = pybind11;

void torch_Qstep(torch::Tensor &w, torch::Tensor &r1, torch::Tensor &r2, 
int Nsteps, int cpN, bool initflag, bool cpflag){
    timestep((double*)w.data_ptr(), (double*)r1.data_ptr(), (double*)r2.data_ptr(), Nsteps, cpN, initflag, cpflag);
}

void torch_Qinit(Qsolver &Qsol, param p, int BSZ, torch::Tensor &w, torch::Tensor &r1, torch::Tensor &r2){
    Qsol.init(p, BSZ, (Qreal*)w.data_ptr(), (Qreal*)r1.data_ptr(), (Qreal*)r2.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_Qstep",
          &torch_Qstep,
          "step forward function of Q equation system");

    m.def("torch_Qinit",
          &torch_Qinit,
          "a wrapper to initialize the Qsolver by torch tensor");

    py::class_<param>(m, "param")
    .def(py::init<Qreal, Qreal, Qreal, Qreal, 
    Qreal, Qreal, Qreal, Qreal, int, int, int>())
    .def_readwrite("lambda", &param::lambda)
    .def_readwrite("Rf", &param::Rf)
    .def_readwrite("alpha", &param::alpha)
    .def_readwrite("Re", &param::Re)
    .def_readwrite("Er", &param::Er)
    .def_readwrite("Lx", &param::Lx)
    .def_readwrite("Ly", &param::Ly)
    .def_readwrite("dt", &param::dt)
    .def_readwrite("Ns", &param::Ns)
    .def_readwrite("Nx", &param::Nx)
    .def_readwrite("Ny", &param::Ny)
    ;

    py::class_<Qsolver>(m, "Qsolver")
    .def(py::init<>())
    .def("init",&Qsolver::init)
    .def("step",&Qsolver::step)
    ;
}