#include <torch/extension.h>
#include "nonl_ker.h"

// void nonl_ker(Qreal* nl0, Qreal *nl1, Qreal *nl2);

void torch_Qnonl(torch::Tensor &nl0, torch::Tensor &nl1, torch::Tensor &nl2){

    nonl_ker((double*)nl0.data_ptr(), (double*)nl1.data_ptr(), (double*)nl2.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_Qnonl",
          &torch_Qnonl,
          "nonlinear kernel warpper");
}

// void torch_launch_add2(torch::Tensor &c,
//                        const torch::Tensor &a,
//                        const torch::Tensor &b,
//                        int n) {
//     launch_add2((float *)c.data_ptr(),
//                 (const float *)a.data_ptr(),
//                 (const float *)b.data_ptr(),
//                 n);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("torch_launch_add2",
//           &torch_launch_add2,
//           "add2 kernel warpper");
// }