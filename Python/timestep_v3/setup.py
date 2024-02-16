from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="Qstep_v3",
    include_dirs=["include","../.."],
    ext_modules=[
        CUDAExtension(
            name = "Qstep_v3",
            sources = ["kernel/timestep.cpp", "kernel/timestep_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)