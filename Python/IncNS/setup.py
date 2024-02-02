from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="Incstep",
    include_dirs=["include","../.."],
    ext_modules=[
        CUDAExtension(
            name = "Incstep",
            sources = ["kernel/timestep.cpp", "kernel/timestep_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)