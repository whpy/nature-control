from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="Qnonl",
    include_dirs=["include","../.."],
    ext_modules=[
        CUDAExtension(
            name = "Qnonl",
            sources = ["kernel/nonl.cpp", "kernel/nonl_ker.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)