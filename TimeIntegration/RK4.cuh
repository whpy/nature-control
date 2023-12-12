#ifndef __RK4_CUH
#define __RK4_CUH

#include <Basic/cuComplexBinOp.cuh>
#include <Basic/cudaErr.h>
#include <Basic/QActFlowDef.cuh>

#include <Basic/Mesh.cuh>
#include <BasicUtils/BasicUtils.cuh>
#include <FldOp/QFldfuncs.cuh>

__global__
void integrate_func0D(Qcomp *old_spec, Qcomp* curr_spec, Qcomp* new_spec, 
Qreal *IF, Qreal *IFh, int Nxh, int Ny, int BSZ, Qreal dt){
    int i = blockIdx.x*BSZ + threadIdx.x;
    int j = blockIdx.y*BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if (i<Nxh && j<Ny){
        new_spec[index] = old_spec[index]*IF[index];
        curr_spec[index] = old_spec[index];
    }
}

__global__
void integrate_func1D(Qcomp *old_spec, Qcomp* curr_spec, Qcomp* new_spec, Qcomp *nonl_spec, 
Qreal *IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt){
    int i = blockIdx.x*BSZ + threadIdx.x;
    int j = blockIdx.y*BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if (i<Nxh && j<Ny){
        Qcomp an = nonl_spec[index]*dt;
        new_spec[index] = new_spec[index] + 1.0/6.0 * (IF[index])*an;
        curr_spec[index] = (old_spec[index] + an/2.0)* (IFh[index]);
    }
}

__global__
void integrate_func2D(Qcomp *old_spec, Qcomp* curr_spec, Qcomp* new_spec, Qcomp *nonl_spec, 
Qreal *IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt){
    int i = blockIdx.x*BSZ + threadIdx.x;
    int j = blockIdx.y*BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if (i<Nxh && j<Ny){
        Qcomp bn = nonl_spec[index]*dt; 
        new_spec[index] = new_spec[index] + 1.0/3.0 * IFh[index]*bn;
        curr_spec[index] = old_spec[index]* (IFh[index]) + bn/2.0;
    }
}

__global__
void integrate_func3D(Qcomp *old_spec, Qcomp* curr_spec, Qcomp* new_spec, Qcomp *nonl_spec, 
Qreal *IF, Qreal *IFh, int Nxh, int Ny, int BSZ, Qreal dt){
    int i = blockIdx.x*BSZ + threadIdx.x;
    int j = blockIdx.y*BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if (i<Nxh && j<Ny){
        Qcomp cn = nonl_spec[index]*dt;
        new_spec[index] = new_spec[index] + 1.0/3.0 * IFh[index]*cn;
        curr_spec[index] = old_spec[index]*IF[index] + cn*IFh[index];
    }
}

__global__
void integrate_func4D(Qcomp *old_spec, Qcomp* curr_spec, Qcomp* new_spec, Qcomp *nonl_spec, 
Qreal *IF, Qreal *IFh, int Nxh, int Ny, int BSZ, Qreal dt){
    int i = blockIdx.x*BSZ + threadIdx.x;
    int j = blockIdx.y*BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if (i<Nxh && j<Ny){
        Qcomp dn = nonl_spec[index]*dt;
        new_spec[index] = new_spec[index] + 1.0/6.0*dn;
        // curr_spec[index] = old_spec[index];
    }
}
#endif