#ifndef __IFRK2_CUH
#define __IFRK2_CUH

#include <cuComplexBinOp.cuh>
#include <cudaErr.h>
#include <QActFlowDef.cuh>

#include <Mesh.cuh>
#include <Qfunctions.cuh>
#include <QFldfuncs.cuh>


namespace IF{
namespace IFRK2{

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
        new_spec[index] = new_spec[index] + 1.0/2.0 * (IF[index])*an;
        curr_spec[index] = (old_spec[index] + an)* (IFh[index]);
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
        new_spec[index] = new_spec[index] + 1.0/2.0 * bn;
    }
}


} // end of IFRK2
} // end of IF


#endif