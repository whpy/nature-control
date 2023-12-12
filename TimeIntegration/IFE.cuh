#ifndef __IFE_CUH
#define __IFE_CUH

#include <cuComplexBinOp.cuh>
#include <cudaErr.h>
#include <QActFlowDef.cuh>

#include <Mesh.cuh>
#include <Qfunctions.cuh>
#include <QFldfuncs.cuh>

namespace IF{
    namespace IFE{

__global__
void integrate_func1D(Qcomp *old_spec, Qcomp* new_spec, Qcomp *nonl_spec, 
Qreal *IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt){
    int i = blockIdx.x*BSZ + threadIdx.x;
    int j = blockIdx.y*BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if (i<Nxh && j<Ny){
        Qcomp an = nonl_spec[index]*dt;
        new_spec[index] = (old_spec[index]+an)*IF[index];
    }
}

    } // end of IFE namespace
}// end of IF namespace



#endif