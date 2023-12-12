#ifndef QFLDFUNCS_CUH
#define QFLDFUNCS_CUH

#include <Basic/Mesh.cuh>
#include <Basic/cuComplexBinOp.cuh>
#include <cublas_v2.h>

__global__ 
void coeff(Qreal *f, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}

inline void FwdTrans(Qcomp* ft, Qreal* f, Mesh* mesh){
    cudaMemcpy(mesh->phys, f, mesh->physsize, cudaMemcpyDeviceToDevice);
    cufft_error_func( cufftExecD2Z(mesh->transf, mesh->phys, ft));
}

inline void BwdTrans( Qreal* f, Qcomp* ft, Mesh* mesh){
    cudaMemcpy(mesh->spec, ft, mesh->specsize, cudaMemcpyDeviceToDevice);
    cufft_error_func( cufftExecZ2D(mesh->inv_transf, mesh->spec, f));
    coeff<<<mesh->dimGridp,mesh->dimBlockp>>>(f, mesh->Nx, mesh->Ny, mesh->BSZ);
}


////////////////////////////////////////////////////////////////////////////
//FUNCTIONS TO assist field and spectrum operating//
///////////////////////////////////////////////////////////////////////////

// __Device: phys Field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
__global__ 
 void FldMul(Qreal* pc, Qreal* pa, Qreal* pb, Qreal C, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = C*pa[index]*pb[index];
    }
}

// __Device: phys Field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
__global__  
 void FldAdd(Qreal* pc, Qreal a, Qreal* pa, Qreal b, Qreal* pb, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
    }
}

// __Device: pc = a*pa + b
__global__
 void FldAdd(Qreal* pc, Qreal a, Qreal* pa, Qreal b, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b;
    }
}

// __Device: set physical Field equals to a constant Field
__global__
 void FldSet(Qreal *pf, Qreal c, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pf[index] = c;
    }
}

// __Device: set two physical Field equals
__global__
 void FldSet(Qreal * pf, Qreal* c, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pf[index] = c[index];
    }
}

// __Device: set two spectrum equal
__global__
 void SpecSet(Qcomp * pa, Qcomp* pb, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        pa[index] = pb[index];
    }
}

// __Device: set a constant spectrum 
__global__
 void SpecSet(Qcomp * pa, Qcomp c, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        pa[index] = c;
    }
}


// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
__global__
 void SpecAdd(Qcomp* spc, Qreal a, Qcomp* spa, Qreal b, Qcomp* spb, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = a*spa[index] + b*spb[index];
    }
}

__global__ 
void SpecMul(Qcomp *spc, Qcomp *spa, Qreal C, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index];
    }
}


////////////////////////////////////////////////////////////////////////////
//FUNCTIONS ABOUT DERIVATIVES//
///////////////////////////////////////////////////////////////////////////
__global__
void xDerivD(Qcomp *fx, Qcomp *f, Qreal* kx, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        fx[index] = im()*kx[index]*f[index];
    }
}
inline void xDeriv(Qcomp *fx, Qcomp *f, Mesh* mesh){
    xDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(fx, f, mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
}

__global__
void yDerivD(Qcomp *fy, Qcomp *f, Qreal* ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        fy[index] = im()*ky[index]*f[index];
    }
}
inline void yDeriv(Qcomp *fy, Qcomp *f, Mesh* mesh){
    yDerivD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(fy, f, mesh->ky, mesh->Nxh, mesh->Ny, mesh->BSZ);
}

__global__
void LaplacianD(Qcomp *fL, Qcomp *f, Qreal* k_squared, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        fL[index] = -1.0*k_squared[index]*f[index];
    }
}
inline void Laplacian(Qcomp *fL, Qcomp *f, Mesh* mesh){
    LaplacianD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(fL, f, mesh->k_squared, mesh->Nxh, mesh->Ny, mesh->BSZ);
}

__global__
void vel_funcD(Qcomp *w_c, Qcomp *u_c, Qcomp *v_c, 
Qreal* k_squared, Qreal* kx, Qreal*ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        if (i==0 && j==0)
        {
            u_c[index] = make_cuDoubleComplex(0.0,0.0);
            v_c[index] = make_cuDoubleComplex(0.0,0.0);
        }
        else{
            u_c[index] = ky[index]*im()*w_c[index]/(k_squared[index]);
            v_c[index] = -1.0*kx[index]*im()*w_c[index]/(k_squared[index]);
        }
    }
}
inline void vel_func(Qcomp *w_c, Qcomp *u_c, Qcomp *v_c, Mesh *mesh){
    vel_funcD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(w_c, u_c, v_c, mesh->k_squared, mesh->kx, mesh->ky, mesh->Nxh, mesh->Ny, mesh->BSZ);
}


#endif