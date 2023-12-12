#ifndef __MESH_CUH
#define __MESH_CUH
#include <Basic/QActFlowDef.cuh>
#include <Basic/cudaErr.h>

inline void waveinit(Qreal* kx, Qreal* ky, Qreal* k_squared, int Nxh, int Ny, Qreal Lx, Qreal Ly){
    for (int i = 0; i < Nxh; i++){
        for (int j = 0; j < Ny; j++){
            int index = i + j*Nxh;
            
            if (j<Ny/2+1){
                ky[index] = 2*M_PI/Ly * j;
            }
            else{
                ky[index] = 2*M_PI/Ly * (j-Ny);
            }
            kx[index] = 2*M_PI/Lx * i;
            k_squared[index] = kx[index]*kx[index] + ky[index]*ky[index]; 
        }
    }
}

inline 
void cutoff_func(Qreal *cutoff, int Nxh, int Ny){
    for (int i = 0; i < Nxh; i++){
        for (int j = 0; j < Ny; j++){
            int index = i + j*Nxh;
            if ( i < (Nxh-1)*2/3+1){
                if ( (j<Ny/3+1) || ((Ny-j)<Ny/3)) {
                    cutoff[index] = 1.0;
                }
            }
            else{
                cutoff[index] = 0.0;
            }
            if ( (i < ((Nxh-1)*2/3+1) )&&( (j<Ny/3+1) || ((Ny-j)<Ny/3)) )
            {
                cutoff[index] = 1.0;
            }
            else{
                cutoff[index] = 0.0;
            }

        }
    }
}

typedef struct Mesh{
    int Nx, Nxh, Ny;
    Qreal Lx, Ly, dx, dy;
    int specsize, physsize, wavesize;

    int BSZ;
    dim3 dimGridp, dimBlockp, dimGridsp, dimBlocksp;
    
    cufftHandle transf, inv_transf;

    Qcomp* spec; Qreal* phys;
    Qreal *kx, *ky, *k_squared;
    Qreal *cutoff;
    Mesh(int pNx, int pNy, Qreal pLx, Qreal pLy, int pBSZ):
    Nx(pNx),Ny(pNy),Nxh(Nx/2+1),BSZ(pBSZ), Lx(pLx), Ly(pLy), dx(pLx/Nx), dy(pLy/Ny)
    {
        
        specsize = Nxh*Ny*sizeof(Qcomp);
        physsize = Nx*Ny*sizeof(Qreal);
        wavesize = Nxh*Ny*sizeof(Qreal);
        cudaMalloc((void**)&spec, specsize);
        cudaMalloc((void**)&phys, physsize);

        
        cufft_error_func( cufftPlan2d( &(transf), Ny, Nx, CUFFT_D2Z ) );
        cufft_error_func( cufftPlan2d( &(inv_transf), Ny, Nx, CUFFT_Z2D ) );

        // thread information for physical space
        dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
        dimBlockp = dim3(BSZ, BSZ);

        // thread information for spectral space
        dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
        dimBlocksp = dim3(BSZ, BSZ);


        cudaMalloc((void**)&kx, sizeof(Qreal)*Nxh*Ny);
        cudaMalloc((void**)&ky, sizeof(Qreal)*Nxh*Ny);
        cudaMalloc((void**)&k_squared, sizeof(Qreal)*Nxh*Ny);
        cudaMalloc((void**)&cutoff, sizeof(Qreal)*Nxh*Ny);

        Qreal *kxh = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);
        Qreal *kyh = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);
        Qreal *ksh = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);
        Qreal *cutoff_h = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);

        waveinit(kxh, kyh, ksh, Nxh, Ny, Lx, Ly);
        cutoff_func(cutoff_h, Nxh, Ny);

        cudaMemcpy(kx, kxh, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);
        cudaMemcpy(ky, kyh, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);
        cudaMemcpy(k_squared, ksh, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);
        cudaMemcpy(cutoff, cutoff_h, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);

        free(kxh);
        free(kyh);
        free(ksh);
        free(cutoff_h);

    }

    ~Mesh(){
        cudaFree(spec);
        cudaFree(phys);
    }
} Mesh;

typedef struct Field{
    Qcomp* spec;
    Qreal* phys;
    Mesh* mesh;

    Field(Mesh* pmesh):mesh(pmesh){
        cudaMalloc((void**)&spec, mesh->specsize);
        cudaMalloc((void**)&phys, mesh->physsize);
    }

    ~Field(){
        cudaFree(spec);
        cudaFree(phys);
    }
}Field;
#endif