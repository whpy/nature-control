/*test the derivative function*/

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <Basic/cuComplexBinOp.cuh>
#include <Basic/cudaErr.h>
#include <Basic/QActFlowDef.cuh>

#include <Basic/Mesh.cuh>
#include <BasicUtils/BasicUtils.cuh>
#include <FldOp/QFldfuncs.cuh>

// #define M_PI 3.141592653589

using namespace std;
inline void phiinit(Qreal* phi, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            Qreal x = i*dx;
            Qreal y = j*dy;
            phi[index] = 1.0*sin(x+y);
        }
    }
}

void wexact(Qreal *w, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            Qreal x = i*dx;
            Qreal y = j*dy;
            w[index] = -2.0*sin(x+y);
        }
    }
}

void vexact(Qreal *v, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            Qreal x = i*dx;
            Qreal y = j*dy;
            v[index] = cos(x+y);
        }
    }
}

void uexact(Qreal *u, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            Qreal x = i*dx;
            Qreal y = j*dy;
            u[index] = -1.0*cos(x+y);
        }
    }
}

int main(){
    int Nx = 512;
    int Ny = Nx;
    int BSZ = 16;
    int Nxh = Nx/2+1;
    int specsize = Nxh*Ny*sizeof(Qcomp);
    int physize = Nx*Ny*sizeof(Qreal);
    int wavesize = Nxh*Ny*sizeof(Qreal);
    Qreal Lx = 33*2*M_PI;
    Qreal Ly = Lx;
    Qreal dx = Lx/Nx;
    Qreal dy = Ly/Ny;

    cufftHandle transf;
    cufftHandle inv_transf;
    cufft_error_func( cufftPlan2d( &(transf), Ny, Nx, CUFFT_D2Z ) );
    cufft_error_func( cufftPlan2d( &(inv_transf), Ny, Nx, CUFFT_Z2D ) );

    dim3 dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dim3 dimBlockp = dim3(BSZ, BSZ);

    dim3 dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dim3 dimBlocksp = dim3(BSZ, BSZ);

    Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);
    coord(dx, dy, Nx, Ny);

    Field *w = new Field(mesh);
    Field *phi = new Field(mesh);
    Field *u = new Field(mesh);
    Field *v = new Field(mesh);
    FldSet<<<dimGridp, dimBlockp>>>(phi->phys, Qreal(0.0), Nx, Ny, BSZ);
    cudaMemcpy(phys_h, phi->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    cuda_error_func(cudaDeviceSynchronize());
    field_visual(phys_h, "zeroset.csv", Nx, Ny);

    for (int i = 0; i<1; i++){
        cout << i << endl;
        // prepare phi and the exact solution
    phiinit(phys_h, Nx, Ny, dx, dy);
    cudaMemcpy(phi->phys, phys_h, mesh->physsize, cudaMemcpyHostToDevice);
    field_visual(phys_h, "phiexact.csv", Nx, Ny);
    
    uexact(phys_h, Nx, Ny, dx, dy);
    field_visual(phys_h, "uexact.csv", Nx, Ny);

    vexact(phys_h, Nx, Ny, dx, dy);
    field_visual(phys_h, "vexact.csv", Nx, Ny);

    wexact(phys_h, Nx, Ny, dx, dy);
    field_visual(phys_h, "wexact.csv", Nx, Ny);


    FwdTrans(phi->spec, phi->phys, mesh);
    BwdTrans(phi->phys, phi->spec, mesh);
    cudaMemcpy(phys_h, phi->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "phi.csv",Nx, Ny);

    Laplacian(w->spec, phi->spec, mesh);
    BwdTrans(w->phys, w->spec, mesh);
    cudaMemcpy(phys_h, w->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "w.csv",Nx, Ny);

    xDeriv(v->spec, phi->spec, mesh);
    BwdTrans(v->phys,v->spec, mesh);
    cudaMemcpy(phys_h, v->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "v.csv",Nx, Ny);

    yDeriv(u->spec, phi->spec, mesh);
    SpecMul<<<dimGridsp, dimBlocksp>>>(u->spec, u->spec, -1.0, Nxh, Ny, BSZ);
    BwdTrans(u->phys,u->spec, mesh);
    cudaMemcpy(phys_h, u->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "u.csv",Nx, Ny);

    vel_func(w->spec, u->spec, v->spec, mesh);
    BwdTrans(u->phys, u->spec, mesh);
    cudaMemcpy(phys_h, u->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "velu.csv",Nx, Ny);
    BwdTrans(v->phys, v->spec, mesh);
    cudaMemcpy(phys_h, v->phys, mesh->physsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "velv.csv",Nx, Ny);
    }
    




    
    delete mesh;
    return 0;

}

// int main(){
//     int Nx = 8;
//     int Ny = Nx;
//     int BSZ = 16;
//     int Nxh = Nx/2+1;
//     int specsize = Nxh*Ny*sizeof(Qcomp);
//     int physize = Nx*Ny*sizeof(Qreal);
//     int wavesize = Nxh*Ny*sizeof(Qreal);
//     Qreal Lx = 2*M_PI;
//     Qreal Ly = Lx;
//     Qreal dx = Lx/Nx;
//     Qreal dy = Ly/Ny;

//     cufftHandle transf;
//     cufftHandle inv_transf;
//     cufft_error_func( cufftPlan2d( &(transf), Ny, Nx, CUFFT_D2Z ) );
//     cufft_error_func( cufftPlan2d( &(inv_transf), Ny, Nx, CUFFT_Z2D ) );

//     dim3 dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
//     dim3 dimBlockp = dim3(BSZ, BSZ);

//     dim3 dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
//     dim3 dimBlocksp = dim3(BSZ, BSZ);

//     Qreal *phi;
//     Qreal *w;
//     Qcomp *w_c, *u_c, *v_c, *phi_c;
//     Qreal *u, *v;
//     Qreal *kx, *ky, *k_squared;

//     Qreal *p1, *p2, *p3;
//     Qcomp *sp1, *sp2, *sp3;
//     cudaMalloc((void**)&phi, sizeof(Qreal)*Nx*Ny);
//     cudaMalloc((void**)&w, sizeof(Qreal)*Nx*Ny);
//     cudaMalloc((void**)&u, sizeof(Qreal)*Nx*Ny);
//     cudaMalloc((void**)&v, sizeof(Qreal)*Nx*Ny);

//     cudaMalloc((void**)&phi_c, sizeof(Qcomp)*Nxh*Ny);
//     cudaMalloc((void**)&w_c, sizeof(Qcomp)*Nxh*Ny);
//     cudaMalloc((void**)&u_c, sizeof(Qcomp)*Nxh*Ny);
//     cudaMalloc((void**)&v_c, sizeof(Qcomp)*Nxh*Ny);

//     Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
//     Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
//     Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

//     Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);
//     coord(dx, dy, Nx, Ny);

//     // Field *w = new Field(mesh);
//     // Field *phi = new Field(mesh);
//     // Field *u = new Field(mesh);
//     // Field *v = new Field(mesh);
//     // prepare phi and the exact solution
//     phiinit(phys_h, Nx, Ny, dx, dy);
//     cudaMemcpy(phi, phys_h, mesh->physsize, cudaMemcpyHostToDevice);
//     field_visual(phys_h, "phiexact.csv", Nx, Ny);
    
//     uexact(phys_h, Nx, Ny, dx, dy);
//     field_visual(phys_h, "uexact.csv", Nx, Ny);

//     vexact(phys_h, Nx, Ny, dx, dy);
//     field_visual(phys_h, "vexact.csv", Nx, Ny);

//     wexact(phys_h, Nx, Ny, dx, dy);
//     field_visual(phys_h, "wexact.csv", Nx, Ny);

//     cudaMemcpy(w, phys_h, mesh->physsize, cudaMemcpyHostToDevice);
//     FwdTrans(phi_c, phi, mesh);
//     BwdTrans(phi, phi_c, mesh);
//     cudaMemcpy(phys_h, phi, mesh->physsize, cudaMemcpyDeviceToHost);
//     field_visual(phys_h, "phi.csv",Nx, Ny);

//     xDeriv(v_c, phi_c, mesh);
//     BwdTrans(v,v_c, mesh);
//     cudaMemcpy(phys_h, v, mesh->physsize, cudaMemcpyDeviceToHost);
//     field_visual(phys_h, "v.csv",Nx, Ny);

//     yDeriv(u_c, phi_c, mesh);
//     BwdTrans(u,u_c, mesh);
//     cudaMemcpy(phys_h, u, mesh->physsize, cudaMemcpyDeviceToHost);
//     field_visual(phys_h, "u.csv",Nx, Ny);


    
//     delete mesh;
//     return 0;

// }
