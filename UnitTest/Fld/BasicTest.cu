/*test the fwd and bwd function */


#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <cuComplexBinOp.cuh>
#include <cudaErr.h>
#include <QActFlowDef.cuh>

#include <Basic/Mesh.cuh>
#include <Qfunctions.cuh>
#include <QFldfuncs.cuh>

// #define M_PI 3.141592653589

using namespace std;
void winit(Qreal *w, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            w[index] = -2*cos(x)*cos(y); 
        }
    }
}

int main(){
    int Nx = 8*3/2;
    int Ny = Nx;
    int BSZ = 8;
    int Nxh = Nx/2+1;
    int specsize = Nxh*Ny*sizeof(Qcomp);
    int physize = Nx*Ny*sizeof(Qreal);
    int wavesize = Nxh*Ny*sizeof(Qreal);
    Qreal Lx = 2*M_PI;
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

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);
    coord(dx, dy, Nx, Ny);
    
    
    Qreal *w;
    Qcomp *w_c, *u_c, *v_c;
    Qreal *u, *v;
    Qreal *kx, *ky, *k_squared;

    Qreal *p1, *p2, *p3;
    Qcomp *sp1, *sp2, *sp3;
    cudaMalloc((void**)&w, sizeof(Qreal)*Nx*Ny);
    cudaMalloc((void**)&u, sizeof(Qreal)*Nx*Ny);
    cudaMalloc((void**)&v, sizeof(Qreal)*Nx*Ny);

    cudaMalloc((void**)&w_c, sizeof(Qcomp)*Nxh*Ny);
    cudaMalloc((void**)&u_c, sizeof(Qcomp)*Nxh*Ny);
    cudaMalloc((void**)&v_c, sizeof(Qcomp)*Nxh*Ny);

    cudaMalloc((void**)&kx, sizeof(Qreal)*Nxh*Ny);
    cudaMalloc((void**)&ky, sizeof(Qreal)*Nxh*Ny);
    cudaMalloc((void**)&k_squared, sizeof(Qreal)*Nxh*Ny);

    p1 = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    p2 = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    p3 = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    sp1 = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    sp2 = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny);
    sp3 = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny);

    Qreal *kxh = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);
    Qreal *kyh = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);
    Qreal *ksh = (Qreal*)malloc(sizeof(Qreal)*Nxh*Ny);
    

    cudaMemcpy(kxh, mesh->cutoff, sizeof(Qreal)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cout << "cutoff" << endl;
    print_spec(kxh, Nxh, Ny);
    winit(p1, Nx, Ny, dx, dy);
    cudaMemcpy(w, p1, sizeof(Qreal)*Nx*Ny, cudaMemcpyHostToDevice);
    FwdTrans(w_c, w, mesh);
    cudaMemcpy(p2, w, sizeof(Qreal)*Nx*Ny, cudaMemcpyDeviceToHost);
    field_visual(p2, "w0.csv", Nx, Ny);

    cudaMemcpy(sp2, w_c, sizeof(Qcomp)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cuda_error_func(cudaDeviceSynchronize());
    cout << "w_c" << endl;
    print_spec(sp2, Nxh, Ny);
    cuda_error_func(cudaDeviceSynchronize());
    
    BwdTrans(w, w_c, mesh);

    cudaMemcpy(p2, w, sizeof(Qreal)*Nx*Ny, cudaMemcpyDeviceToHost);
    field_visual(p2, "w.csv", Nx, Ny);

    cufft_error_func( cufftExecD2Z(transf, w, w_c));
    cudaMemcpy(sp2, w_c, sizeof(Qcomp)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cuda_error_func(cudaDeviceSynchronize());
    cout << "w_c 3" << endl;
    print_spec(sp2, Nxh, Ny);
    cuda_error_func(cudaDeviceSynchronize());

    waveinit(kxh, kyh, ksh, Nxh, Ny, Lx, Ly);
    cudaMemcpy(kx, kxh, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(ky, kyh, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(k_squared, ksh, sizeof(Qreal)*Nxh*Ny, cudaMemcpyHostToDevice);

    cudaMemcpy(kxh, mesh->kx, sizeof(Qreal)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cout << "kx" << endl;
    print_spec(kxh,Nxh,Ny);
    cudaMemcpy(kyh, mesh->ky, sizeof(Qreal)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cout << "ky" << endl;
    print_spec(kyh,Nxh,Ny);
    cudaMemcpy(ksh, mesh->k_squared, sizeof(Qreal)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cout << "k_squared" << endl;
    print_spec(ksh,Nxh,Ny);

    cuda_error_func(cudaDeviceSynchronize());
    vel_func(w_c, u_c, v_c, mesh);
    cudaMemcpy(sp2, u_c, sizeof(Qcomp)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cuda_error_func(cudaDeviceSynchronize());
    cout << "u_c" << endl;
    print_spec(sp2, Nxh, Ny);
    cudaMemcpy(sp2, v_c, sizeof(Qcomp)*Nxh*Ny, cudaMemcpyDeviceToHost);
    cuda_error_func(cudaDeviceSynchronize());
    cout << "v_c" << endl;
    print_spec(sp2,Nxh, Ny);
    cuda_error_func(cudaDeviceSynchronize());
    BwdTrans(u,u_c,mesh);
    BwdTrans(v,v_c,mesh);
    
    cudaMemcpy(p1, u, sizeof(Qreal)*Nx*Ny, cudaMemcpyDeviceToHost);
    field_visual(p1, "u.csv", Nx, Ny);
    cudaMemcpy(p2, v, sizeof(Qreal)*Nx*Ny, cudaMemcpyDeviceToHost);
    field_visual(p2, "v.csv", Nx, Ny);

    delete mesh;
    return 0;

}
