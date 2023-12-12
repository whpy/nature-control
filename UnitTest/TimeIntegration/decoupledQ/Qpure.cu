#include "Qsolver.cuh"
#include <TimeIntegration/RK4.cuh>

void winit(Qreal *w, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            w[index] = 0.0; 
        }
    }
}

void r1init(Qreal *r1, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            r1[index] = 0.9*(sin(x+y)*sin(x+y) - 0.5); 
        }
    }
}

void r2init(Qreal *r2, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            r2[index] = 0.9*sin(x+y)*cos(x+y); 
        }
    }
}

void Rainit(Qreal alpha, Qreal *f, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            f[index] = alpha; 
        }
    }
}

// void nl0exact(Qreal *nl0, int Nx, int Ny, Qreal dx, Qreal dy){
//     for (int i=0; i<Nx; i++){
//         for (int j=0; j<Ny; j++){
//             Qreal x = i*dx;
//             Qreal y = j*dy;
//             int index = i + j*Nx;
//             nl0[index] = -60*cos(x);
//         }
//     }
// }

// void nl1exact(Qreal *nl1, int Nx, int Ny, Qreal dx, Qreal dy){
//     for (int i=0; i<Nx; i++){
//         for (int j=0; j<Ny; j++){
//             Qreal x = i*dx;
//             Qreal y = j*dy;
//             int index = i + j*Nx;
//             nl1[index] = -4*sin(x) + 0.2*sin(x+y) + cos(x)*(cos(x+y) + 2*sin(x+y));
//         }
//     }
// }

// void nl2exact(Qreal *nl2, int Nx, int Ny, Qreal dx, Qreal dy){
//     for (int i=0; i<Nx; i++){
//         for (int j=0; j<Ny; j++){
//             Qreal x = i*dx;
//             Qreal y = j*dy;
//             int index = i + j*Nx;
//             nl2[index] = -4*cos(x) - sin(x)*(cos(x+y) + 2*sin(x+y));
//         }
//     }
// }

int main(){
    Qreal lambda = 0.1;
    Qreal Rf = 0.0;
    Qreal alpha = 0.2;
    Qreal Re = 40;
    Qreal Er = 40;
    Qreal dt = 0.02;
    int Ns = 40001;

    int Nx = 512;
    int Ny = Nx;
    int BSZ = 16;
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

    Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);
    coord(dx, dy, Nx, Ny);

    Field *wold = new Field(mesh); Field *wcurr = new Field(mesh); Field *wnew = new Field(mesh); Field *wnonl = new Field(mesh);
    Field *phi = new Field(mesh);
    Field *u = new Field(mesh);
    Field *v = new Field(mesh);

    Field *r1old = new Field(mesh); Field *r1curr = new Field(mesh); Field *r1new = new Field(mesh); Field *r1nonl = new Field(mesh);
    Field *r2old = new Field(mesh); Field *r2curr = new Field(mesh); Field *r2new = new Field(mesh); Field *r2nonl = new Field(mesh);
    Field *S = new Field(mesh);

    Field* p11 = new Field(mesh); Field *p12 = new Field(mesh); Field *p21 = new Field(mesh);
    Field *Ra = new Field(mesh);

    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh);
    winit(phys_h, Nx, Ny, dx, dy);
    cudaMemcpy(wold->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(wold->spec, wold->phys, mesh);
    BwdTrans(wold->phys, wold->spec, mesh);
    cudaMemcpy(phys_h, wold->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "w.csv", Nx, Ny);

    r1init(phys_h, Nx, Ny, dx, dy);
    cudaMemcpy(r1old->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(r1old->spec, r1old->phys, mesh);
    BwdTrans(r1old->phys, r1old->spec, mesh);
    cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "r1.csv", Nx, Ny);

    r2init(phys_h, Nx, Ny, dx, dy);
    cudaMemcpy(r2old->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(r2old->spec, r2old->phys, mesh);
    BwdTrans(r2old->phys, r2old->spec, mesh);
    cudaMemcpy(phys_h, r2old->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "r2.csv", Nx, Ny);

    S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1old->phys, r2old->phys, Nx, Ny, BSZ);
    cudaMemcpy(phys_h, S->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "S.csv", Nx, Ny);

    Rainit(alpha, phys_h, Nx, Ny, dx, dy);
    cudaMemcpy(Ra->phys, phys_h, physize, cudaMemcpyHostToDevice);
    cudaMemcpy(phys_h, Ra->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "Ra.csv", Nx, Ny);

    Qreal *IFw, *IFwh;
    cudaMalloc((void**)&IFw, mesh->wavesize);
    cudaMalloc((void**)&IFwh, mesh->wavesize);

    Qreal *IFr1, *IFr1h;
    cudaMalloc((void**)&IFr1, mesh->wavesize);
    cudaMalloc((void**)&IFr1h, mesh->wavesize);

    Qreal *IFr2, *IFr2h;
    cudaMalloc((void**)&IFr2, mesh->wavesize);
    cudaMalloc((void**)&IFr2h, mesh->wavesize);

    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFw, IFwh, mesh->k_squared, Re, Rf, dt, Nxh, Ny, BSZ);
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFr1, IFr1h, mesh->k_squared, dt, Nxh, Ny, BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFr2, IFr2h, mesh->k_squared, dt, Nxh, Ny, BSZ);

    for (int m=0; m<Ns; m++){
        // integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);

        // wnonl_func(wnonl, wcurr, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        // integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // wnonl_func(wnonl, wcurr, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        // integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        // wnonl_func(wnonl, wcurr, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        // integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // wnonl_func(wnonl, wcurr, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wold, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        // integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        // SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, Nxh, Ny, BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, Nxh, Ny, BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, Nxh, Ny, BSZ);

        if(m%1 == 0) cout << "t = " << m*dt << endl;
        if (m%20 == 0){
            // BwdTrans(wold->phys, wold->spec, mesh);
            BwdTrans(r1old->phys, r1old->spec, mesh);
            cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m)+"r1.csv", Nx, Ny);

            BwdTrans(r2old->phys, r2old->spec, mesh);
            cudaMemcpy(phys_h, r2old->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m)+"r2.csv", Nx, Ny);

            S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1old->phys, r2old->phys, Nx, Ny, BSZ);
            cudaMemcpy(phys_h, S->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m)+"S.csv", Nx, Ny);
        }
    }

    delete mesh;
    return 0;
}