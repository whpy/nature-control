#include "Qsolver.cuh"
void winit(Qreal *w, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            w[index] = -2*sin(x+y); 
        }
    }
}

void r1init(Qreal *r1, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            r1[index] = sin(x); 
        }
    }
}

void r2init(Qreal *r2, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            r2[index] = cos(x); 
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

void nl0exact(Qreal *nl0, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            nl0[index] = -60*cos(x);
        }
    }
}

void nl1exact(Qreal *nl1, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            nl1[index] = -4*sin(x) + 0.2*sin(x+y) + cos(x)*(cos(x+y) + 2*sin(x+y));
        }
    }
}

void nl2exact(Qreal *nl2, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            nl2[index] = -4*cos(x) - sin(x)*(cos(x+y) + 2*sin(x+y));
        }
    }
}
int main(){
    Qreal lambda = 0.1;
    Qreal Rf = 0.000075;
    Qreal alpha = 0.2;
    Qreal Re = 0.1;
    Qreal Er = 0.1;
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

    Rainit(alpha, phys_h, Nx, Ny, dx, dy);
    cudaMemcpy(Ra->phys, phys_h, physize, cudaMemcpyHostToDevice);
    cudaMemcpy(phys_h, Ra->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "Ra.csv", Nx, Ny);

    nl0exact(phys_h, Nx, Ny, dx, dy);
    field_visual(phys_h, "nl0e.csv", Nx, Ny);

    nl1exact(phys_h, Nx, Ny, dx, dy);
    field_visual(phys_h, "nl1e.csv", Nx, Ny);

    nl2exact(phys_h, Nx, Ny, dx, dy);
    field_visual(phys_h, "nl2e.csv", Nx, Ny);

    // FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(wold->phys, 0.0, Nx, Ny, BSZ);
    // FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(r1old->phys, 0.0, Nx, Ny, BSZ);
    // FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(r2old->phys, 0.0, Nx, Ny, BSZ);
    for (int i=0; i<100; i++){

        wnonl_func(wnonl, wold, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        BwdTrans(wnonl->phys, wnonl->spec, mesh);
        r1nonl_func(r1nonl, wold, u, v, r1old, r2old, S, lambda, aux, aux1);
        BwdTrans(r1nonl->phys, r1nonl->spec, mesh);
        r2nonl_func(r2nonl, wold, u, v, r1old, r2old, S, lambda, aux, aux1);
        BwdTrans(r2nonl->phys, r2nonl->spec, mesh);
        cout << i << endl;
    }
    wnonl_func(wnonl, wold, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
    BwdTrans(wnonl->phys, wnonl->spec, mesh);
    cudaMemcpy(phys_h, wnonl->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "nl0.csv", Nx, Ny);

    r1nonl_func(r1nonl, wold, u, v, r1old, r2old, S, lambda, aux, aux1);
    BwdTrans(r1nonl->phys, r1nonl->spec, mesh);
    cudaMemcpy(phys_h, r1nonl->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "nl1.csv", Nx, Ny);

    r2nonl_func(r2nonl, wold, u, v, r1old, r2old, S, lambda, aux, aux1);
    BwdTrans(r2nonl->phys, r2nonl->spec, mesh);
    cudaMemcpy(phys_h, r2nonl->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "nl2.csv", Nx, Ny);

    // p11func(p11, r1old, S, Ra, lambda);
    // BwdTrans(p11->phys,p11->spec,mesh);
    // cudaMemcpy(phys_h, p11->phys, physize, cudaMemcpyDeviceToHost);
    // field_visual(phys_h, "p11.csv", Nx, Ny);

    // p12func(p12, r1old, r2old, S, Ra, lambda, aux, aux1);
    // BwdTrans(p12->phys,p12->spec,mesh);
    // cudaMemcpy(phys_h, p12->phys, physize, cudaMemcpyDeviceToHost);
    // field_visual(phys_h, "p12.csv", Nx, Ny);

    // p21func(p21, r1old, r2old, S, Ra, lambda, aux, aux1);
    // BwdTrans(p21->phys,p21->spec,mesh);
    // cudaMemcpy(phys_h, p21->phys, physize, cudaMemcpyDeviceToHost);
    // field_visual(phys_h, "p21.csv", Nx, Ny);

    // FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(wnonl->phys, 0.0, Nx, Ny, BSZ);
    // FwdTrans(wnonl->spec, wnonl->phys, mesh);
    // PiDeriv<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wnonl->spec, p12->spec, p11->spec, p21->spec, mesh->kx, mesh->ky, Er, Re, Nxh, Ny, BSZ);
    // BwdTrans(wnonl->phys, wnonl->spec, mesh);
    // cudaMemcpy(phys_h, wnonl->phys, physize, cudaMemcpyDeviceToHost);
    // field_visual(phys_h, "PiDeriv.csv", Nx, Ny);

    // Field* tmp = new Field(mesh);
    // Field* tmp1 = new Field(mesh);
    // xDeriv(tmp->spec, p11->spec,mesh);
    // xDeriv(tmp->spec, tmp->spec, mesh);

    // yDeriv(tmp1->spec, p21->spec,mesh);
    // yDeriv(tmp1->spec, tmp->spec, mesh);

    // SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(tmp->spec, 1.0/(Re*Er), tmp->spec, -1.0/(Re*Er), tmp1->spec, Nxh, Ny, BSZ);
    // xDeriv(tmp1->spec, p11->spec,mesh);
    // yDeriv(tmp1->spec, tmp->spec, mesh);
    // SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(tmp->spec, 1.0, tmp->spec, -2.0/(Re*Er), tmp1->spec, Nxh, Ny, BSZ);
    // BwdTrans(tmp->phys, tmp->spec, mesh);
    // cudaMemcpy(phys_h, tmp->phys, physize, cudaMemcpyDeviceToHost);
    // field_visual(phys_h, "PiDeriv1.csv", Nx, Ny);
    // cudaMemcpy(spec_h, wnonl->spec, specsize, cudaMemcpyDeviceToHost);
    // print_spec(spec_h, Nxh, Ny);
    // cudaMemcpy(spec_h, tmp->spec, specsize, cudaMemcpyDeviceToHost);
    // print_spec(spec_h, Nxh, Ny);

    // cout << "p11" << endl;
    // cudaMemcpy(spec_h, p11->spec, specsize, cudaMemcpyDeviceToHost);
    // print_spec(spec_h, Nxh, Ny);

    // cout << "p21" << endl;
    // cudaMemcpy(spec_h, p21->spec, specsize, cudaMemcpyDeviceToHost);
    // print_spec(spec_h, Nxh, Ny);

    // cout << "p12" << endl;
    // cudaMemcpy(spec_h, p12->spec, specsize, cudaMemcpyDeviceToHost);
    // print_spec(spec_h, Nxh, Ny);


    delete mesh;
    return 0;
}