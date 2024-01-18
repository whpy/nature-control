#include <Qsolver.cuh>
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
void nonl_ker(Qreal* nl0, Qreal *nl1, Qreal *nl2){
    cout << "start " << endl;
    cudaError_t error;


    const Qreal lambda = 0.1;
    const Qreal Rf = 0.000075;
    const Qreal alpha = 0.2;
    const Qreal Re = 0.1;
    const Qreal Er = 0.1;
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

    static Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    static Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    static Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    static Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);
    coord(dx, dy, Nx, Ny);

    static Field *wold = new Field(mesh); static Field *wcurr = new Field(mesh); static Field *wnew = new Field(mesh); Field *wnonl = new Field(mesh);
    static Field *phi = new Field(mesh);
    static Field *u = new Field(mesh);
    static Field *v = new Field(mesh);

    static Field *r1old = new Field(mesh); static Field *r1curr = new Field(mesh); static Field *r1new = new Field(mesh); Field *r1nonl = new Field(mesh);
    static Field *r2old = new Field(mesh); static Field *r2curr = new Field(mesh); static Field *r2new = new Field(mesh); Field *r2nonl = new Field(mesh);
    static Field *S = new Field(mesh);

    static Field* p11 = new Field(mesh); static Field *p12 = new Field(mesh); static Field *p21 = new Field(mesh);
    static Field *Ra = new Field(mesh);

    static Field *aux = new Field(mesh); static Field *aux1 = new Field(mesh);

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

    

    r2nonl_func(r2nonl, wold, u, v, r1old, r2old, S, lambda, aux, aux1);
    BwdTrans(r2nonl->phys, r2nonl->spec, mesh);
    error = cudaMemcpy(nl2, r2nonl->phys, physize, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        std::cout << "Error in cudaMalloc" << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }
    cudaMemcpy(phys_h, r2nonl->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "nl2.csv", Nx, Ny);

    ofstream t;
    t.open("cudaout.txt");

    wnonl->phys = nl0;
    t<< "wnonl&: " << wnonl->phys << "\n";
    t<< "nl0&: " << nl0 << "\n"; 
    t.close();
    wnonl_func(wnonl, wold, u, v, r1old, r2old, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
    BwdTrans(wnonl->phys, wnonl->spec, mesh);
    error = cudaMemcpy(nl0, wnonl->phys, physize, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        std::cout << "Error in cudaMalloc" << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }
    cudaMemcpy(phys_h, wnonl->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "nl0.csv", Nx, Ny);

    r1nonl_func(r1nonl, wold, u, v, r1old, r2old, S, lambda, aux, aux1);
    BwdTrans(r1nonl->phys, r1nonl->spec, mesh);
    error = cudaMemcpy(nl1, r1nonl->phys, physize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(phys_h, r1nonl->phys, physize, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cout << "Error in cudaMalloc" << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }
    field_visual(phys_h, "tnl1.csv", Nx, Ny);

    cudaDeviceSynchronize();

    // delete mesh;
    // return 0;
}