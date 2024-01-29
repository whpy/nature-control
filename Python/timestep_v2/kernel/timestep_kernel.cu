#include <Qsolver.cuh>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <TimeIntegration/RK4.cuh>

#include <timestep.h>
using namespace std;

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
            r1[index] = 0.2*(sin(x+y)*sin(x+y) - 0.5) + 0.001*(Qreal(rand())/RAND_MAX); 
        }
    }
}

void r2init(Qreal *r2, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            r2[index] = 0.2*sin(x+y)*cos(x+y) + 0.001*(Qreal(rand())/RAND_MAX); 
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

template <class T>
T string2num(const string& str){
    istringstream iss(str);
    T num;
    iss >> num;
    return num;
}

void file_init(string filename, Qreal* f, Mesh* mesh){
    int Nx = mesh->Nx;
    int Ny = mesh->Ny;

    ifstream infile(filename);
    if (!infile.good()){
        cout << "no such file!" << "\n";
        exit(0);
        return;
    }
    vector<vector<string>> data;
    string strline;
    while(getline(infile, strline)){
        stringstream ss(strline);
        string str;
        vector<string> dataline;

        while(getline(ss, str, ',')){
            dataline.push_back(str);
        }
        data.push_back(dataline);
    }

    if(data.size() != Ny || data[0].size() != Nx){
        printf("READ_CSV_ERROR: size not match! \n");
        return;
    }
    int index = 0;
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            index = j*Nx + i;
            f[index] = string2num<double>(data[j][i]);
        }
    }
    infile.close();
}

// typedef struct param{
//     Qreal lambda; Qreal Rf; Qreal alpha; Qreal Re; Qreal Er;
//     Qreal Lx; Qreal Ly; Qreal dt; int Ns; int Nx; int Ny;
// }param;

param::param(){};
param::param(Qreal plam, Qreal pRf, Qreal palpha, Qreal pRe, Qreal pEr, Qreal pLx, Qreal pLy, Qreal pdt,
int pNs, int pNx, int pNy):lambda(plam),Rf(pRf),alpha(palpha),Re(pRe),Er(pEr),Lx(pLx),Ly(pLy),dt(pdt),Ns(pNs),Nx(pNx),Ny(pNy){}

// class Qsolver
// {
//     public:
//     // store the geometry parameters
//     param parameters; Mesh *mesh; int BSZ;

//     // allocate memory on CPU
//     Qreal *phys_h; Qcomp *spec_h; Qreal *wave_h;

//     // fields of fluid
//     Field *wold; Field *wcurr; Field *wnew; Field *wnonl;
//     Field *phi; Field *u; Field *v;
//     // fields of director field
//     Field *r1old ; Field *r1curr ;  Field *r1new ;  Field *r1nonl ;
//     Field *r2old ;  Field *r2curr ;  Field *r2new ;  Field *r2nonl ;
//     Field *S ;

//     // auxiliary fields
//     Field* p11 ;  Field *p12 ;  Field *p21 ;
//     Field *Ra ;

//     Field *aux ;  Field *aux1 ;

//     // time step linear factors for independent variables
//     Qreal *IFw; Qreal *IFwh;
//     Qreal *IFr1; Qreal *IFr1h;
//     Qreal *IFr2; Qreal *IFr2h;

//     void init(param pParam, int pBSZ, Qreal* w, Qreal* r1, Qreal* r2);
//     void step();

//     private:
//     bool initflag;
//     // int specsize = Nxh*Ny*sizeof(Qcomp);
//     // int physize = Nx*Ny*sizeof(Qreal);
//     // int wavesize = Nxh*Ny*sizeof(Qreal);
//     // Qreal Lx = 33*2*M_PI;
//     // Qreal Ly = Lx;
//     // Qreal dx = Lx/Nx;
//     // Qreal dy = Ly/Ny;
    
//     // Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
//     // Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
//     // Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

//     // Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);

//     // Field *wold = new Field(mesh,w);  Field *wcurr = new Field(mesh);  Field *wnew = new Field(mesh);  Field *wnonl = new Field(mesh);
//     // Field *phi = new Field(mesh);
//     // Field *u = new Field(mesh);
//     // Field *v = new Field(mesh);

//     // Field *r1old = new Field(mesh, r1); Field *r1curr = new Field(mesh);  Field *r1new = new Field(mesh);  Field *r1nonl = new Field(mesh);
//     // Field *r2old = new Field(mesh, r2);  Field *r2curr = new Field(mesh);  Field *r2new = new Field(mesh);  Field *r2nonl = new Field(mesh);
//     // Field *S = new Field(mesh);

//     // Field* p11 = new Field(mesh);  Field *p12 = new Field(mesh);  Field *p21 = new Field(mesh);
//     // Field *Ra = new Field(mesh);

//     // Field *aux = new Field(mesh);  Field *aux1 = new Field(mesh);

//     // Qreal *IFw;
//     // Qreal *IFwh;
//     // Qreal *IFr1;
//     // Qreal *IFr1h;
//     // Qreal *IFr2;
//     // Qreal *IFr2h;
// };

void Qsolver::init(param pParam, int pBSZ, Qreal* w, Qreal* r1, Qreal* r2){
    this->parameters = pParam;
    this->BSZ = pBSZ;

    mesh = new Mesh(parameters.Nx, parameters.Ny, parameters.Lx, parameters.Ly, BSZ);

    int Nxh = mesh->Nxh;
    int Ny = mesh->Ny;
    int Nx = mesh->Nx;
    int specsize = Nxh*Ny*sizeof(Qcomp);
    int physize = Nx*Ny*sizeof(Qreal);
    int wavesize = Nxh*Ny*sizeof(Qreal);

    phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    

    wold = new Field(mesh,w);  wcurr = new Field(mesh); wnew = new Field(mesh); wnonl = new Field(mesh);
    phi = new Field(mesh);
    u = new Field(mesh);
    v = new Field(mesh);

    r1old = new Field(mesh, r1); r1curr = new Field(mesh);  r1new = new Field(mesh);  r1nonl = new Field(mesh);
    r2old = new Field(mesh, r2); r2curr = new Field(mesh);  r2new = new Field(mesh);  r2nonl = new Field(mesh);
    S = new Field(mesh);

    p11 = new Field(mesh); p12 = new Field(mesh); p21 = new Field(mesh);
    Ra = new Field(mesh);

    aux = new Field(mesh); aux1 = new Field(mesh);

    cudaMalloc((void**)&IFw, mesh->wavesize);
    cudaMalloc((void**)&IFwh, mesh->wavesize);

    cudaMalloc((void**)&IFr1, mesh->wavesize);
    cudaMalloc((void**)&IFr1h, mesh->wavesize);

    cudaMalloc((void**)&IFr2, mesh->wavesize);
    cudaMalloc((void**)&IFr2h, mesh->wavesize);

    // cout << "startpoint = " << startpoint << "\n";
    cout << "Re = " << parameters.Re << "  "; cout << "Er = " << parameters.Er << "   "; 
    cout << "Ra = " << parameters.alpha << "  "; cout << "Rf = " << parameters.Rf << "   "; 
    cout << "Lx = " << mesh->Lx << " "<< "Ly = " << mesh->Ly << " " << "   ";
    cout << "Nx = " << mesh->Nx << " "<< "Ny = " << mesh->Ny << " " << endl;
    cout << "dx = " << mesh->dx << " "<< "dy = " << mesh->dy << " " << " ";
    cout << "dt = " << parameters.dt << endl;
    cout << "Nx*dx = " << mesh->Nx*mesh->dx << " "<< "Ny*dy = " << mesh->Ny*mesh->dy << " " << endl;
    cout << "End time: Ns*dt = " << (0+parameters.Ns)*parameters.dt << endl;

    coord(mesh->dx, mesh->dy, mesh->Nx, mesh->Ny);
        // winit(phys_h, Nx, Ny, dx, dy);
    file_init(string("./init/") +"w_init.csv", phys_h, mesh);
    cudaMemcpy(wold->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(wold->spec, wold->phys, mesh);
    BwdTrans(wold->phys, wold->spec, mesh);
    cudaMemcpy(phys_h, wold->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "w.csv", Nx, Ny);

    // r1init(phys_h, Nx, Ny, dx, dy);
    file_init(string("./init/")+"r1_init.csv", phys_h, mesh);
    cudaMemcpy(r1old->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(r1old->spec, r1old->phys, mesh);
    BwdTrans(r1old->phys, r1old->spec, mesh);
    cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "r1.csv", Nx, Ny);

    // r2init(phys_h, Nx, Ny, dx, dy);
    file_init(string("./init/")+"r2_init.csv", phys_h, mesh);
    cudaMemcpy(r2old->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(r2old->spec, r2old->phys, mesh);
    BwdTrans(r2old->phys, r2old->spec, mesh);
    cudaMemcpy(phys_h, r2old->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "r2.csv", Nx, Ny);

    S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1old->phys, r2old->phys, Nx, Ny, BSZ);
    cudaMemcpy(phys_h, S->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "S.csv", Nx, Ny);

    Rainit(parameters.alpha, phys_h, Nx, Ny, mesh->dx, mesh->dy);
    cudaMemcpy(Ra->phys, phys_h, physize, cudaMemcpyHostToDevice);
    cudaMemcpy(phys_h, Ra->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "Ra.csv", Nx, Ny);

    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFw, IFwh, mesh->k_squared, parameters.Re, parameters.Rf, parameters.dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFr1, IFr1h, mesh->k_squared, parameters.dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFr2, IFr2h, mesh->k_squared, parameters.dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
}

void Qsolver::step(){
    Qreal Re = parameters.Re;
    Qreal Er = parameters.Er;
    Qreal dt = parameters.dt;
    Qreal lambda = parameters.lambda;
    int Nx = mesh->Nx;
    int Ny = mesh->Ny;
    int Nxh = mesh->Nxh;

    integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, IFw, IFwh, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, IFr1, IFr1h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, IFr2, IFr2h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);

    curr_func(r1curr, r2curr, wcurr, u, v, S);
    wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
    r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    // cuda_error_func( cudaDeviceSynchronize() );

    curr_func(r1curr, r2curr, wcurr, u, v, S);
    wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
    r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    // cuda_error_func( cudaDeviceSynchronize() );
    
    curr_func(r1curr, r2curr, wcurr, u, v, S);
    wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
    r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    // cuda_error_func( cudaDeviceSynchronize() );

    curr_func(r1curr, r2curr, wcurr, u, v, S);
    wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
    r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
    integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, mesh->Nxh, mesh->Ny, mesh->BSZ, parameters.dt);
    // cuda_error_func( cudaDeviceSynchronize() );

    dealias<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wnew->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
    dealias<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1new->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);
    dealias<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2new->spec, mesh->cutoff, mesh->Nxh, mesh->Ny, mesh->BSZ);

    SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
    SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

    BwdTrans(wold->phys, wold->spec, mesh);
    BwdTrans(r1old->phys, r1old->spec, mesh);
    BwdTrans(r2old->phys, r2old->spec, mesh);
    S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1old->phys, r2old->phys, Nx, Ny, BSZ);

    // cudaDeviceSynchronize();
    // if(m%10 == 0) {cout << "\r" << "t = " << (m+startpoint)*dt << flush;};
    // bool cpflag = true;
    // int specsize = mesh->Nxh*mesh->Ny*sizeof(Qcomp);
    // int physize = mesh->Nx*mesh->Ny*sizeof(Qreal);
    // int wavesize = mesh->Nxh*mesh->Ny*sizeof(Qreal);
    
    // if (cpflag){
    //     // BwdTrans(wold->phys, wold->spec, mesh);
    //     BwdTrans(wold->phys, wold->spec, mesh);
    //     cudaMemcpy(phys_h, wold->phys, physize, cudaMemcpyDeviceToHost);
    //     field_visual(phys_h, to_string(114)+"w.csv", Nx, Ny);

    //     BwdTrans(r1old->phys, r1old->spec, mesh);
    //     cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
    //     field_visual(phys_h, to_string(114)+"r1.csv", Nx, Ny);

    //     BwdTrans(r2old->phys, r2old->spec, mesh);
    //     cudaMemcpy(phys_h, r2old->phys, physize, cudaMemcpyDeviceToHost);
    //     field_visual(phys_h, to_string(114)+"r2.csv", Nx, Ny);

    //     S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1old->phys, r2old->phys, Nx, Ny, BSZ);
    //     cudaMemcpy(phys_h, S->phys, physize, cudaMemcpyDeviceToHost);
    //     field_visual(phys_h, to_string(114)+"S.csv", Nx, Ny);

    //     // if(std::isnan(phys_h[0])) {cout << "NAN" << "\n"; exit(0);}
    // }
}
void timestep(Qreal* w, Qreal* r1, Qreal* r2, int Nsteps, int cpN, bool initflag, bool cpflag){
    const Qreal lambda = 0.1;
    const Qreal Rf = 0.00075;
    const Qreal alpha = 0.2;
    const Qreal Re = 0.1;
    const Qreal Er = 0.1;
    const Qreal dt = 0.002;
    int Ns = Nsteps;
    int startpoint = 0;

    int Nx = 512*3/2;
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
    
    static Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    static Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    static Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    static Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);

    static Field *wold = new Field(mesh,w); static Field *wcurr = new Field(mesh); static Field *wnew = new Field(mesh); static Field *wnonl = new Field(mesh);
    static Field *phi = new Field(mesh);
    static Field *u = new Field(mesh);
    static Field *v = new Field(mesh);

    static Field *r1old = new Field(mesh, r1); static Field *r1curr = new Field(mesh); static Field *r1new = new Field(mesh); static Field *r1nonl = new Field(mesh);
    static Field *r2old = new Field(mesh, r2); static Field *r2curr = new Field(mesh); static Field *r2new = new Field(mesh); static Field *r2nonl = new Field(mesh);
    static Field *S = new Field(mesh);

    static Field* p11 = new Field(mesh); static Field *p12 = new Field(mesh); static Field *p21 = new Field(mesh);
    static Field *Ra = new Field(mesh);

    static Field *aux = new Field(mesh); static Field *aux1 = new Field(mesh);

    static Qreal *IFw;
    static Qreal *IFwh;
    static Qreal *IFr1;
    static Qreal *IFr1h;
    static Qreal *IFr2;
    static Qreal *IFr2h;

    if(initflag == true){
        cudaMalloc((void**)&IFw, mesh->wavesize);
        cudaMalloc((void**)&IFwh, mesh->wavesize);

        cudaMalloc((void**)&IFr1, mesh->wavesize);
        cudaMalloc((void**)&IFr1h, mesh->wavesize);

        cudaMalloc((void**)&IFr2, mesh->wavesize);
        cudaMalloc((void**)&IFr2h, mesh->wavesize);
        
        cout << "startpoint = " << startpoint << "\n";
        cout << "Re = " << Re << "  "; cout << "Er = " << Er << "   "; 
        cout << "Ra = " << alpha << "  "; cout << "Rf = " << Rf << "   "; 
        cout << "Lx = " << mesh->Lx << " "<< "Ly = " << mesh->Ly << " " << "   ";
        cout << "Nx = " << mesh->Nx << " "<< "Ny = " << mesh->Ny << " " << endl;
        cout << "dx = " << mesh->dx << " "<< "dy = " << mesh->dy << " " << " ";
        cout << "dt = " << dt << endl;
        cout << "Nx*dx = " << mesh->Nx*mesh->dx << " "<< "Ny*dy = " << mesh->Ny*mesh->dy << " " << endl;
        cout << "End time: Ns*dt = " << (startpoint+Ns)*dt << endl;
        coord(dx, dy, Nx, Ny);
            // winit(phys_h, Nx, Ny, dx, dy);
        file_init(string("./init/") +"w_init.csv", phys_h, mesh);
        cudaMemcpy(wold->phys, phys_h, physize, cudaMemcpyHostToDevice);
        FwdTrans(wold->spec, wold->phys, mesh);
        BwdTrans(wold->phys, wold->spec, mesh);
        cudaMemcpy(phys_h, wold->phys, physize, cudaMemcpyDeviceToHost);
        field_visual(phys_h, "w.csv", Nx, Ny);

        // r1init(phys_h, Nx, Ny, dx, dy);
        file_init(string("./init/")+"r1_init.csv", phys_h, mesh);
        cudaMemcpy(r1old->phys, phys_h, physize, cudaMemcpyHostToDevice);
        FwdTrans(r1old->spec, r1old->phys, mesh);
        BwdTrans(r1old->phys, r1old->spec, mesh);
        cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
        field_visual(phys_h, "r1.csv", Nx, Ny);

        // r2init(phys_h, Nx, Ny, dx, dy);
        file_init(string("./init/")+"r2_init.csv", phys_h, mesh);
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
    }
    

    wlin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFw, IFwh, mesh->k_squared, Re, Rf, dt, Nxh, Ny, BSZ);
    r1lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFr1, IFr1h, mesh->k_squared, dt, Nxh, Ny, BSZ);
    r2lin_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(IFr2, IFr2h, mesh->k_squared, dt, Nxh, Ny, BSZ);

    for (int m=0; m<Ns; m++){
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);

        curr_func(r1curr, r2curr, wcurr, u, v, S);
        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        // cuda_error_func( cudaDeviceSynchronize() );

        curr_func(r1curr, r2curr, wcurr, u, v, S);
        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        // cuda_error_func( cudaDeviceSynchronize() );
        
        curr_func(r1curr, r2curr, wcurr, u, v, S);
        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        // cuda_error_func( cudaDeviceSynchronize() );

        curr_func(r1curr, r2curr, wcurr, u, v, S);
        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        // cuda_error_func( cudaDeviceSynchronize() );

        dealias<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wnew->spec, mesh->cutoff, Nxh, Ny, BSZ);
        dealias<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1new->spec, mesh->cutoff, Nxh, Ny, BSZ);
        dealias<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2new->spec, mesh->cutoff, Nxh, Ny, BSZ);

        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, Nxh, Ny, BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, Nxh, Ny, BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, Nxh, Ny, BSZ);

        if(m%10 == 0) {cout << "\r" << "t = " << (m+startpoint)*dt << flush;};
        if (cpflag){
            // BwdTrans(wold->phys, wold->spec, mesh);
            BwdTrans(wold->phys, wold->spec, mesh);
            cudaMemcpy(phys_h, wold->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m+startpoint)+"w.csv", Nx, Ny);

            BwdTrans(r1old->phys, r1old->spec, mesh);
            cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m+startpoint)+"r1.csv", Nx, Ny);

            BwdTrans(r2old->phys, r2old->spec, mesh);
            cudaMemcpy(phys_h, r2old->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m+startpoint)+"r2.csv", Nx, Ny);

            S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1old->phys, r2old->phys, Nx, Ny, BSZ);
            cudaMemcpy(phys_h, S->phys, physize, cudaMemcpyDeviceToHost);
            field_visual(phys_h, to_string(m+startpoint)+"S.csv", Nx, Ny);

            if(std::isnan(phys_h[0])) {cout << "NAN" << "\n"; exit(0);}
        }
    }

    // delete mesh;
    // return 0;
}