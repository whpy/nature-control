#include "Qsolver.cuh"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <TimeIntegration/RK4.cuh>

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
            r1[index] = 0.5*(Qreal(rand())/RAND_MAX)-0.25; 
        }
    }
}

void r2init(Qreal *r2, int Nx, int Ny, Qreal dx, Qreal dy){
    for (int i=0; i<Nx; i++){
        for (int j=0; j<Ny; j++){
            Qreal x = i*dx;
            Qreal y = j*dy;
            int index = i + j*Nx;
            r2[index] = 0.5*(Qreal(rand())/RAND_MAX)-0.25; 
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

int main(){
    Qreal lambda = 0.1;
    Qreal Rf = 0.000075;
    Qreal alpha = 0.2;
    Qreal Re = 0.1;
    Qreal Er = 0.1;
    Qreal dt = 0.002;
    int startpoint = 25100;
    int Ns = 80001;

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

    cout << "startpoint = " << startpoint << "\n";
    cout << "Re = " << Re << "  "; cout << "Er = " << Er << "   "; 
    cout << "Lx = " << mesh->Lx << " "<< "Ly = " << mesh->Ly << " " << "   ";
    cout << "Nx = " << mesh->Nx << " "<< "Ny = " << mesh->Ny << " " << endl;
    cout << "dx = " << mesh->dx << " "<< "dy = " << mesh->dy << " " << " ";
    cout << "dt = " << dt << endl;
    cout << "Nx*dx = " << mesh->Nx*mesh->dx << " "<< "Ny*dy = " << mesh->Ny*mesh->dy << " " << endl;
    cout << "End time: Ns*dt = " << (startpoint+Ns)*dt << endl;

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
    // winit(phys_h, Nx, Ny, dx, dy);
    file_init("./init/"+ to_string(startpoint) +"w.csv", phys_h, mesh);
    cudaMemcpy(wold->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(wold->spec, wold->phys, mesh);
    BwdTrans(wold->phys, wold->spec, mesh);
    cudaMemcpy(phys_h, wold->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "w.csv", Nx, Ny);

    // r1init(phys_h, Nx, Ny, dx, dy);
    file_init("./init/"+ to_string(startpoint) +"r1.csv", phys_h, mesh);
    cudaMemcpy(r1old->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(r1old->spec, r1old->phys, mesh);
    BwdTrans(r1old->phys, r1old->spec, mesh);
    cudaMemcpy(phys_h, r1old->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "r1.csv", Nx, Ny);

    // r2init(phys_h, Nx, Ny, dx, dy);
    file_init("./init/"+ to_string(startpoint) +"r2.csv", phys_h, mesh);
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
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func0D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);

        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func1D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func2D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );
        
        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func3D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        wnonl_func(wnonl, wcurr, u, v, r1curr, r2curr, Ra, S, p11, p12, p21, Re, Er, lambda, aux, aux1);
        r1nonl_func(r1nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        r2nonl_func(r2nonl, wcurr, u, v, r1curr, r2curr, S, lambda, aux, aux1);
        integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wcurr->spec, wnew->spec, wnonl->spec, IFw, IFwh, Nxh, Ny, BSZ, dt);
        integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1curr->spec, r1new->spec, r1nonl->spec, IFr1, IFr1h, Nxh, Ny, BSZ, dt);
        integrate_func4D<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2curr->spec, r2new->spec, r2nonl->spec, IFr2, IFr2h, Nxh, Ny, BSZ, dt);
        cuda_error_func( cudaDeviceSynchronize() );

        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(wold->spec, wnew->spec, Nxh, Ny, BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1old->spec, r1new->spec, Nxh, Ny, BSZ);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2old->spec, r2new->spec, Nxh, Ny, BSZ);

        if(m%10 == 0) {cout << "\r" << "t = " << (m+startpoint)*dt << flush;};
        if (m%100 == 0){
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
        }
    }

    delete mesh;
    return 0;
}