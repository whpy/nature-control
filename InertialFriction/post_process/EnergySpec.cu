#include "Qsolver.cuh"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;
inline void wave_visual(Qreal *f, string name, int Nxh, int Ny){
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int index = j*Nxh + i;
            fval << f[index] << ",";
        }
        fval << endl;
    }
    fval.close();
}

inline void spec_visual(Qcomp *f, string name, int Nxh, int Ny){
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int index = j*Nxh + i;
            fval << "("<< f[index].x << "," << f[index].y <<")" << ",";
        }
        fval << endl;
    }
    fval.close();
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

Qreal Espec_func(Qreal k, Qcomp *ut, Qcomp *vt, Qreal *kx, Qreal *ky, Qreal alpha, int Nxh, int Ny){
    Qreal E = 0.0;

    int index = 0;
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            index = j*Nxh + i;
            Qreal mod = (kx[index]*kx[index] + ky[index]*ky[index]);
            if( ( mod < ((k+alpha)*(k+alpha)) ) && ( mod >= (k*k)) ){
                E += ut[index].x*ut[index].x + ut[index].y*ut[index].y + vt[index].x*vt[index].x + vt[index].y*vt[index].y;
            }
        }
    }

    return E/(2.0*alpha*(Nxh-1)*2*Ny*(Nxh-1)*2*Ny);
}


int main(int argc, char *argv[]){
    int starts = string2num<int>(string(argv[1]));
    Qreal lambda = 0.1;
    Qreal Rf = 0.0000;
    Qreal alpha = 0.2;
    Qreal Re = 0.1;
    Qreal Er = 0.1;
    Qreal dt = 0.002;
    int Ns = 800001;
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
    
    

    cufftHandle transf;
    cufftHandle inv_transf;
    cufft_error_func( cufftPlan2d( &(transf), Ny, Nx, CUFFT_D2Z ) );
    cufft_error_func( cufftPlan2d( &(inv_transf), Ny, Nx, CUFFT_Z2D ) );

    dim3 dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dim3 dimBlockp = dim3(BSZ, BSZ);

    dim3 dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dim3 dimBlocksp = dim3(BSZ, BSZ);

    
    Qreal *w_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);

    Qcomp *wt_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    Qcomp *ut_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    Qcomp *vt_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);

    Qreal *kx_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny);
    Qreal *ky_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny);
    cudaMemcpy(kx_h, mesh->kx,wavesize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ky_h, mesh->ky,wavesize, cudaMemcpyDeviceToHost);

    wave_visual(kx_h, "kx.txt", Nxh, Ny);
    wave_visual(ky_h, "ky.txt", Nxh, Ny);

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

    Field *w = new Field(mesh);
    Field *u = new Field(mesh);
    Field *v = new Field(mesh);

    Field *r1 = new Field(mesh);
    Field *r2 = new Field(mesh);

    Field *S = new Field(mesh);
    Qreal Etot;

    ofstream Eval;
    string fname = "EspecIF.csv";
    file_init("friction/"+to_string(starts)+"w.csv", phys_h, mesh);
    cudaMemcpy(w->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(w->spec, w->phys, mesh);
    cudaMemcpy(wt_h, w->spec,mesh->specsize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "w.csv", Nx, Ny);
    spec_visual(wt_h,"wt.txt",Nxh, Ny);
    
    vel_func(w->spec, u->spec, v->spec, mesh);
    cudaMemcpy(ut_h, u->spec,mesh->specsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(vt_h, v->spec,mesh->specsize, cudaMemcpyDeviceToHost);

    spec_visual(ut_h,"ut.txt",Nxh, Ny);
    spec_visual(vt_h,"vt.txt",Nxh, Ny);
    Eval.open(fname);
    // Eval << "dx = " << dx << " dy = " << dy << " dt = " << dt << " \\delta k = " <<  2*M_PI/Lx << endl;
    // Eval << "=======================================================" << endl;
    Qreal dk = 2*M_PI/Lx;

    Qreal kmax = 180*dk;

    Qreal k = 0.0;
    while (k < kmax){
        Qreal r = Espec_func(k, ut_h, vt_h, kx_h, ky_h, dk, Nxh, Ny);
        Eval << k << "," << r << "\n";
        cout << "\r" << kmax - k << ":  " << r << flush;
        k += dk;
    }
    cout << "\n";
    Eval.close();



    delete mesh;
    return 0;
}