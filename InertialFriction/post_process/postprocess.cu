#include "Qsolver.cuh"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

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

Qreal Etot_func(Qreal *u, Qreal *v, int Nx, int Ny, Qreal dx, Qreal dy){
    Qreal E = 0.0;
    int index = 0;
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            index = j*Nx + i;
            E += (u[index]*u[index] + v[index]*v[index])*0.5*dx*dy;
        }
    }

    return E;
}


int main(int argc, char *argv[]){
    string path = string(argv[1]);
    int starts = string2num<int>(string(argv[2]));
    int duration = string2num<int>(string(argv[3]));
    int ends = string2num<int>(string(argv[4]));
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

    Qreal *u_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qreal *v_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qreal *w_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);
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
    string fname = "Etot4.txt";
    Eval.open(fname);
    Eval << "dx = " << dx << " dy = " << dy << " dt = " << dt << endl;
    Eval << "=======================================================" << endl;
    for(int i=starts; i<=ends; i = i+duration) {
        Eval<< i << "   ";
        file_init(path + to_string(i) + "w.csv", phys_h, mesh);
        cudaMemcpy(w->phys, phys_h, physize, cudaMemcpyHostToDevice);
        FwdTrans(w->spec, w->phys, mesh);
        
        vel_func(w->spec, u->spec, v->spec, mesh);
        BwdTrans(u->phys, u->spec, mesh);
        BwdTrans(v->phys, v->spec, mesh);

        cudaMemcpy(u_h, u->phys, physize, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_h, v->phys, physize, cudaMemcpyDeviceToHost);

        field_visual(u_h,"u.csv",Nx, Ny);

        field_visual(v_h,"v.csv",Nx, Ny);

        // the so-called total energy in kolin, but actually it is divided by the area of the domain
        // refered to A.Alexakis and L.Biferale, Phys.Rep.767,1(2018)
        Etot = Etot_func(u_h, v_h, Nx, Ny,dx,dy);
        Etot = Etot/(Lx*Ly);
        Eval<< Etot << endl;
        cout << i << " Etot = " << Etot << endl;
    }
    
    Eval.close();



    delete mesh;
    return 0;
}