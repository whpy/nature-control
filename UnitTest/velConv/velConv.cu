#include <Basic/cuComplexBinOp.cuh>
#include <Basic/cudaErr.h>
#include <Basic/QActFlowDef.cuh>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <Basic/cuComplexBinOp.cuh>
#include <Basic/cudaErr.h>
#include <Basic/QActFlowDef.cuh>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <Basic/Mesh.cuh>
#include <BasicUtils/BasicUtils.cuh>
#include <FldOp/QFldfuncs.cuh>

// #define M_PI 3.141592653589

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

int main(int argc, char* argv[]){
    string prename = argv[1];

    int Nx = 512*3/2;
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

    Qreal *phys_h = (Qreal*)malloc(sizeof(Qreal)*Nx*Ny);
    Qcomp *spec_h = (Qcomp*)malloc(sizeof(Qcomp)*Nxh*Ny); 
    Qreal *wave_h = (Qreal*)malloc(sizeof(Qcomp)*Nxh*Ny); 

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly, BSZ);

    Field *w = new Field(mesh);

    Field *u = new Field(mesh);
    Field *v = new Field(mesh);

    file_init(prename, phys_h, mesh);
    cudaMemcpy(w->phys, phys_h, physize, cudaMemcpyHostToDevice);
    FwdTrans(w->spec, w->phys, mesh);
    BwdTrans(w->phys, w->spec, mesh);
    cudaMemcpy(phys_h, w->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "w.csv", Nx, Ny);

    vel_func(w->spec, u->spec, v->spec, mesh);
    BwdTrans(u->phys, u->spec, mesh);
    BwdTrans(v->phys, v->spec, mesh);

    cudaMemcpy(phys_h, u->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "u.csv", Nx, Ny);

    cudaMemcpy(phys_h, v->phys, physize, cudaMemcpyDeviceToHost);
    field_visual(phys_h, "v.csv", Nx, Ny);
}