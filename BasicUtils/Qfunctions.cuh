#ifndef QFUNCTIONS_CUH
#define QFUNCTIONS_CUH

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <QActFlowDef.cuh>


using std::string;
using std::endl;
using std::ofstream;
using std::cout;

inline void print_spec(Qcomp* f, int Nxh, int Ny){
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = i + j*Nxh;
            printf("(%.2f, %.2f)  ", f[index].x, f[index].y);
        }
        cout << endl;
    }
    cout << endl;
}

inline void print_spec(Qreal* f, int Nxh, int Ny){
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = i + j*Nxh;
            printf("%.2f  ", f[index]);
        }
        cout << endl;
    }
    cout << endl;
}

inline void field_visual(Qreal *f, string name, int Nx, int Ny){
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            int index = j*Nx + i;
            fval << f[index] << ",";
        }
        fval << endl;
    }
    fval.close();
}

inline void coord(Qreal dx, Qreal dy, int Nx, int Ny){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<Ny; j++){
        for ( int i=0; i< Nx; i++){
            float x = dx*i;
            float y = dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}

// // to denote the type of initialization
// enum eInitType{
//     Phy_init,
//     File_init
// };

// string InitType[] = {
//     "Physical init",
//     "File init"
// };
//  // preclaimation
// void r1p_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);

// void r2p_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);

// void wp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);

// void r1sp_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny);

// void r2sp_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny);

// void wsp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny);

// void precompute_func(Field* r1, Field* r2, Field* w, eInitType flag);

// // definitions
// void file_init(string filename, Field* f){
//     int Nx = f->mesh->Nx;
//     int Ny = f->mesh->Ny;

//     ifstream infile(filename);
//     vector<vector<string>> data;
//     string strline;
//     while(getline(infile, strline)){
//         stringstream ss(strline);
//         string str;
//         vector<string> dataline;

//         while(getline(ss, str, ',')){
//             dataline.push_back(str);
//         }
//         data.push_back(dataline);
//     }

//     if(data.size() != Ny || data[0].size() != Nx){
//         printf("READ_CSV_ERROR: size not match! \n");
//         return;
//     }
//     int index = 0;
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             index = j*Nx + i;
//             f->phys[index] = string2num<double>(data[j][i]);
//         }
//     }
//     infile.close();
// }

// void r1p_init(Qreal *r1, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             float x = dx*i;
//             float y = dy*j;
//             // r1[index] = (double(rand())/RAND_MAX-1)*0.5;
//             r1[index] = 0.5*(sin((x+y))*sin((x+y)) - 0.5);
//         }
//     }
// }

// void r2p_init(Qreal *r2, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             float x = dx*i;
//             float y = dy*j;
//             // r2[index] = (double(rand())/RAND_MAX-1)*0.5;
//             r2[index] = 0.2*sin((x+y))*cos((x+y));
//         }
//     }
// }

// void wp_init(Qreal *w, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             float x = dx*i;
//             float y = dy*j;
//             // w[index] = -2*cos(x)*cos(y);
//             w[index] = 0.0;
//         }
//     }
// }

// void alpha_init(Qreal *alpha, Qreal Ra, Qreal dx, Qreal dy, int Nx, int Ny){
//     for (int j=0; j<Ny; j++){
//         for (int i=0; i<Nx; i++){
//             int index = i+j*Nx;
//             alpha[index] = (Ra);
//         }
//     }
// }

// void precompute_func(Field* r1, Field* r2, Field* w, eInitType flag){
//     Mesh* mesh = r1->mesh;
//     int Nx = mesh->Nx; int Ny = mesh->Ny;
//     Qreal dx = mesh->dx; Qreal dy = mesh->dy;

//     if (flag == Phy_init){
//         r1p_init(r1->phys, dx, dy, Nx, Ny);
//         r2p_init(r2->phys, dx, dy, Nx, Ny);
//         wp_init(w->phys, dx, dy, Nx, Ny);
//         FwdTrans(mesh, r1->phys, r1->spec);
//         FwdTrans(mesh, r2->phys, r2->spec);
//         FwdTrans(mesh, w->phys, w->spec);
//     }
//     else if (flag == File_init){
//         file_init("./init/r1_init.csv", r1);
//         file_init("./init/r2_init.csv", r2);
//         file_init("./init/w_init.csv", w);
//         FwdTrans(r1->spec, r1->phys,mesh);
//         FwdTrans(r2->spec, r2->phys, mesh);
//         FwdTrans(w->spec, w->phys, mesh);
//     }
//     else{
//         cout << "no such type initialization!" << "\n";
//     }
    
// }
#endif