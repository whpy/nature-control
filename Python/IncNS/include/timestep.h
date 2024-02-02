#include <Basic/QActFlowDef.cuh>
#include <torch/extension.h>

struct Field;
struct Mesh;

void timestep(Qreal* w, Qreal* r1, Qreal* r2, int Nsteps, int cpN, bool initflag, bool fileflag);

typedef struct param{
    Qreal lambda; Qreal Rf; Qreal alpha; Qreal Re; Qreal Er;
    Qreal Lx; Qreal Ly; Qreal dt; int Ns; int Nx; int Ny;

    param();
    param(Qreal plambda, Qreal pRf, Qreal palpha, Qreal pRe, 
    Qreal pEr, Qreal pLx, Qreal pLy, Qreal pdt, int pNs, int pNx, int pNy);
}param;

typedef struct Incparam{
    Qreal Re; Qreal Lx; Qreal Ly; Qreal dt; int Nx; int Ny;
    Incparam();
    Incparam(Qreal pRe, Qreal pLx, Qreal pLy, Qreal pdt, int pNx, int pNy);
}Incparam;

class IncSolver{
    public:
    // store the geometry parameters
    Incparam parameters; Mesh *mesh; int BSZ;

    // allocate memory on CPU
    Qreal *phys_h; Qcomp *spec_h; Qreal *wave_h;

    // fields of fluid
    Field *wold; Field *wcurr; Field *wnew; Field *wnonl;
    Field *phi; Field *u; Field *v;

    // auxiliary fields
    Field *aux ;  Field *aux1 ;

    // time step linear factors for independent variables
    Qreal *IFw; Qreal *IFwh;

    IncSolver();
    void init(Incparam pParam, int pBSZ, torch::Tensor &wt); // initialization 
    void step(); // time step
    bool initstate(); // return the state of initialization

    void VelConvert(Qreal* w, Qreal* u, Qreal* v); // Convert the received velocity field into vorticity field
    void VortConvert(Qreal* u, Qreal* v, Qreal* w); // Convert the vorticity field into velocity field

    void log();

    private:
    bool initflag;
};

class Qsolver
{
    public:
    // store the geometry parameters
    param parameters; Mesh *mesh; int BSZ;

    // allocate memory on CPU
    Qreal *phys_h; Qcomp *spec_h; Qreal *wave_h;

    // fields of fluid
    Field *wold; Field *wcurr; Field *wnew; Field *wnonl;
    Field *phi; Field *u; Field *v;
    // fields of director field
    Field *r1old ; Field *r1curr ;  Field *r1new ;  Field *r1nonl ;
    Field *r2old ;  Field *r2curr ;  Field *r2new ;  Field *r2nonl ;
    Field *S ;

    // auxiliary fields
    Field* p11 ;  Field *p12 ;  Field *p21 ;
    Field *Ra ;

    Field *aux ;  Field *aux1 ;

    // time step linear factors for independent variables
    Qreal *IFw; Qreal *IFwh;
    Qreal *IFr1; Qreal *IFr1h;
    Qreal *IFr2; Qreal *IFr2h;

    Qsolver();
    void init(param pParam, int pBSZ, Qreal* w, Qreal* r1, Qreal* r2);
    void step();
    bool initstate();
    void log();

    private:
    bool initflag;
};