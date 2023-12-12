#ifndef __QSOLVER_CUH
#define __QSOLVER_CUH

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <cuComplexBinOp.cuh>
#include <cudaErr.h>
#include <QActFlowDef.cuh>

#include <Mesh.cuh>
#include <Qfunctions.cuh>
#include <QFldfuncs.cuh>

//////////////////////////////////////// linear ////////////////////////////////////////////////////
__global__
void wlin_func(Qreal *IFw, Qreal *IFwh, Qreal *k_squared, Qreal Re, Qreal Rf, Qreal dt, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nxh && j<Ny){
        int index = i + j*Nxh;
        Qreal alpha = 1.0/Re * (-1.0*k_squared[index] - Rf);
        IFw[index] = exp(alpha *dt);
        IFwh[index] = exp(alpha *dt*0.5);
    }
}

__global__
void r1lin_func(Qreal *IFr1, Qreal *IFr1h, Qreal *k_squared, Qreal dt, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nxh && j<Ny){
        int index = i + j*Nxh;
        Qreal alpha1 = (-1.0*k_squared[index] + 1.0);
        IFr1[index] = exp(alpha1*dt);
        IFr1h[index] = exp(alpha1 *dt*0.5);
    }
}

__global__
void r2lin_func(Qreal *IFr2, Qreal *IFr2h, Qreal *k_squared, Qreal dt, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nxh && j<Ny){
        int index = i + j*Nxh;
        Qreal alpha2 = (-1.0*k_squared[index] + 1.0);
        IFr2[index] = exp(alpha2*dt);
        IFr2h[index] = exp(alpha2 *dt*0.5);
    }
}







//////////////////////////////////////// nonlinear /////////////////////////////////////////////////
__global__
void S_func(Qreal *S, Qreal *r1, Qreal *r2, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nx && j<Ny){
        int index = i + j*Nx;
        S[index] = 2*sqrt(r1[index]*r1[index] + r2[index]*r2[index]);
    }
}

//////// vorticity nonlinear terms //////////
// x = x - Ra*ri
__global__
void Rar(Qreal *rar, const Qreal *r, const Qreal *Ra, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nx && j<Ny){
        int index = i + j*Nx;
        rar[index] = rar[index] - Ra[index]*r[index];
    }
}

//  x= x + \lambda S(S^2-1)ri
__global__
void single1(Qreal *s1, const Qreal *r, const Qreal *S, Qreal lambda, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nx && j<Ny){
        int index = i + j*Nx;
        s1[index] = s1[index] + lambda* S[index]* (S[index]*S[index]-1.0)*r[index];
    }
}
// -\lambda S\Delta(ri) + \lambda S(S^2-1)ri
inline 
void single(Field *s0, Field *r, Field *S, Qreal lambda){
    Mesh* mesh = s0->mesh;
    Laplacian(s0->spec, r->spec, mesh);
    BwdTrans(s0->phys, s0->spec, mesh);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(s0->phys, s0->phys, S->phys, -1.0*lambda, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    single1<<<mesh->dimGridp, mesh->dimBlockp>>>(s0->phys, r->phys, S->phys, lambda, 
    mesh->Nx,mesh->Ny, mesh->BSZ);
}

inline 
void cross(Field *cs, Field *r1, Field *r2, Field *aux){
    Mesh *mesh = cs->mesh;
    Laplacian(cs->spec, r1->spec, mesh);
    BwdTrans(cs->phys, cs->spec, mesh);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(cs->phys, cs->phys, r2->phys, 2.0,
    mesh->Nx, mesh->Ny, mesh->BSZ);

    Laplacian(aux->spec, r2->spec, mesh);
    BwdTrans(aux->phys, aux->spec, mesh);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(aux->phys, r1->phys, aux->phys, -2.0, 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(cs->phys, 1.0, cs->phys, 1.0, aux->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
}

inline 
void p11func(Field *p11, Field *r1, Field *S, Field *Ra, Qreal lambda){
    Mesh *mesh = p11->mesh;
    single(p11, r1, S, lambda);
    Rar<<<mesh->dimGridp,mesh->dimBlockp>>>(p11->phys, r1->phys, Ra->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(p11->phys, 1.0, p11->phys, -1.0, Ra->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(p11->spec, p11->phys, p11->mesh);
}

inline
void p12func(Field *p12, Field *r1, Field *r2, Field *S, Field *Ra, Qreal lambda, Field *aux, Field *aux1){
    Mesh *mesh = p12->mesh;
    single(p12, r2, S, lambda);
    cross(aux, r1, r2, aux1);
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(p12->phys, 1.0, p12->phys, 1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    Rar<<<mesh->dimGridp,mesh->dimBlockp>>>(p12->phys, r2->phys, Ra->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(p12->spec, p12->phys, p12->mesh);
}

inline
void p21func(Field *p21, Field *r1, Field *r2, Field *S, Field *Ra, Qreal lambda, Field *aux, Field *aux1){
    Mesh *mesh = p21->mesh;
    single(p21, r2, S, lambda);
    cross(aux, r2, r1, aux1);
    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(p21->phys, 1.0, p21->phys, 1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    Rar<<<mesh->dimGridp,mesh->dimBlockp>>>(p21->phys, r2->phys, Ra->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(p21->spec, p21->phys, p21->mesh);
}


__global__
void PiDeriv(Qcomp* dP, Qcomp* p12, Qcomp* p11, Qcomp* p21, Qreal *kx, Qreal *ky, Qreal Er, Qreal Re, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nxh && j<Ny){
        int index = j*Nxh + i;
        dP[index] = dP[index] + 1.0/(Er*Re)*(-1.0*kx[index]*kx[index]*p12[index] 
        + 2.0*kx[index]*ky[index]*p11[index] 
        + ky[index]*ky[index]*p21[index]);
    }
}

inline 
void wnonl_func(Field* wnonl, Field *wcurr, Field *u, Field *v, Field *r1curr, Field *r2curr, Field *Ra, Field *S, 
Field *p11, Field *p12, Field *p21, Qreal Re, Qreal Er, Qreal lambda, Field *aux, Field* aux1){
    Mesh *mesh = wnonl->mesh;
    BwdTrans(r1curr->phys, r1curr->spec,mesh);
    BwdTrans(r2curr->phys, r2curr->spec,mesh);
    BwdTrans(wcurr->phys, wcurr->spec,mesh);
    S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1curr->phys, r2curr->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    vel_func(wcurr->spec, u->spec, v->spec, mesh);
    BwdTrans(u->phys, u->spec, mesh);
    BwdTrans(v->phys, v->spec, mesh);

    // convective term of vorticity : -D(uw,x) - D(vw,y)
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(wnonl->phys, u->phys, wcurr->phys, -1.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(wnonl->spec, wnonl->phys, mesh);
    xDeriv(wnonl->spec, wnonl->spec, mesh);
    BwdTrans(wnonl->phys, wnonl->spec, mesh);

    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(aux->phys, v->phys, wcurr->phys, -1.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux->spec, aux->phys, mesh);
    yDeriv(aux->spec, aux->spec, mesh);
    BwdTrans(aux->phys, aux->spec, mesh);

    FldAdd<<<mesh->dimGridp,mesh->dimBlockp>>>(wnonl->phys, 1.0, wnonl->phys, 1.0, aux->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(wnonl->spec, wnonl->phys, mesh);

    p11func(p11, r1curr, S, Ra, lambda);
    p12func(p12, r1curr,r2curr, S, Ra, lambda, aux, aux1);
    p21func(p21, r1curr, r2curr, S, Ra, lambda, aux, aux1);
    PiDeriv<<<mesh->dimGridsp,mesh->dimBlocksp>>>(wnonl->spec, p12->spec, p11->spec, p21->spec, mesh->kx, mesh->ky, Er, Re, mesh->Nxh, mesh->Ny, mesh->BSZ);

}



// Q nonlinear terms
// x = x - S^2*ri
__global__
void S2r(Qreal *rs, Qreal *r, Qreal *S, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    if (i<Nx && j<Ny){
        int index = i + j*Nx;
        rs[index] = rs[index] - S[index]*S[index]*r[index];
    }
}

inline 
void r1nonl_func(Field *r1nonl, Field *wcurr, Field *u, Field *v, Field *r1curr, Field *r2curr, Field *S, Qreal lambda, Field* aux, Field* aux1){
    Mesh *mesh = r1nonl->mesh;
    BwdTrans(r1curr->phys, r1curr->spec,mesh);
    BwdTrans(r2curr->phys, r2curr->spec,mesh);
    BwdTrans(wcurr->phys, wcurr->spec,mesh);
    S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1curr->phys, r2curr->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    vel_func(wcurr->spec, u->spec, v->spec, mesh);
    BwdTrans(u->phys, u->spec, mesh);
    BwdTrans(v->phys, v->spec, mesh);

    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(r1nonl->phys, wcurr->phys, r2curr->phys, -1.0,
     mesh->Nx, mesh->Ny, mesh->BSZ);
    S2r<<<mesh->dimGridp, mesh->dimBlockp>>>(r1nonl->phys, r1curr->phys, S->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(r1nonl->spec, r1nonl->phys, mesh);

    // convective term of r1 : -D(ur1,x) - D(vr1,y)
    // -D(ur1,x)
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(aux->phys, u->phys, r1curr->phys, -1.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux->spec, aux->phys, mesh);
    xDeriv(aux->spec, aux->spec, mesh);
    // BwdTrans(aux->phys, aux->spec, mesh);

    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(aux1->phys, v->phys, r1curr->phys, -1.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux1->spec, aux1->phys, mesh);
    // - D(vr1,y)
    yDeriv(aux1->spec, aux1->spec, mesh);
    // BwdTrans(aux1->phys, aux1->spec, mesh);
    // -D(ur1,x) - D(vr1,y)
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(aux->spec, 1.0, aux->spec, 1.0, aux1->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1nonl->spec, 1.0, r1nonl->spec, 1.0, aux->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

    xDeriv(aux->spec, u->spec, mesh);
    BwdTrans(aux->phys, aux->spec, mesh);
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(aux->phys, S->phys, aux->phys, lambda, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux->spec, aux->phys, mesh);
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r1nonl->spec, 1.0, r1nonl->spec, 1.0, aux->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
}

inline 
void r2nonl_func(Field *r2nonl, Field *wcurr, Field *u, Field *v, Field *r1curr, Field *r2curr, Field *S, Qreal lambda, Field* aux, Field* aux1){
    Mesh *mesh = r2nonl->mesh;
    BwdTrans(r1curr->phys, r1curr->spec,mesh);
    BwdTrans(r2curr->phys, r2curr->spec,mesh);
    BwdTrans(wcurr->phys, wcurr->spec,mesh);
    S_func<<<mesh->dimGridp, mesh->dimBlockp>>>(S->phys, r1curr->phys, r2curr->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    vel_func(wcurr->spec, u->spec, v->spec, mesh);
    BwdTrans(u->phys, u->spec, mesh);
    BwdTrans(v->phys, v->spec, mesh);

    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(r2nonl->phys, wcurr->phys, r1curr->phys, 1.0,
     mesh->Nx, mesh->Ny, mesh->BSZ);
    S2r<<<mesh->dimGridp, mesh->dimBlockp>>>(r2nonl->phys, r2curr->phys, S->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(r2nonl->spec, r2nonl->phys, mesh);

    // convective term of r2 : -D(u*r2,x) - D(v*r2,y)
    // -D(ur1,x)
    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(aux->phys, u->phys, r2curr->phys, -1.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux->spec, aux->phys, mesh);
    xDeriv(aux->spec, aux->spec, mesh);
    // BwdTrans(aux->phys, aux->spec, mesh);

    FldMul<<<mesh->dimGridp,mesh->dimBlockp>>>(aux1->phys, v->phys, r2curr->phys, -1.0, mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux1->spec, aux1->phys, mesh);
    // - D(vr1,y)
    yDeriv(aux1->spec, aux1->spec, mesh);
    // BwdTrans(aux1->phys, aux1->spec, mesh);
    // -D(ur1,x) - D(vr1,y)
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(aux->spec, 1.0, aux->spec, 1.0, aux1->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2nonl->spec, 1.0, r2nonl->spec, 1.0, aux->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);

    yDeriv(aux->spec, u->spec, mesh);
    xDeriv(aux1->spec, v->spec, mesh);
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(aux->spec, 1.0, aux->spec, 1.0, aux1->spec, 
    mesh->Nxh, mesh->Ny, mesh->BSZ);
    BwdTrans(aux->phys, aux->spec, mesh);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(aux->phys, S->phys, aux->phys, 0.5*lambda, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    FwdTrans(aux->spec, aux->phys, mesh);

    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(r2nonl->spec, 1.0, r2nonl->spec, 1.0, aux->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
}

#endif