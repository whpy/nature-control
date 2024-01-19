#include <Basic/QActFlowDef.cuh>

// void init(Field* wold, Field *r1old, Field *r2old);

void timestep(Qreal* w, Qreal* r1, Qreal* r2, int Nsteps, int cpN, bool initflag, bool fileflag);
