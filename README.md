# Qsolver application samples

## Intro
In this repository, we preliminary reproduce the computation of *"Role of Advectiv Inertia in Active Nematic Turbulence"*(PhysRevLett.127.268.005). The whole code is constructed under the header-only principle.

## Configuration
The code has been tested on the configurations below:

|dependency | version|
|---|---|
|Ubuntu| 20.04.1 |
|Nvidia Driver| 525.147.05 |
|nvcc| V10.1.243 |
|gcc| 9.4.0 |

## Structure
### Basic
`cuComplexBinOp.cuh`: defines the operations in $C$

`QActFlowDef.cuh`: The Real and comp control the precision in the programe. Remember that it concerns the FFT operation definition in `FldOp/QFldfuncs.cuh` while changing.

`Mesh.cuh`: Defines the struct `Mesh` which faciliates the spatial parameter delivery while coding.

### BasicUtils
`BasicUtils`: It provides some utils like field print and generating .csv files of physical field.

### UnitTest
Stores the test samles to verify the validation of basic field operations and the correctness of nonlinear part(refered to the latex).

### InertiaFriction
The main laboratory to compute the simulation. Noted that `post_process/` stores some post process tools to calculate the total energy and the energy spectrum.

## Notes
There still exists large space to optimize the code. For example, we copy the data of physical and spectral data for twice while forwards and backwards operation; We allocate one thread for each point which might not be the best strategy; It is possible to allocate the wavenumber on the share memory to accelerate the speed.


