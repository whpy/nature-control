# Qsolver application samples

## Intro
In this repository, we preliminary reproduce the computation of *"Role of Advectiv Inertia in Active Nematic Turbulence"*(Colin-Marius Koch and Michael Wilczek
Phys. Rev. Lett. 127, 268005). The whole code is constructed under the header-only principle.

## Configuration
The code has been tested on the configurations below:

10.X:
|dependency | version|
|---|---|
|Ubuntu| 20.04.1 |
|Nvidia Driver| 525.147.05 |
|nvcc| V10.1.243 |
|gcc| 9.4.0 |

11.X:
|dependency | version|
|---|---|
|Linux| 4.18.0-305.25.1.el8_4.x86_64 |
|Nvidia Driver| 535.104.05 |
|nvcc| V11.6.124 |
|gcc| 8.4.1 |

12.X:
|dependency | version|
|---|---|
|Linux| 4.18.0-305.25.1.el8_4.x86_64 |
|Nvidia Driver| 535.104.05 |
|nvcc| V12.0.140 |
|gcc| 8.4.1 |

also
|dependency | version|
|---|---|
|Ubuntu(wsl2)|  22.04.1 LTS|
|Nvidia Driver| 537.13 |
|nvcc| V12.3.103 |
|gcc| 11.3.0 |

## Usage
There is some existing `makefile` files in different existing directories. The arguments probably needed to be modified is the `$(CUDA_HOME)` and the `$(NVCC)` and the path to CUDA specific `include/` which contains the CUDA headers. It is not so easy to understand maybe so it would be modified to be more direct later. As it is header-only, so it is not so hard that compile the source code directly without makefile(that is how we do on public shared device). Please contact me(h_wu@u.nus.edu) if you come across problems.

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

## To be updated
0. Introduce the python interface which faciliates RL;
1. Better post-, pre-process modules, introduce convenient file format(to restart);
2. we copy the data of physical and spectral data for twice while forwards and backwards operation; 
3. We allocate one thread for each point which might not be the best strategy;
4. It is possible to allocate the wavenumber on the share memory to accelerate the speed;
5. Using the official linear algorithm libs(cublas);


