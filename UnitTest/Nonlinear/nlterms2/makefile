# flags and paths
ROOT = ../..
INC = -I $(ROOT) -I /usr/local/cuda/include/
SRC = $(ROOT)
LIBRARIES := -lcufft

HOST_COMPILER := gcc
NVCC          := $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)

GENCODE_FLAGS := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
ALL_CCFLAGS := -m64    --threads 0 --std=c++11


# build target rules

Qsolver.o:Qsolver.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
Qsolver.out:Qsolver.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nlterms0.o:nlterms0.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
nlterms0.out:nlterms0.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nlterms1.o:nlterms1.cu
	$(NVCC)  $(INC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
nlterms1.out:nlterms1.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)


# clean
.PHONY: clean
clean:
	rm *.o main *.csv *.out
