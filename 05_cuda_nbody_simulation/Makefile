CC = gcc
NVCC = nvcc

CFLAGS = -O3 -lm
OMP_FLAGS = -fopenmp

GPU_FLAGS = -O3 -arch=sm_37 -use_fast_math -ftz=true -Xcompiler -fopenmp

all: nbody_seq nbody_omp nbody_gpu

nbody_seq: nbody.c
	$(CC) $(CFLAGS) nbody.c -o nbody_seq

nbody_omp: nbody_omp.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) nbody_omp.c -o nbody_omp

nbody_gpu: nbody_gpu.cu
	$(NVCC) $(GPU_FLAGS) nbody_gpu.cu -o nbody_gpu

clean:
	rm -f nbody_seq nbody_omp nbody_gpu results.txt

.PHONY: all clean