High Performance Computing (HPC Systems)

A comprehensive collection of high-performance computing projects developed for the HPC course at the University of Thessaly. This repository demonstrates progressive optimization strategies, scaling from sequential CPU code to multi-core OpenMP parallelization, and finally to massively parallel GPU acceleration using NVIDIA CUDA.

Tech Stack
Languages: C, C++
Parallel Models: OpenMP, NVIDIA CUDA
Hardware Architecture Concepts: Memory Coalescing, Shared Memory, PCIe Latency Hiding, Instruction-Level Parallelism.

Project Modules

1.
CPU Sequential Optimizations: Optimized a sequential Sobel edge detection algorithm on the CPU. Applied loop interchange, unrolling, fusion, function inlining, and common subexpression elimination to minimize execution time.  

2.
OpenMP K-Means Clustering: Parallelized the K-Means clustering algorithm using OpenMP. Scaled the application on a multi-core Intel Xeon processor utilizing up to 56 threads, optimizing loop scheduling and chunk sizes.  

3.
CUDA 2D Convolution: Implemented a 2D Convolution image filter on the GPU using CUDA. Optimized thread grid geometry, handled memory transfers, and resolved warp divergence issues using shared memory padding.  

4.
CUDA CLAHE Image Processing: Developed a CUDA-accelerated Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm. Utilized shared memory and atomic operations for tiled histogram generation, clipping, and CDF calculation, followed by bilinear interpolation.  

5.
CUDA N-Body Simulation: Accelerated an N-Body simulation using both OpenMP and CUDA. Implemented shared memory tiling, transitioned data layouts from Array of Structures (AoS) to Structure of Arrays (SoA) for coalesced access, and utilized Asynchronous CUDA Streams with pinned memory to hide PCIe transfer latency. Achieved a peak throughput of 405 GInt/s on a Tesla K80 GPU.