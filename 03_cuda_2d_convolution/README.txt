Problem Statement: CUDA 2D Convolution

The goal of this assignment was to transition from CPU programming to GPU acceleration by implementing a 2D Convolution filter using NVIDIA CUDA.  

The implementation manages device memory allocation, host-to-device transfers, and kernel execution using 2D block and grid geometries. A significant part of the optimization involved resolving warp divergence issues caused by boundary checks, which was addressed by applying memory padding techniques. Performance was evaluated by comparing GPU execution times against the CPU baseline across various image sizes and filter radii.