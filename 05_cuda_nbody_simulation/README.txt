Problem Statement: CUDA N-Body Simulation

This project accelerates an N-Body physics simulation using advanced CUDA optimization techniques, focusing on maximizing throughput and minimizing memory latency.  

Shared Memory Tiling: Reduced global memory bandwidth saturation by having threads cooperatively load tiles of body positions into fast shared memory.
Memory Layout Optimization: Transitioned the data layout from an Array of Structures (AoS) to a Structure of Arrays (SoA), enabling fully coalesced memory access and 100 percent global memory bus utilization.

Asynchronous Pipelining: Utilized CUDA Streams and Pinned Memory to overlap Host-to-Device transfers, Kernel Execution, and Device-to-Host transfers, effectively hiding the PCIe bus latency.

Hardware Intrinsics: Exploited hardware-accelerated special function units by using the rsqrtf intrinsic for distance calculations, coupled with explicit loop unrolling.