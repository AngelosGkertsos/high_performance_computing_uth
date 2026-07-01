#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define SOFTENING 0.01f
#define BLOCK_SIZE 256
#define N_STREAMS 4

// Structure for file I/O (Array of Structures)
typedef struct {
    float x, y, z, vx, vy, vz;
} Body_File;

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )
void checkCuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, 
                static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

// CUDA Kernel for N-Body integration using SoA layout
__global__ void bodyForceIntegrateKernel_SoA(float * __restrict__ sys_data, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Set pointers to the start of each array (X, Y, Z, VX, VY, VZ)
    float *rx_arr = sys_data;
    float *ry_arr = sys_data + n;
    float *rz_arr = sys_data + 2 * n;
    float *vx_arr = sys_data + 3 * n;
    float *vy_arr = sys_data + 4 * n;
    float *vz_arr = sys_data + 5 * n;

    // Load current body data using read-only cache
    float my_x = __ldg(&rx_arr[i]);
    float my_y = __ldg(&ry_arr[i]);
    float my_z = __ldg(&rz_arr[i]);
    float my_vx = __ldg(&vx_arr[i]);
    float my_vy = __ldg(&vy_arr[i]);
    float my_vz = __ldg(&vz_arr[i]);

    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    // Shared memory buffers for tiling
    __shared__ float sh_x[BLOCK_SIZE];
    __shared__ float sh_y[BLOCK_SIZE];
    __shared__ float sh_z[BLOCK_SIZE];

    int num_tiles = n / BLOCK_SIZE; 

    // Loop over all tiles
    for (int tile = 0; tile < num_tiles; tile++) {
        int load_idx = tile * BLOCK_SIZE + threadIdx.x;
        
        // Load tile data into shared memory
        sh_x[threadIdx.x] = __ldg(&rx_arr[load_idx]);
        sh_y[threadIdx.x] = __ldg(&ry_arr[load_idx]);
        sh_z[threadIdx.x] = __ldg(&rz_arr[load_idx]);
        
        __syncthreads();

        // Compute forces against bodies in the current tile
        #pragma unroll 32
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dz = sh_z[j] - my_z;
            
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr); 
            float invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3; 
            Fy += dy * invDist3; 
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocity
    my_vx += dt * Fx; 
    my_vy += dt * Fy; 
    my_vz += dt * Fz;

    // Update position
    my_x += my_vx * dt; 
    my_y += my_vy * dt; 
    my_z += my_vz * dt;

    // Write updated data back to global memory
    vx_arr[i] = my_vx;
    vy_arr[i] = my_vy;
    vz_arr[i] = my_vz;
    rx_arr[i] = my_x;
    ry_arr[i] = my_y;
    rz_arr[i] = my_z;
}

int main(const int argc, const char *argv[]) {
    int num_systems = 32;       
    int bodies_per_system = 8192;
    int nIters = 20; 
    const float dt = 0.01f;
    
    FILE *fp;
    int total_bodies_all;
    Body_File *h_file_data; 
    float *h_pinned_soa;

    double t_start_app = omp_get_wtime();
    double t_alloc_host = 0.0;
    double t_alloc_device = 0.0;
    double t_exec_pipeline = 0.0;

    // Read dataset metadata
    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Dataset: %d systems, %d bodies/system.\n", num_systems, bodies_per_system);
    } else {
        printf("No dataset. Using random init.\n");
    }

    total_bodies_all = num_systems * bodies_per_system;
    
    size_t total_floats = (size_t)total_bodies_all * 6;
    size_t total_bytes = total_floats * sizeof(float);

    double t1 = omp_get_wtime();
    
    // Allocate host memory (Standard malloc for file data, Pinned for GPU transfers)
    h_file_data = (Body_File *) malloc(total_bodies_all * sizeof(Body_File));
    checkCudaErrors(cudaHostAlloc((void**)&h_pinned_soa, total_bytes, cudaHostAllocDefault));
    
    t_alloc_host = omp_get_wtime() - t1;

    // Initialize data
    if (fp) {
        fread(h_file_data, sizeof(Body_File), total_bodies_all, fp);
        fclose(fp);
    } else {
        float *buf = (float *) h_file_data;
        for (int i = 0; i < 6 * total_bodies_all; i++) {
            buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    // Convert AoS (file format) to SoA (GPU format) using OpenMP
    #pragma omp parallel for
    for (int sys = 0; sys < num_systems; sys++) {
        size_t sys_offset_file = sys * bodies_per_system;
        size_t sys_offset_soa  = sys * bodies_per_system * 6;
        
        float *X  = &h_pinned_soa[sys_offset_soa];
        float *Y  = &h_pinned_soa[sys_offset_soa + bodies_per_system];
        float *Z  = &h_pinned_soa[sys_offset_soa + 2*bodies_per_system];
        float *VX = &h_pinned_soa[sys_offset_soa + 3*bodies_per_system];
        float *VY = &h_pinned_soa[sys_offset_soa + 4*bodies_per_system];
        float *VZ = &h_pinned_soa[sys_offset_soa + 5*bodies_per_system];

        for (int i = 0; i < bodies_per_system; i++) {
            Body_File b = h_file_data[sys_offset_file + i];
            X[i] = b.x;
            Y[i] = b.y;
            Z[i] = b.z;
            VX[i] = b.vx;
            VY[i] = b.vy;
            VZ[i] = b.vz;
        }
    }

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    printf("Running SoA Version on %d GPUs...\n", num_gpus);

    // Context warmup for the GPUs
    #pragma omp parallel num_threads(num_gpus)
    {
        cudaSetDevice(omp_get_thread_num());
        cudaFree(0);
    }

    double start_pipe = 0.0, end_pipe = 0.0;
    double start_malloc = 0.0, end_malloc = 0.0;

    // Multi-GPU Execution Block
    #pragma omp parallel num_threads(num_gpus)
    {
        int dev_id = omp_get_thread_num();
        checkCudaErrors(cudaSetDevice(dev_id)); 

        // Distribute systems among GPUs
        int systems_per_gpu = num_systems / num_gpus;
        int remainder = num_systems % num_gpus;
        int start_sys_idx = dev_id * systems_per_gpu + (dev_id < remainder ? dev_id : remainder);
        int my_system_count = systems_per_gpu + (dev_id < remainder ? 1 : 0);

        if (my_system_count > 0) {
            // Create CUDA Streams
            cudaStream_t streams[N_STREAMS];
            for(int s=0; s<N_STREAMS; s++) checkCudaErrors(cudaStreamCreate(&streams[s]));

            #pragma omp barrier
            if(dev_id==0) start_malloc = omp_get_wtime();
            
            // Allocate Device Memory
            float *d_data;
            size_t my_floats = (size_t)my_system_count * bodies_per_system * 6;
            checkCudaErrors(cudaMalloc((void**)&d_data, my_floats * sizeof(float)));
            
            #pragma omp barrier
            if(dev_id==0) end_malloc = omp_get_wtime();

            float *h_ptr_local = &h_pinned_soa[start_sys_idx * bodies_per_system * 6];

            #pragma omp barrier 
            if(dev_id == 0) start_pipe = omp_get_wtime();

            // Processing Pipeline: Copy -> Compute -> Copy
            for (int i = 0; i < my_system_count; i++) {
                int stream_id = i % N_STREAMS;
                
                size_t galaxy_floats = bodies_per_system * 6;
                size_t galaxy_offset = i * galaxy_floats;
                size_t galaxy_bytes  = galaxy_floats * sizeof(float);
                
                // Async H2D Copy
                checkCudaErrors(cudaMemcpyAsync(&d_data[galaxy_offset], &h_ptr_local[galaxy_offset], 
                                                galaxy_bytes, cudaMemcpyHostToDevice, streams[stream_id]));

                // Launch Kernel
                int grid_size = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;
                for (int iter = 0; iter < nIters; iter++) {
                    bodyForceIntegrateKernel_SoA<<<grid_size, BLOCK_SIZE, 0, streams[stream_id]>>>(
                        &d_data[galaxy_offset], dt, bodies_per_system);
                }

                // Async D2H Copy
                checkCudaErrors(cudaMemcpyAsync(&h_ptr_local[galaxy_offset], &d_data[galaxy_offset], 
                                                galaxy_bytes, cudaMemcpyDeviceToHost, streams[stream_id]));
            }

            // Synchronize and destroy streams
            for(int s=0; s<N_STREAMS; s++) {
                checkCudaErrors(cudaStreamSynchronize(streams[s]));
                checkCudaErrors(cudaStreamDestroy(streams[s]));
            }
            
            #pragma omp barrier
            if(dev_id == 0) end_pipe = omp_get_wtime();

            checkCudaErrors(cudaFree(d_data));
        }
    }

    t_alloc_device = end_malloc - start_malloc;
    t_exec_pipeline = end_pipe - start_pipe;

    // Convert SoA back to AoS
    #pragma omp parallel for
    for (int sys = 0; sys < num_systems; sys++) {
        size_t sys_offset_file = sys * bodies_per_system;
        size_t sys_offset_soa  = sys * bodies_per_system * 6;
        
        float *X  = &h_pinned_soa[sys_offset_soa];
        float *Y  = &h_pinned_soa[sys_offset_soa + bodies_per_system];
        float *Z  = &h_pinned_soa[sys_offset_soa + 2*bodies_per_system];
        float *VX = &h_pinned_soa[sys_offset_soa + 3*bodies_per_system];
        float *VY = &h_pinned_soa[sys_offset_soa + 4*bodies_per_system];
        float *VZ = &h_pinned_soa[sys_offset_soa + 5*bodies_per_system];

        for (int i = 0; i < bodies_per_system; i++) {
            h_file_data[sys_offset_file + i].x  = X[i];
            h_file_data[sys_offset_file + i].y  = Y[i];
            h_file_data[sys_offset_file + i].z  = Z[i];
            h_file_data[sys_offset_file + i].vx = VX[i];
            h_file_data[sys_offset_file + i].vy = VY[i];
            h_file_data[sys_offset_file + i].vz = VZ[i];
        }
    }

    double total_time_app = omp_get_wtime() - t_start_app;
    double total_interactions = (double)bodies_per_system * bodies_per_system * num_systems * nIters;

    printf("\n--- METRICS (SoA Version) ---\n");
    printf("1. Host Allocation Time:   %.4f s\n", t_alloc_host);
    printf("2. Device Allocation Time: %.4f s\n", t_alloc_device);
    printf("3. Pipeline Exec Time:     %.4f s\n", t_exec_pipeline);
    printf("4. Total App Time:         %.4f s\n", total_time_app);
    
    printf("Throughput (Pipeline):     %.3f Billion Interactions / second\n", 
           1e-9 * total_interactions / t_exec_pipeline);

    printf("\n--- VERIFICATION ---\n");
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           h_file_data[0].x, h_file_data[0].y, h_file_data[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           h_file_data[1].x, h_file_data[1].y, h_file_data[1].z);

    cudaFreeHost(h_pinned_soa);
    free(h_file_data);
    return 0;
}