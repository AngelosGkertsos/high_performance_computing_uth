/*
* Removes 'if' checks inside kernel to avoid Warp Divergence.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gputimer.h" 

unsigned int filter_radius;
#define FILTER_LENGTH (2 * filter_radius + 1)
#define BLOCK_SIZE 16 

// --- KERNELS (NO IF CHECKS) ---

__global__ void convolutionRowGPU_Pad(float *d_Dst, float *d_Src, float *d_Filter, 
                                      int imageW, int imageH, int filterR) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (ix >= imageW || iy >= (imageH + 2*filterR)) return;

    float sum = 0;
    int padded_width = imageW + 2 * filterR;
    
    for (int j = -filterR; j <= filterR; j++) {
        sum += d_Src[iy * padded_width + (ix + filterR + j)] * d_Filter[filterR - j];
    }
    
    d_Dst[iy * imageW + ix] = sum;
}

__global__ void convolutionColumnGPU_Pad(float *d_Dst, float *d_Src, float *d_Filter, 
                                         int imageW, int imageH, int filterR) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= imageW || iy >= imageH) return;

    float sum = 0;
    for (int j = -filterR; j <= filterR; j++) {
        int row_idx = iy + filterR + j;
        sum += d_Src[row_idx * imageW + ix] * d_Filter[filterR - j];
    }
    d_Dst[iy * imageW + ix] = sum;
}

int main(int argc, char **argv) {
    float *h_Filter, *h_Input, *h_InputPadded, *h_OutputGPU;
    float *d_Filter, *d_InputPadded, *d_Buffer, *d_Output;
    int imageW, imageH;
    unsigned int i;
    GpuTimer gpu_timer; 

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <Image Size N> <Filter Radius>\n", argv[0]);
        exit(1);
    }

    imageW = atoi(argv[1]);
    filter_radius = atoi(argv[2]);
    imageH = imageW;

    printf("Image Size: %dx%d, Filter Radius: %d\n", imageW, imageH, filter_radius);
    printf("Allocating and initializing host arrays...\n");

    // Sizes
    int paddedW = imageW + 2 * filter_radius;
    int paddedH = imageH + 2 * filter_radius;
    
    size_t filter_bytes = FILTER_LENGTH * sizeof(float);
    size_t output_bytes = imageW * imageH * sizeof(float);
    size_t padded_input_bytes = paddedW * paddedH * sizeof(float);
    // Buffer needs to store N width, but (N + 2R) height
    size_t buffer_bytes = imageW * paddedH * sizeof(float);

    // Allocations
    h_Filter = (float*)malloc(filter_bytes);
    h_Input = (float*)malloc(output_bytes); // Original size for init
    h_InputPadded = (float*)calloc(paddedW * paddedH, sizeof(float)); // Calloc zeros the padding
    h_OutputGPU = (float*)malloc(output_bytes);

    if (!h_Filter || !h_Input || !h_InputPadded || !h_OutputGPU) {
        fprintf(stderr, "Error: Host memory allocation failed.\n");
        exit(1);
    }

    // Init
    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) h_Filter[i] = (float)(rand() % 16);
    for (i = 0; i < imageW * imageH; i++) h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;

    // COPY Input to Center of InputPadded
    for (int y = 0; y < imageH; y++) {
        for (int x = 0; x < imageW; x++) {
            h_InputPadded[(y + filter_radius) * paddedW + (x + filter_radius)] = h_Input[y * imageW + x];
        }
    }

    // GPU Alloc
    if (cudaMalloc((void **)&d_Filter, filter_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc(&d_InputPadded, padded_input_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc(&d_Buffer, buffer_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc(&d_Output, output_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }

    printf("Starting GPU computation (Padding Optimization)...\n");
    gpu_timer.Start();

    cudaMemcpy(d_Filter, h_Filter, filter_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_InputPadded, h_InputPadded, padded_input_bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    // ROW KERNEL LAUNCH
    // Runs for width=N, height=N+2R
    dim3 blocksRow((imageW + 15)/16, (paddedH + 15)/16);
    convolutionRowGPU_Pad<<<blocksRow, threads>>>(d_Buffer, d_InputPadded, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    // COL KERNEL LAUNCH
    // Runs for width=N, height=N
    dim3 blocksCol((imageW + 15)/16, (imageH + 15)/16);
    convolutionColumnGPU_Pad<<<blocksCol, threads>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    cudaMemcpy(h_OutputGPU, d_Output, output_bytes, cudaMemcpyDeviceToHost);

    gpu_timer.Stop();
    printf("GPU Time (Transfers + Compute): %f ms\n", gpu_timer.Elapsed());

    // Clean up
    cudaFree(d_Filter); cudaFree(d_InputPadded); cudaFree(d_Buffer); cudaFree(d_Output);
    free(h_Filter); free(h_Input); free(h_InputPadded); free(h_OutputGPU);
    return 0;
}