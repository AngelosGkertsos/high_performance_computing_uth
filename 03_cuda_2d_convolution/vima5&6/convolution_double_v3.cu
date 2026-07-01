/*
* Grid Implementation + Timing + Double Precision
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "gputimer.h" 

unsigned int filter_radius;

#define FILTER_LENGTH   (2 * filter_radius + 1)
#define ABS(val)        ((val)<0.0 ? (-(val)) : (val))
#define BLOCK_SIZE      16 

// --- KERNELS (DOUBLE) ---
__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
                                  int imageW, int imageH, int filterR) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= imageW || iy >= imageH) return;
    int k = iy * imageW + ix;
    double sum = 0;
    for (int j = -filterR; j <= filterR; j++) {
        int d = ix + j;
        if (d >= 0 && d < imageW) {
            sum += d_Src[iy * imageW + d] * d_Filter[filterR - j];
        }
    }
    d_Dst[k] = sum;
}

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter, 
                                     int imageW, int imageH, int filterR) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= imageW || iy >= imageH) return;
    int k = iy * imageW + ix;
    double sum = 0;
    for (int j = -filterR; j <= filterR; j++) {
        int d = iy + j;
        if (d >= 0 && d < imageH) {
            sum += d_Src[d * imageW + ix] * d_Filter[filterR - j];
        }
    }
    d_Dst[k] = sum;
}

// --- CPU REFERENCE (DOUBLE) ---
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {
  int x, y, k;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;
      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
             int imageW, int imageH, int filterR) {
  int x, y, k;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;
      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;
        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

double get_cpu_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_usec / 1000.0;
}

// --- MAIN ---
int main(int argc, char **argv) {
    
    double *h_Filter, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;
    double *d_Filter, *d_Input, *d_Buffer, *d_Output;
    int imageW, imageH;
    unsigned int i;
    
    // Variables for timing
    double cpu_start, cpu_end;
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

    size_t size_bytes = imageW * imageH * sizeof(double);
    size_t filter_bytes = FILTER_LENGTH * sizeof(double);

    // Host Malloc
    h_Filter    = (double *)malloc(filter_bytes);
    h_Input     = (double *)malloc(size_bytes);
    h_Buffer    = (double *)malloc(size_bytes);
    h_OutputCPU = (double *)malloc(size_bytes);
    h_OutputGPU = (double *)malloc(size_bytes);
    
    if (!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU) {
        fprintf(stderr, "Error: Host memory allocation failed.\n");
        exit(1);
    }

    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) h_Filter[i] = (double)(rand() % 16);
    for (i = 0; i < imageW * imageH; i++) h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;

    // --- CPU TIMING ---
    if (imageW <= 16384) {
        printf("Starting CPU computation...\n");
        cpu_start = get_cpu_time();
        convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
        convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);
        cpu_end = get_cpu_time();
        printf("CPU Time: %f ms\n", cpu_end - cpu_start);
    } else {
        printf("Skipping CPU computation (too large).\n");
    }

    // --- GPU TIMING ---
    printf("Starting GPU computation...\n");

    if (cudaMalloc((void **)&d_Filter, filter_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Input, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Buffer, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Output, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }

    gpu_timer.Start();

    cudaMemcpy(d_Filter, h_Filter, filter_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, size_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((imageW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (imageH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolutionRowGPU<<<numBlocks, threadsPerBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    
    convolutionColumnGPU<<<numBlocks, threadsPerBlock>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    cudaMemcpy(h_OutputGPU, d_Output, size_bytes, cudaMemcpyDeviceToHost);

    gpu_timer.Stop();
    printf("GPU Time (Transfers + Compute): %f ms\n", gpu_timer.Elapsed());

    // --- ACCURACY CHECK ---
    printf("\nAnalyzing Accuracy (Iterative Tolerance Check)...\n");
    
    double tolerance = 0.5;
    double last_passing_tolerance = -1.0; // Flag to indicate failure
    int passed_at_least_once = 0;

    // Loop until failure occurs
    while (1) {
        int passed = 1;
        
        for (i = 0; i < imageW * imageH; i++) {
            if (ABS(h_OutputCPU[i] - h_OutputGPU[i]) > tolerance) {
                passed = 0;
                break; 
            }
        }

        if (passed) {
            last_passing_tolerance = tolerance;
            passed_at_least_once = 1;
            // printf("Passed at tolerance: %g\n", tolerance); // Debug 
            tolerance /= 10.0; // Reduce tolerance (stricter)
        } else {
            // Failed at this tolerance, stop.
            break;
        }
    }

    printf("--------------------------------------------------\n");
    if (passed_at_least_once) {
        printf("Validation Results:\n");
        printf("Filter Radius: %d (Size: %d)\n", filter_radius, FILTER_LENGTH);
        printf("Max Passing Accuracy (Tolerance): %g\n", last_passing_tolerance);
        // an to 0.05 pernaei -> 1 decimal, 0.005 -> 2 decimals, ktl
        printf("Approximate Decimal Digits of Precision: %d\n", (int)log10(0.5 / last_passing_tolerance));
    } else {
        printf("Validation FAILED even at high tolerance (0.5).\n");
        printf("Likely cause: Kernel execution failure (too many threads) or logic error.\n");
    }
    printf("--------------------------------------------------\n");

    cudaFree(d_Filter); cudaFree(d_Input); cudaFree(d_Buffer); cudaFree(d_Output);
    free(h_Filter); free(h_Input); free(h_Buffer); free(h_OutputCPU); free(h_OutputGPU);

    return 0;
}