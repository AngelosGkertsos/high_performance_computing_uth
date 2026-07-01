/*
* Grid Implementation
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

unsigned int filter_radius;

#define FILTER_LENGTH   (2 * filter_radius + 1)
#define ABS(val)        ((val)<0.0 ? (-(val)) : (val))
#define BLOCK_SIZE      16 // kathe block 16x16 = 256 threads

////////////////////////////////////////////////////////////////////////////////
// GPU Kernels (Grid Logic)
////////////////////////////////////////////////////////////////////////////////

// Row Convolution Kernel
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                                  int imageW, int imageH, int filterR) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    int iy = blockIdx.y * blockDim.y + threadIdx.y; 

    if (ix >= imageW || iy >= imageH) return;

    int k = iy * imageW + ix; 
    float sum = 0;
    
    for (int j = -filterR; j <= filterR; j++) {
        int d = ix + j; 
        if (d >= 0 && d < imageW) {
            sum += d_Src[iy * imageW + d] * d_Filter[filterR - j];
        }
    }
    d_Dst[k] = sum;
}

// Column Convolution Kernel
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                                     int imageW, int imageH, int filterR) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= imageW || iy >= imageH) return;

    int k = iy * imageW + ix;
    float sum = 0;

    for (int j = -filterR; j <= filterR; j++) {
        int d = iy + j; 
        if (d >= 0 && d < imageH) {
            sum += d_Src[d * imageW + ix] * d_Filter[filterR - j];
        }
    }
    d_Dst[k] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Functions
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {
  int x, y, k;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;
      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     
      }
      h_Dst[y * imageW + x] = sum;
    }
  }
}

void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
             int imageW, int imageH, int filterR) {
  int x, y, k;
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;
      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;
        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
      }
      h_Dst[y * imageW + x] = sum;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float *h_Filter, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;
    float *d_Filter, *d_Input, *d_Buffer, *d_Output;
    int imageW, imageH;
    unsigned int i;
    cudaError_t err;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <Image Size N> <Filter Radius>\n", argv[0]);
        exit(1);
    }

    imageW = atoi(argv[1]);
    filter_radius = atoi(argv[2]);
    imageH = imageW;

    printf("Image Size: %dx%d, Filter Radius: %d\n", imageW, imageH, filter_radius);
    printf("Allocating and initializing host arrays...\n");

    size_t size_bytes = imageW * imageH * sizeof(float);
    size_t filter_bytes = FILTER_LENGTH * sizeof(float);

    printf("Allocating Host memory...\n");
    h_Filter    = (float *)malloc(filter_bytes);
    h_Input     = (float *)malloc(size_bytes);
    h_Buffer    = (float *)malloc(size_bytes);
    h_OutputCPU = (float *)malloc(size_bytes);
    h_OutputGPU = (float *)malloc(size_bytes);

    if (!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU) {
        fprintf(stderr, "Error: Host memory allocation failed.\n");
        exit(1);
    }

    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }
    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    // --- CPU Computation ---
    printf("CPU computation...\n");
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);

    // --- GPU Computation ---
    printf("GPU computation (Grid approach)...\n");

    if (cudaMalloc((void **)&d_Filter, filter_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Input, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Buffer, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Output, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }

    cudaMemcpy(d_Filter, h_Filter, filter_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, size_bytes, cudaMemcpyHostToDevice);

    // --- GRID CONFIGURATION ---
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    dim3 numBlocks((imageW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (imageH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching Grid: Blocks(%d, %d), Threads/Block(%d, %d)\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    convolutionRowGPU<<<numBlocks, threadsPerBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    
    convolutionColumnGPU<<<numBlocks, threadsPerBlock>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU Kernel Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_OutputGPU, d_Output, size_bytes, cudaMemcpyDeviceToHost);

    // ---------------- Accuracy Check Loop ----------------
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