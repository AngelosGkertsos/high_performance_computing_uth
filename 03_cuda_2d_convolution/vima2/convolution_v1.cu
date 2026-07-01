/*
* Single Block Separable Convolution
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

unsigned int filter_radius;

#define FILTER_LENGTH   (2 * filter_radius + 1)
#define ABS(val)        ((val)<0.0 ? (-(val)) : (val))

////////////////////////////////////////////////////////////////////////////////
// GPU Kernels
////////////////////////////////////////////////////////////////////////////////

// GPU Kernels (Double Precision)
// __global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
//                                   int imageW, int imageH, int filterR) {

// Row Convolution Kernel (GPU) - Single Block Logic
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                                  int imageW, int imageH, int filterR) {
    int k = threadIdx.x; // Single block assumption

    if (k >= imageW * imageH) return;

    int y = k / imageW; 
    int x = k % imageW; 

    float sum = 0;
    
    for (int j = -filterR; j <= filterR; j++) {
        int d = x + j;
        if (d >= 0 && d < imageW) {
            sum += d_Src[y * imageW + d] * d_Filter[filterR - j];
        }
    }
    d_Dst[k] = sum;
}

// __global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter, 
//                                      int imageW, int imageH, int filterR) {

// Column Convolution Kernel (GPU) - Single Block Logic
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                                     int imageW, int imageH, int filterR) {
    int k = threadIdx.x; // Single block assumption

    if (k >= imageW * imageH) return;

    int y = k / imageW;
    int x = k % imageW;

    float sum = 0;

    for (int j = -filterR; j <= filterR; j++) {
        int d = y + j;
        if (d >= 0 && d < imageH) {
            sum += d_Src[d * imageW + x] * d_Filter[filterR - j];
        }
    }
    d_Dst[k] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Functions
////////////////////////////////////////////////////////////////////////////////

// CPU Reference (Double Precision)
// void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
//                        int imageW, int imageH, int filterR) {
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
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

// void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
//              int imageW, int imageH, int filterR) {
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
        h_Dst[y * imageW + x] = sum;
      }
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
    cudaError_t err; // Variable to check CUDA errors

    if (argc != 3) {
        printf("Usage: %s <image_size> <filter_radius>\n", argv[0]);
        return 1;
    }
    
    imageW = atoi(argv[1]);
    filter_radius = atoi(argv[2]);
    imageH = imageW;

    printf("Image Size: %dx%d, Filter Radius: %d\n", imageW, imageH, filter_radius);
    printf("Allocating and initializing host arrays...\n");

    size_t size_bytes = imageW * imageH * sizeof(float);
    size_t filter_bytes = FILTER_LENGTH * sizeof(float);

    // ---------------- Host Allocation with Error Checking ----------------
    h_Filter    = (float *)malloc(filter_bytes);
    h_Input     = (float *)malloc(size_bytes);
    h_Buffer    = (float *)malloc(size_bytes);
    h_OutputCPU = (float *)malloc(size_bytes);
    h_OutputGPU = (float *)malloc(size_bytes);

    if (!h_Filter || !h_Input || !h_Buffer || !h_OutputCPU || !h_OutputGPU) {
        fprintf(stderr, "Error: Host memory allocation failed.\n");
        exit(1);
    }

    // Initialization
    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }
    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    // ---------------- CPU Execution ----------------
    printf("CPU computation...\n");
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);

    // ---------------- GPU Execution ----------------
    printf("GPU computation (Single Block approach)...\n");

    if (cudaMalloc((void **)&d_Filter, filter_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Input, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Buffer, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }
    if (cudaMalloc((void **)&d_Output, size_bytes) != cudaSuccess) { fprintf(stderr, "GPU malloc failed\n"); exit(1); }

    // Host -> Device Transfer
    cudaMemcpy(d_Filter, h_Filter, filter_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, size_bytes, cudaMemcpyHostToDevice);

    // Launch Configuration (Single Block)
    dim3 grid(1); 
    dim3 block(imageW * imageH); 

    // Kernel Execution
    convolutionRowGPU<<<grid, block>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    
    convolutionColumnGPU<<<grid, block>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    // Check for Kernel Errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU Kernel Error: %s\n", cudaGetErrorString(err));
    }

    // Device -> Host Transfer
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

    // Free Memory
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_Output);

    free(h_Filter);
    free(h_Input);
    free(h_Buffer);
    free(h_OutputCPU);
    free(h_OutputGPU);

    return 0;
}

// Main for double precision testing to check implementation correctness //
// int main(int argc, char **argv) {
    
//     double *h_Filter, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;
//     double *d_Filter, *d_Input, *d_Buffer, *d_Output;
//     int imageW, imageH;
//     unsigned int i;
//     cudaError_t err;

//     if (argc != 3) {
//         printf("Usage: %s <image_size> <filter_radius>\n", argv[0]);
//         return 1;
//     }
//     // Προσοχή: Η atoi θέλει το <stdlib.h>
//     imageW = atoi(argv[1]);
//     filter_radius = atoi(argv[2]);

//     printf("=== DOUBLE PRECISION TEST ===\n");

//     imageH = imageW;

//     size_t size_bytes = imageW * imageH * sizeof(double);
//     size_t filter_bytes = FILTER_LENGTH * sizeof(double);

//     h_Filter    = (double *)malloc(filter_bytes);
//     h_Input     = (double *)malloc(size_bytes);
//     h_Buffer    = (double *)malloc(size_bytes);
//     h_OutputCPU = (double *)malloc(size_bytes);
//     h_OutputGPU = (double *)malloc(size_bytes);

//     srand(200);
//     for (i = 0; i < FILTER_LENGTH; i++) {
//         h_Filter[i] = (double)(rand() % 16);
//     }
//     for (i = 0; i < imageW * imageH; i++) {
//         h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
//     }

//     convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
//     convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);

//     cudaMalloc((void **)&d_Filter, filter_bytes);
//     cudaMalloc((void **)&d_Input, size_bytes);
//     cudaMalloc((void **)&d_Buffer, size_bytes);
//     cudaMalloc((void **)&d_Output, size_bytes);

//     cudaMemcpy(d_Filter, h_Filter, filter_bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_Input, h_Input, size_bytes, cudaMemcpyHostToDevice);

//     dim3 grid(1); 
//     dim3 block(imageW * imageH); 

//     convolutionRowGPU<<<grid, block>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
//     cudaDeviceSynchronize();
//     convolutionColumnGPU<<<grid, block>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
//     cudaDeviceSynchronize();

//     err = cudaGetLastError();
//     if (err != cudaSuccess) printf("Kernel Error: %s\n", cudaGetErrorString(err));

//     cudaMemcpy(h_OutputGPU, d_Output, size_bytes, cudaMemcpyDeviceToHost);

//     // Check Accuracy
//     printf("\nChecking Accuracy with DOUBLES...\n");
//     double tolerance = 0.5;
//     double last_passing = -1.0;

//     while (tolerance >= 1e-12) { // Πάμε πολύ χαμηλά, μέχρι 10^-12
//         int passed = 1;
//         for (i = 0; i < imageW * imageH; i++) {
//             if (ABS(h_OutputCPU[i] - h_OutputGPU[i]) > tolerance) {
//                 passed = 0;
//                 break; 
//             }
//         }
//         if (passed) {
//             last_passing = tolerance;
//             tolerance /= 10.0;
//         } else {
//             break;
//         }
//     }

//     if (last_passing > 0) {
//         printf("PASSED! Max Accuracy: %g\n", last_passing);
//         printf("This proves the logic is CORRECT and any previous errors were due to float precision.\n");
//     } else {
//         printf("FAILED! Even doubles failed. There is a bug in the logic.\n");
//     }

//     cudaFree(d_Filter); cudaFree(d_Input); cudaFree(d_Buffer); cudaFree(d_Output);
//     free(h_Filter); free(h_Input); free(h_Buffer); free(h_OutputCPU); free(h_OutputGPU);

//     return 0;
// }