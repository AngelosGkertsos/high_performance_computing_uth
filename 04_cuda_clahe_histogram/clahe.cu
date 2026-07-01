#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "clahe.h"

// --- CONSTANT MEMORY ---
// Stored in constant cache for fast broadcast access to all threads
__constant__ int c_width;
__constant__ int c_height;
__constant__ int c_tile_size;
__constant__ int c_clip_limit;

// --- BANK CONFLICT MACROS ---
// Offsets indices to avoid shared memory bank conflicts (32 banks)
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define SHM_IDX(n) ((n) + CONFLICT_FREE_OFFSET(n))

// Texture reference for cached read-only access during interpolation
texture<unsigned char, 1, cudaReadModeElementType> texLUT;

// --- KERNEL 1: Histogram generation, Clipping, and CDF calculation ---
__global__ void calculate_lut_kernel(unsigned char* img_in, unsigned char* all_luts, int grid_w, int y_grid_offset) {
    // Shared memory with padding to prevent bank conflicts
    __shared__ int s_hist[256 + 16]; 
    __shared__ int s_excess;

    int tx = threadIdx.x; 
    int bx = blockIdx.x;
    int by = blockIdx.y + y_grid_offset;

    // Initialize shared memory
    s_hist[SHM_IDX(tx)] = 0;
    if (tx == 0) s_excess = 0;
    __syncthreads();

    // Determine tile boundaries
    int start_x = bx * c_tile_size;
    int start_y = by * c_tile_size;
    int end_x = min(start_x + c_tile_size, c_width);
    int end_y = min(start_y + c_tile_size, c_height);

    // --- STEP 1: BUILD HISTOGRAM ---
    int tile_pixels = c_tile_size * c_tile_size; 
    
    // Iterate over all pixels in the tile using a stride loop
    for (int i = tx; i < tile_pixels; i += blockDim.x) {
        int local_y = i / c_tile_size;
        int local_x = i % c_tile_size;
        
        int global_x = start_x + local_x;
        int global_y = start_y + local_y;

        // Clamp coordinates to stay within image bounds (prevents segfault without branching)
        int safe_x = min(global_x, c_width - 1);
        int safe_y = min(global_y, c_height - 1);

        unsigned char val = img_in[safe_y * c_width + safe_x];

        // Predicate: 1 if valid pixel, 0 if padding.
        int is_active = (global_x < c_width) && (global_y < c_height);

        // Atomic update avoids race conditions. Adds 0 if pixel is out of bounds.
        atomicAdd(&s_hist[SHM_IDX(val)], is_active);
    }
    __syncthreads();

    // --- STEP 2: CLIP HISTOGRAM ---
    int index_w_padding = SHM_IDX(tx);
    int bin_val = s_hist[index_w_padding];
    
    // Calculate clipped value using min() instead of if()
    int clipped_val = min(bin_val, c_clip_limit);
    
    // Calculate excess pixels to redistribute
    int diff = bin_val - clipped_val;

    // Accumulate total excess
    atomicAdd(&s_excess, diff);
    
    // Update the bin with the clipped value
    s_hist[index_w_padding] = clipped_val;
    
    __syncthreads();

    // --- STEP 3: REDISTRIBUTE EXCESS ---
    int total_excess = s_excess;
    int avg_inc = total_excess / 256; // Simple equal redistribution
    int val = s_hist[index_w_padding] + avg_inc;
    s_hist[index_w_padding] = val;
    __syncthreads();

    // --- STEP 4: PARALLEL SCAN (Prefix Sum) ---
    // scan to convert Histogram -> CDF
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        int in1;
        if (stride <= tx) {
            int idx_read = tx - stride;
            in1 = s_hist[SHM_IDX(idx_read)];
        }
        __syncthreads();
        if (stride <= tx) {
            s_hist[index_w_padding] += in1;
        }
    }
    __syncthreads();

    // --- STEP 5: FINALIZE LUT ---
    // Map CDF to [0-255] and write to global memory
    if (tx < 256) {
        int cdf = s_hist[index_w_padding];
        int total_pixels = (end_x - start_x) * (end_y - start_y);
        
        // Normalize CDF
        int final_val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
        final_val = min(final_val, 255); 
        
        int lut_index = (by * grid_w + bx) * 256 + tx;
        all_luts[lut_index] = (unsigned char)final_val;
    }
}

// --- KERNEL 2: Bilinear Interpolation ---
__global__ void render_clahe_kernel(unsigned char* img_in, unsigned char* img_out, int grid_w, int grid_h, int y_offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_y = y + y_offset; // Adjust for stream chunking

    if (x >= c_width || global_y >= c_height) return;

    // Calculate normalized coordinates relative to tiles
    float tx_f = (float)x / c_tile_size - 0.5f;
    float ty_f = (float)global_y / c_tile_size - 0.5f;

    // Get the integer part (top-left tile index)
    int x1_raw = (int)floorf(tx_f);
    int y1_raw = (int)floorf(ty_f);

    // Clamp tile indices to ensure we don't read outside the grid
    int x1 = max(0, x1_raw);
    int y1 = max(0, y1_raw);
    int x2 = min(grid_w - 1, x1_raw + 1);
    int y2 = min(grid_h - 1, y1_raw + 1);
    
    // Calculate weights for interpolation
    float x_weight = tx_f - x1_raw;
    float y_weight = ty_f - y1_raw;

    unsigned char val = img_in[global_y * c_width + x];

    // Fetch LUT values from the four surrounding tiles via Texture Memory
    unsigned char tl = tex1Dfetch(texLUT, (y1 * grid_w + x1) * 256 + val);
    unsigned char tr = tex1Dfetch(texLUT, (y1 * grid_w + x2) * 256 + val);
    unsigned char bl = tex1Dfetch(texLUT, (y2 * grid_w + x1) * 256 + val);
    unsigned char br = tex1Dfetch(texLUT, (y2 * grid_w + x2) * 256 + val);

    // Bilinear Interpolation Formula
    float top = tl * (1.0f - x_weight) + tr * x_weight;
    float bot = bl * (1.0f - x_weight) + br * x_weight;
    float final_val = top * (1.0f - y_weight) + bot * y_weight;

    img_out[global_y * c_width + x] = (unsigned char)(final_val + 0.5f);
}

// --- HOST HELPER FUNCTIONS ---
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;
    in_file = fopen(path, "rb");
    if (in_file == NULL){ printf("Input file not found!\n"); exit(1); }
    fscanf(in_file, "%s", sbuf); 
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); 
    // Use Pinned Memory (cudaMallocHost) for faster transfers
    cudaMallocHost((void**)&result.img, result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img) {
    if(img.img) cudaFreeHost(img.img);
}

// --- MAIN PROCESSING FUNCTION ---
PGM_IMG apply_clahe(PGM_IMG img_in) {
    PGM_IMG img_out;
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    
    // Allocate Pinned Memory for output
    cudaMallocHost((void**)&img_out.img, img_in.w * img_in.h * sizeof(unsigned char));

    int tile_size = TILE_SIZE;
    int clip_limit = CLIP_LIMIT;
    int grid_w = (img_in.w + tile_size - 1) / tile_size;
    int grid_h = (img_in.h + tile_size - 1) / tile_size;

    unsigned char *d_img_in, *d_img_out;
    unsigned char *d_all_luts;
    
    size_t img_bytes = img_in.w * img_in.h * sizeof(unsigned char);
    size_t lut_bytes = grid_w * grid_h * 256 * sizeof(unsigned char);

    // Device memory allocation
    cudaMalloc(&d_img_in, img_bytes);
    cudaMalloc(&d_img_out, img_bytes);
    cudaMalloc(&d_all_luts, lut_bytes);

    // Copy parameters to Constant Memory
    cudaMemcpyToSymbol(c_width, &img_in.w, sizeof(int));
    cudaMemcpyToSymbol(c_height, &img_in.h, sizeof(int));
    cudaMemcpyToSymbol(c_tile_size, &tile_size, sizeof(int));
    cudaMemcpyToSymbol(c_clip_limit, &clip_limit, sizeof(int));

    // Bind texture to the LUT array for fast caching
    cudaBindTexture(0, texLUT, d_all_luts, lut_bytes);

    // --- SETUP STREAMS ---
    int n_streams = 4;
    cudaStream_t streams[4];
    for(int i=0; i<n_streams; i++) cudaStreamCreate(&streams[i]);

    // --- PHASE 1: LUT CALCULATION ---
    int tiles_per_chunk = grid_h / n_streams; 
    for (int i = 0; i < n_streams; ++i) {
        // Calculate chunk boundaries
        int tile_start_y = i * tiles_per_chunk;
        int current_tiles_h = (i == n_streams - 1) ? (grid_h - tile_start_y) : tiles_per_chunk;
        int pixel_start_y = tile_start_y * tile_size;
        int pixel_end_y = (tile_start_y + current_tiles_h) * tile_size;
        if (pixel_end_y > img_in.h) pixel_end_y = img_in.h; 
        
        int pixels_h = pixel_end_y - pixel_start_y;
        size_t chunk_bytes = img_in.w * pixels_h * sizeof(unsigned char);
        int pixel_offset = pixel_start_y * img_in.w;

        // Async Copy: Host -> Device
        if (chunk_bytes > 0) {
            cudaMemcpyAsync(d_img_in + pixel_offset, img_in.img + pixel_offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        }
        
        // Launch Histogram/LUT kernel for this chunk
        dim3 gridHist(grid_w, current_tiles_h);
        dim3 blockHist(256); 
        calculate_lut_kernel<<<gridHist, blockHist, 0, streams[i]>>>(d_img_in, d_all_luts, grid_w, tile_start_y);
    }
    cudaDeviceSynchronize();

    // --- PHASE 2: RENDERING (INTERPOLATION) ---
    int chunk_h_render = img_in.h / n_streams;
    dim3 blockRender(32, 32); 
    for (int i = 0; i < n_streams; ++i) {
        int y_offset = i * chunk_h_render;
        int current_h = (i == n_streams - 1) ? (img_in.h - y_offset) : chunk_h_render;
        size_t chunk_bytes = img_in.w * current_h * sizeof(unsigned char);
        int offset_pixels = y_offset * img_in.w;
        
        // Launch Render kernel
        dim3 gridRender((img_in.w + blockRender.x - 1) / blockRender.x, (current_h + blockRender.y - 1) / blockRender.y);
        render_clahe_kernel<<<gridRender, blockRender, 0, streams[i]>>>(d_img_in, d_img_out, grid_w, grid_h, y_offset);
        
        // Async Copy: Device -> Host
        cudaMemcpyAsync(img_out.img + offset_pixels, d_img_out + offset_pixels, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();

    // Cleanup
    for(int i=0; i<n_streams; i++) cudaStreamDestroy(streams[i]);
    cudaUnbindTexture(texLUT);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_all_luts);
    return img_out;
}