Problem Statement: CUDA CLAHE Image Processing

This project implements the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm on the GPU.  

The algorithm was broken down into parallel stages. First, the image is divided into tiles where local histograms are generated, clipped, and converted to Cumulative Distribution Functions (CDF) using Shared Memory and atomic operations to prevent data races. To avoid visible boundaries between tiles, Bilinear Interpolation was implemented, mapping each pixel based on the CDFs of its four nearest tile centers. The implementation demonstrates advanced GPU memory management and parallel reduction patterns.