Problem Statement: OpenMP K-Means Clustering

This project focuses on parallelizing a sequential implementation of the K-Means clustering algorithm using OpenMP directives.  

The parallelization strategy required profiling the code to identify intensive loops and applying appropriate OpenMP parallel regions. Key challenges included defining private variables per thread and tuning the loop scheduling and chunk sizes to achieve optimal load balancing. The application was benchmarked on a dual-socket Intel Xeon system, measuring execution time and speedup across 1, 4, 8, 14, 28, and 56 threads.