Problem Statement: CPU Sequential Optimizations

The objective of this assignment was to manually apply compiler-level optimizations to a sequential C program implementing the Sobel edge detection operator.  

Step-by-step optimizations were applied to reduce the execution time without altering the mathematical results. The techniques implemented include Loop Interchange to improve memory locality, Loop Unrolling to reduce control overhead, Loop Fusion, Function Inlining, Loop Invariant Code Motion, Common Subexpression Elimination, and Strength Reduction. Performance was evaluated by measuring execution time and verifying the Peak Signal to Noise Ratio (PSNR) against a reference image.