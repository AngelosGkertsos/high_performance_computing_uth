// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE    4096
#define MAX_P 65025 // 255 * 255 for the lookup table values (grayscale images)
#define INPUT_FILE  "input.grey"
#define OUTPUT_FILE "output_sobel.grey"
#define GOLDEN_FILE "golden.grey"

/* The horizontal and vertical operators to be used in the sobel filter */
char horiz_operator[3][3] = {{-1, 0, 1}, 
                             {-2, 0, 2}, 
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1}, 
                            {0, 0, 0}, 
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden, int silent_mode);
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row.          */
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];

// Look up table for the magnitude of the derivative
unsigned char sqrt_lookup_table[MAX_P + 1];

/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.                                                          */

int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]) {
    int res, row_0, row_1, row_2;
    
    res = 0;

    row_0 = (posy - 1)*SIZE;
    row_1 = (posy)*SIZE;
    row_2 = (posy + 1)*SIZE;
    
    posx--;
    res += input[row_0 + posx] * operator[0][0];
    res += input[row_1 + posx] * operator[1][0];
    res += input[row_2 + posx] * operator[2][0];

    posx++;
    res += input[row_0 + posx] * operator[0][1];
    res += input[row_1 + posx] * operator[1][1];
    res += input[row_2 + posx] * operator[2][1];

    posx++;
    res += input[row_0 + posx] * operator[0][2];
    res += input[row_1 + posx] * operator[1][2];
    res += input[row_2 + posx] * operator[2][2];

    return(res);
}

/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.                                  */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden, int silent_mode)
{
    double PSNR = 0, t;
    int i, j;
    unsigned int p;
    int gradient_x, gradient_y;

    struct timespec  tv1, tv2;
    FILE *f_in, *f_out, *f_golden;

    /* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.        */
    memset(output, 0, SIZE*sizeof(unsigned char));
    memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
    for (i = 1; i < SIZE-1; i++) {
        output[i*SIZE] = 0;
        output[i*SIZE + SIZE - 1] = 0;
    }

    /* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.                        */
    f_in = fopen(INPUT_FILE, "r");
    if (f_in == NULL) {
        printf("File " INPUT_FILE " not found\n");
        exit(1);
    }
  
    f_out = fopen(OUTPUT_FILE, "wb");
    if (f_out == NULL) {
        printf("File " OUTPUT_FILE " could not be created\n");
        fclose(f_in);
        exit(1);
    }  
  
    f_golden = fopen(GOLDEN_FILE, "r");
    if (f_golden == NULL) {
        printf("File " GOLDEN_FILE " not found\n");
        fclose(f_in);
        fclose(f_out);
        exit(1);
    }    

    fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
    fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
    fclose(f_in);
    fclose(f_golden);
  
    /* This is the main computation. Get the starting time. */
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    // Create lookup table to avoid sqrt calculation since max resulting value is not
    // greater than 255 for our input. ATTENTION! ONLY FOR GRAYSCALE IMAGES!
    for(int k = 0; k < MAX_P; k++){
        sqrt_lookup_table[k] = sqrt(k);
    }

    /* For each pixel of the output image */
    for (j=1; j<SIZE-1; j+=1) {
        for (i=1; i< (SIZE-1) - 15; i+=16 ) {

            /* Apply the sobel filter and calculate the magnitude *
             * of the derivative.                                 */
            // Pixel i
            gradient_x = convolution2D(i, j, input, horiz_operator);
            gradient_y = convolution2D(i, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i)*SIZE + j] = 255;} else{output[(i)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+1
            gradient_x = convolution2D(i+1, j, input, horiz_operator);
            gradient_y = convolution2D(i+1, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+1)*SIZE + j] = 255;} else{output[(i+1)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+2
            gradient_x = convolution2D(i+2, j, input, horiz_operator);
            gradient_y = convolution2D(i+2, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+2)*SIZE + j] = 255;} else{output[(i+2)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+3
            gradient_x = convolution2D(i+3, j, input, horiz_operator);
            gradient_y = convolution2D(i+3, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+3)*SIZE + j] = 255;} else{output[(i+3)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+4
            gradient_x = convolution2D(i+4, j, input, horiz_operator);
            gradient_y = convolution2D(i+4, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+4)*SIZE + j] = 255;} else{output[(i+4)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+5
            gradient_x = convolution2D(i+5, j, input, horiz_operator);
            gradient_y = convolution2D(i+5, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+5)*SIZE + j] = 255;} else{output[(i+5)*SIZE + j] = sqrt_lookup_table[p];}
            
            // Repeat 
            gradient_x = convolution2D(i+6, j, input, horiz_operator);
            gradient_y = convolution2D(i+6, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+6)*SIZE + j] = 255;} else{output[(i+6)*SIZE + j] = sqrt_lookup_table[p];}

            gradient_x = convolution2D(i+7, j, input, horiz_operator);
            gradient_y = convolution2D(i+7, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+7)*SIZE + j] = 255;} else{output[(i+7)*SIZE + j] = sqrt_lookup_table[p];}

            gradient_x = convolution2D(i+8, j, input, horiz_operator);
            gradient_y = convolution2D(i+8, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+8)*SIZE + j] = 255;} else{output[(i+8)*SIZE + j] = sqrt_lookup_table[p];}

            gradient_x = convolution2D(i+9, j, input, horiz_operator);
            gradient_y = convolution2D(i+9, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+9)*SIZE + j] = 255;} else{output[(i+9)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+10
            gradient_x = convolution2D(i+10, j, input, horiz_operator);
            gradient_y = convolution2D(i+10, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+10)*SIZE + j] = 255;} else{output[(i+10)*SIZE + j] = sqrt_lookup_table[p];}

            gradient_x = convolution2D(i+11, j, input, horiz_operator);
            gradient_y = convolution2D(i+11, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+11)*SIZE + j] = 255;} else{output[(i+11)*SIZE + j] = sqrt_lookup_table[p];}
            
            gradient_x = convolution2D(i+12, j, input, horiz_operator);
            gradient_y = convolution2D(i+12, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+12)*SIZE + j] = 255;} else{output[(i+12)*SIZE + j] = sqrt_lookup_table[p];}

            gradient_x = convolution2D(i+13, j, input, horiz_operator);
            gradient_y = convolution2D(i+13, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+13)*SIZE + j] = 255;} else{output[(i+13)*SIZE + j] = sqrt_lookup_table[p];}

            gradient_x = convolution2D(i+14, j, input, horiz_operator);
            gradient_y = convolution2D(i+14, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+14)*SIZE + j] = 255;} else{output[(i+14)*SIZE + j] = sqrt_lookup_table[p];}

            // Pixel i+15
            gradient_x = convolution2D(i+15, j, input, horiz_operator);
            gradient_y = convolution2D(i+15, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i+15)*SIZE + j] = 255;} else{output[(i+15)*SIZE + j] = sqrt_lookup_table[p];}
        }

        // In case last pixels are not mulitple of 16
        for(; i < SIZE - 1; i++){
            gradient_x = convolution2D(i, j, input, horiz_operator);
            gradient_y = convolution2D(i, j, input, vert_operator);
            p = (gradient_x * gradient_x) + (gradient_y * gradient_y);
            if(p > 65025){output[(i)*SIZE + j] = 255;} else{output[(i)*SIZE + j] = sqrt_lookup_table[p];}
        } 
    }

    /* Now run through the output and the golden output to calculate *
     * the MSE and then the PSNR.                                    */
    for (i=1; i<SIZE-1; i++) {
        for ( j=1; j<SIZE-1; j++ ) {
            t = pow((output[i*SIZE+j] - golden[i*SIZE+j]),2);
            PSNR += t;
        }
    }
  
    PSNR /= (double)(SIZE*SIZE);
    PSNR = 10*log10(65536/PSNR);

    /* This is the end of the main computation. Take the end time,  *
     * calculate the duration of the computation and report it.     */
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    /* MODIFICATION: Check if silent_mode is enabled.
     * If so, print only the number for the time.
     * Otherwise, print the full message. */
    if (silent_mode) {
        printf("%g\n",
            (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
            (double) (tv2.tv_sec - tv1.tv_sec));
    } else {
        printf ("Total time = %10g seconds\n",
            (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
            (double) (tv2.tv_sec - tv1.tv_sec));
    }
  
    /* Write the output file */
    fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
    fclose(f_out);
  
    return PSNR;
}


int main(int argc, char* argv[])
{
    double PSNR;
    
    // MODIFICATION: Check for the '-s' flag for "silent" mode.
    int silent_mode = 0;
    if (argc > 1 && strcmp(argv[1], "-s") == 0) {
        silent_mode = 1;
    }

    PSNR = sobel(input, output, golden, silent_mode);

    // MODIFICATION: Only print these messages if not in silent mode.
    if (!silent_mode) {
        printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
        printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");
    }

    return 0;
}