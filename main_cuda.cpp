/* 
 * File:   main_cuda.c
 * Author: jester
 *
 * Created on February 13, 2015, 10:22 PM
 */

#include <stdio.h>
#include <stdlib.h>

#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "timestamp.h"

#include "settings.h"
#include "common/2d_malloc.h"
#include "common/2d_cuda_malloc.h"
#include "common/file_io.h"
#include "common/filter_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/*
 * 
 */
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <iterations>\n", argv[0]);
        return (EXIT_FAILURE);
    }

    int iterations = atoi(argv[1]);

    printf("main_cuda()\n");
    printf("Iterations: %d\n", iterations);

    bool ok = true;

    /* Read input file into buffer. */

    unsigned char (**image_buffer_h)[CHANNELS];

    if (ok)
        ok = read_image((unsigned char ***) &image_buffer_h);

    /* Allocate memory for image data. */

    float (**image_h)[CHANNELS];

    if (ok)
        ok = alloc_float_array((float ***) &image_h,
            B + HEIGHT + B, B + WIDTH + B, CHANNELS);

    /* Convert input. */

    unsigned int i, j, c;

    if (ok)
    {
        for (i = 0; i < HEIGHT; i++)
            for (j = 0; j < WIDTH; j++)
                for (c = 0; c < CHANNELS; c++)
                    image_h[i + B][j + B][c] = (float) image_buffer_h[i][j][c];
    }

    /* Device memory allocation. */

    float (**prev_image_d)[CHANNELS];
    float (**curr_image_d)[CHANNELS];

    float *prev_image_p;
    float *curr_image_p;

    if (ok)
        ok = alloc_float_array_cuda((float ***) &prev_image_d, &prev_image_p,
            B + HEIGHT + B, B + WIDTH + B, CHANNELS);

    if (ok)
        ok = alloc_float_array_cuda((float ***) &curr_image_d, &curr_image_p,
            B + HEIGHT + B, B + WIDTH + B, CHANNELS);

    /* Initialize filter in device memory space. */

    float (**filter_d)[1];
    float *filter_p;

    if (ok)
        ok = init_filter(&filter_d, &filter_p, filter);

    /* Device parameters for nVidia 9600GT (G94), passed to main filter function. */

    /* nVidia G94 supports 8 resident blocks per SMP, 768 resident threads per SMP. */

    unsigned int block_size = 64; // maximum 512 threads per block for nVidia G94
    printf("Block size: %u\n", block_size);

    /* nVidia G94 supports 2-dimensional grids with a maximum of 65535 for x,y dimension. */

    unsigned int grid_dim = HEIGHT * WIDTH / block_size;
    double sqr = sqrt(grid_dim);
    grid_dim = sqr;
    grid_dim++;
    printf("Grid: %ux%u\n", grid_dim, grid_dim);

    /* Start timing. */

    float memcopy, compute;
    timestamp t_start;
    t_start = getTimestamp();

    /* Copy image data to device. */

    if (ok)
        ok = (cudaSuccess == cudaMemcpy(curr_image_p, &(image_h[0][0][0]),
            (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float),
            cudaMemcpyHostToDevice));

    memcopy = getElapsedtime(t_start);
    t_start = getTimestamp();

    /* Clear host image data. */

    // memset(&(image_h[0][0][0]), 0, (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float));

    /* Apply filter. */

    unsigned int n;

    if (ok)
    {
        for (n = 0; iterations == 0 || n < iterations; n++)
        {
            /* Fill borders with edge image data. */

            fill_borders(curr_image_d, HEIGHT, WIDTH);

            /* Apply filter. */

            apply_filter_cuda(prev_image_d, curr_image_d, filter_d, block_size, grid_dim);

            /* Switch current / previous image buffers. */

            float (**temp)[CHANNELS];
            temp = prev_image_d;
            prev_image_d = curr_image_d;
            curr_image_d = temp;

            float *tmp;
            tmp = prev_image_p;
            prev_image_p = curr_image_p;
            curr_image_p = tmp;
        }
    }

    /* Stop time measurement, print time. */

    cudaThreadSynchronize();

    compute = getElapsedtime(t_start);
    t_start = getTimestamp();

    /* Copy processed image data from device. */

    if (ok)
        ok = (cudaSuccess == cudaMemcpy(&(image_h[0][0][0]), curr_image_p,
            (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float),
            cudaMemcpyDeviceToHost));

    memcopy += getElapsedtime(t_start);

    printf("Completed in %.3f sec\n", compute / 1000);
    printf("Memory copy in %.3f sec\n", memcopy / 1000);

    /* Convert output. */

    if (ok)
    {
        for (i = 0; i < HEIGHT; i++)
            for (j = 0; j < WIDTH; j++)
                for (c = 0; c < CHANNELS; c++)
                    image_buffer_h[i][j][c] = (unsigned char) image_h[i + B][j + B][c];
    }

    /* Create output files, one for each channel. */

    if (ok)
        ok = write_channels(image_buffer_h, HEIGHT, WIDTH);

    /* Free allocated memory. */

    dealloc_uchar_array((unsigned char ***) &image_buffer_h);
    dealloc_float_array((float ***) &image_h);
    dealloc_float_array_cuda((float ***) &prev_image_d, &prev_image_p);
    dealloc_float_array_cuda((float ***) &curr_image_d, &curr_image_p);
    destroy_filter(&filter_d, &filter_p);

    return ok ? (EXIT_SUCCESS) : (EXIT_FAILURE);
}
