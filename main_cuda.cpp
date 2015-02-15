/* 
 * File:   main_serial.c
 * Author: John
 *
 * Created on January 21, 2015, 11:22 AM
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
#include "common/filter.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern "C" bool init_filter(float (***filter_d)[1], const float filter[2 * B + 1][2 * B + 1]);
extern "C" void fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width);
extern "C" void apply_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], float (**filter_d)[1], unsigned int block_size, unsigned int grid_dim);

/*
 * 
 */
int main_cuda(int argc, char** argv)
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
        ok = alloc_float_array((float ***) &image_h, B + HEIGHT + B, B + WIDTH + B, CHANNELS);

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
        ok = alloc_float_array_cuda((float ***) &prev_image_d, &prev_image_p, B + HEIGHT + B, B + WIDTH + B, CHANNELS);

    if (ok)
        ok = alloc_float_array_cuda((float ***) &curr_image_d, &curr_image_p, B + HEIGHT + B, B + WIDTH + B, CHANNELS);

    /* Initialize filter in device memory space. */

    float (**filter_d)[1];

    if (ok)
        ok = init_filter(&filter_d, filter);

    /* Device parameters for nVidia 9600GT (G94), passed to main filter function. */

    /* nVidia G94 supports 8 resident blocks per SMP, 768 resident threads per SMP. */

    unsigned int block_size = 256; // 512 threads per block maximum for nVidia G94

    /* nVidia G94 supports 2-dimensional grids with a maximum of 65535 for x,y dimension. */

    unsigned int threads_required = HEIGHT * WIDTH;
    unsigned int grid_dim = threads_required / block_size;
    double sqr = sqrt(grid_dim);
    grid_dim = sqr;
    grid_dim++;
    printf("Grid dim: %u\n", grid_dim);

    /* Start timing. */

    float memcopy, compute;
    timestamp t_start;
    t_start = getTimestamp();

    /* Copy image data to device. */

    cudaMemcpy(curr_image_p, &(image_h[0][0][0]), (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float), cudaMemcpyHostToDevice);

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

    cudaMemcpy(&(image_h[0][0][0]), curr_image_p, (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float), cudaMemcpyDeviceToHost);

    memcopy += getElapsedtime(t_start);

    printf("Completed in %.3f sec\n", compute / 1000);
    printf("Memory copy in %.3f sec\n", memcopy / 1000);

    //    /* Wide buffer including borders. */
    //
    //    unsigned char (**image_buffer_w)[CHANNELS];
    //
    //    if (ok)
    //        ok = alloc_uchar_array((unsigned char ***) &image_buffer_w, B + HEIGHT + B, B + WIDTH + B, CHANNELS);
    //
    //    /* Convert output. */
    //
    //    if (ok)
    //    {
    //        for (i = 0; i < B + HEIGHT + B; i++)
    //            for (j = 0; j < B + WIDTH + B; j++)
    //                for (c = 0; c < CHANNELS; c++)
    //                    image_buffer_w[i][j][c] = (unsigned char) image_h[i][j][c];
    //    }

    /* Create output files, one for each channel. */

    // write_channels(image_buffer_w, B + HEIGHT + B, B + WIDTH + B);

    /* Convert output. */

    if (ok)
    {
        for (i = 0; i < HEIGHT; i++)
            for (j = 0; j < WIDTH; j++)
                for (c = 0; c < CHANNELS; c++)
                    image_buffer_h[i][j][c] = (unsigned char) image_h[i + B][j + B][c];
    }

    /* Create output files, one for each channel. */

    write_channels(image_buffer_h, HEIGHT, WIDTH);

    /* Free allocated memory. */

    dealloc_uchar_array((unsigned char ***) &image_buffer_h);
    //    dealloc_uchar_array_cuda((unsigned char ***) &image_buffer_d);
    //    dealloc_float_array_cuda((float ***) &curr_image);
    //    dealloc_float_array_cuda((float ***) &prev_image);

    return (EXIT_SUCCESS);
}
