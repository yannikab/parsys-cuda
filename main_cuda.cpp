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

#include <unistd.h>
#include <sys/times.h>

#include "settings.h"
#include "common/2d_malloc.h"
#include "common/2d_cuda_malloc.h"
#include "common/file_io.h"
#include "common/filter.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern "C" void zero_pixels(float (**curr_image_d)[CHANNELS]);
extern "C" void fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width);
extern "C" void apply_inner_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], unsigned int height, unsigned int width, float (**filter_d)[1]);
extern "C" void apply_outer_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], unsigned int height, unsigned int width, float (**filter_d)[1]);
extern "C" bool init_filter(float (***filter_d)[1], const float filter[2 * B + 1][2 * B + 1]);

/*
 * 
 */
int main_cuda(int argc, char** argv)
{
    //    if (argc != 3)
    //    {
    //        printf("Usage: %s <iterations> <convergence>\n", argv[0]);
    //        return (EXIT_FAILURE);
    //    }
    //
    //    int iterations = atoi(argv[1]);
    //    int convergence = atoi(argv[2]);

    int iterations = 1;
    int convergence = 0;

    printf("main_cuda()\n");
    printf("Iterations: %d, Convergence: %d\n", iterations, convergence);
    printf("Threads: %d\n", 1);

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

    /* Copy image data to device. */

    cudaMemcpy(curr_image_p, &(image_h[0][0][0]), (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float), cudaMemcpyHostToDevice);

    /* Clear host image data. */

    memset(&(image_h[0][0][0]), 0, (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float));

    /* Start timing. */

    double t1, t2, real_time;
    struct tms tb1, tb2;
    double tickspersec = (double) sysconf(_SC_CLK_TCK);

    t1 = (double) times(&tb1);

    /* Test: Zero out all pixels. */

    // zero_pixels(curr_image_d);

    /* Initialize filter in device memory space. */

    float (**filter_d)[1];

    if (ok)
        ok = init_filter(&filter_d, filter);

    /* Apply filter. */

    unsigned int n;

    if (ok)
    {
        for (n = 0; iterations == 0 || n < iterations; n++)
        {
            /* Fill borders with edge image data. */

            fill_borders(curr_image_d, HEIGHT, WIDTH);

            /* Apply filter. */

            apply_inner_filter_cuda(prev_image_d, curr_image_d, B + HEIGHT + B, B + WIDTH + B, filter_d);

            apply_outer_filter_cuda(prev_image_d, curr_image_d, B + HEIGHT + B, B + WIDTH + B, filter_d);

            /* Switch current / previous image buffers. */

            float (**temp)[CHANNELS];
            temp = prev_image_d;
            prev_image_d = curr_image_d;
            curr_image_d = temp;

            float *tmp;
            tmp = prev_image_p;
            prev_image_p = curr_image_p;
            curr_image_p = tmp;

            /* Check for convergence. */

            //            if (convergence > 0 && n % convergence == 0)
            //            {
            //                if (images_identical(curr_image, prev_image, B + HEIGHT + B, B + WIDTH + B))
            //                {
            //                    printf("Filter has converged after %d iterations.\n", n);
            //                    break;
            //                }
            //            }
        }
    }

    /* Stop time measurement, print time. */

    t2 = (double) times(&tb2);

    real_time = (double) (t2 - t1) / tickspersec;
    printf("Completed in %.3f sec\n", real_time);

    /* Copy processed image data from device. */

    cudaMemcpy(&(image_h[0][0][0]), curr_image_p, (B + HEIGHT + B) * (B + WIDTH + B) * CHANNELS * sizeof (float), cudaMemcpyDeviceToHost);

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
