#include "filter.h"

#include "../settings.h"
#include "2d_cuda_malloc.h"

#include <stdio.h>
#include <vector_types.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void k_fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width)
{
    unsigned int i, j, c;
    /* Fill borders with outer image data. */

    // south
    for (i = height + B; i < height + 2 * B; i++)
        for (j = B; j < B + width; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[B + height - 1][j][c];

    // north
    for (i = 0; i < B; i++) // north
        for (j = B; j < B + width; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[B][j][c];

    // east
    for (i = B; i < B + height; i++)
        for (j = width + B; j < width + 2 * B; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[i][B + width - 1][c];

    // west
    for (i = B; i < B + height; i++)
        for (j = 0; j < B; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[i][B][c];

    // se
    for (i = height + B; i < height + 2 * B; i++)
        for (j = width + B; j < width + 2 * B; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[B + height - 1][B + width - 1][c];

    // nw
    for (i = 0; i < B; i++)
        for (j = 0; j < B; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[B][B][c];

    // sw
    for (i = height + B; i < height + 2 * B; i++)
        for (j = 0; j < B; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[B + height - 1][B][c];

    // ne
    for (i = 0; i < B; i++)
        for (j = width + B; j < width + 2 * B; j++)
            for (c = 0; c < CHANNELS; c++)
                curr_image_d[i][j][c] = curr_image_d[B][B + width - 1][c];
}

__global__ void k_apply_inner_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], unsigned int height, unsigned int width, float (**filter_d)[1])
{
    unsigned int i, j, c;
    int p, q;

    for (i = 2 * B; i < height - 2 * B; i++)
    {
        for (j = 2 * B; j < width - 2 * B; j++)
        {
            for (c = 0; c < CHANNELS; c++)
            {
                float value = 0.0f;

                for (p = -B; p <= B; p++)
                    for (q = -B; q <= B; q++)
                        value += input_image[i - p][j - q][c] * filter_d[p + B][q + B][0];

                output_image[i][j][c] = value;
            }
        }
    }
}

__global__ void k_apply_outer_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], unsigned int height, unsigned int width, float (**filter_d)[1])
{
    unsigned int i, j, c;
    int p, q;

    /* Left border column. */

    for (i = B; i < height - B; i++)
    {
        for (j = B; j < 2 * B; j++)
        {
            for (c = 0; c < CHANNELS; c++)
            {
                float value = 0.0f;

                for (p = -B; p <= B; p++)
                    for (q = -B; q <= B; q++)
                        value += input_image[i - p][j - q][c] * filter_d[p + B][q + B][0];

                output_image[i][j][c] = value;
            }
        }
    }

    /* Right border column. */

    for (i = B; i < height - B; i++)
    {
        for (j = width - 1 - B; j > width - 1 - 2 * B; j--)
        {
            for (c = 0; c < CHANNELS; c++)
            {
                float value = 0.0f;

                for (p = -B; p <= B; p++)
                    for (q = -B; q <= B; q++)
                        value += input_image[i - p][j - q][c] * filter_d[p + B][q + B][0];

                output_image[i][j][c] = value;
            }
        }
    }

    /* Top border row, avoid recalculating overlap with columns. */

    for (j = 2 * B; j < width - 2 * B; j++)
    {
        for (i = B; i < 2 * B; i++)
        {
            for (c = 0; c < CHANNELS; c++)
            {
                float value = 0.0f;

                for (p = -B; p <= B; p++)
                    for (q = -B; q <= B; q++)
                        value += input_image[i - p][j - q][c] * filter_d[p + B][q + B][0];

                output_image[i][j][c] = value;
            }
        }
    }

    /* Bottom border row, avoid recalculating overlap with columns. */

    for (j = 2 * B; j < width - 2 * B; j++)
    {
        for (i = height - 1 - B; i > height - 1 - 2 * B; i--)
        {
            for (c = 0; c < CHANNELS; c++)
            {
                float value = 0.0f;

                for (p = -B; p <= B; p++)
                    for (q = -B; q <= B; q++)
                        value += input_image[i - p][j - q][c] * filter_d[p + B][q + B][0];

                output_image[i][j][c] = value;
            }
        }
    }
}

extern "C" void apply_inner_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], unsigned int height, unsigned int width, float (**filter_d)[1])
{
    dim3 dimBl(1);
    dim3 dimGr(1);
    k_apply_inner_filter_cuda<<<dimGr, dimBl>>>(output_image, input_image, height, width, filter_d);
}

extern "C" void apply_outer_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], unsigned int height, unsigned int width, float (**filter_d)[1])
{
    dim3 dimBl(1);
    dim3 dimGr(1);
    k_apply_outer_filter_cuda<<<dimGr, dimBl>>>(output_image, input_image, height, width, filter_d);
}

extern "C" void fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width)
{
    dim3 dimBl(1);
    dim3 dimGr(1);
    k_fill_borders<<<dimGr, dimBl>>>(curr_image_d, height, width);
}

extern "C" bool init_filter(float (***filter_d)[1], const float filter[2 * B + 1][2 * B + 1])
{
    float *p;
    float (**filter_p)[1];

    if (!alloc_float_array_cuda((float ***) &filter_p, &p, 2 * B + 1, 2 * B + 1, 1))
        return false;

    cudaMemcpy(p, filter, (2 * B + 1) * (2 * B + 1) * sizeof (float), cudaMemcpyHostToDevice);

    *filter_d = filter_p;

    return true;
}
