#include "filter_cuda.h"

#include "settings.h"

#include "2d_cuda_malloc.h"

#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

extern "C" bool init_filter(float (***filter_d)[1], float **p_d, const float filter[2 * B + 1][2 * B + 1])
{
    float *p;
    float (**filter_p)[1];

    if (!alloc_float_array_cuda((float ***) &filter_p, &p, 2 * B + 1, 2 * B + 1, 1))
        return false;

    cudaMemcpy(p, filter, (2 * B + 1) * (2 * B + 1) * sizeof (float), cudaMemcpyHostToDevice);

    *filter_d = filter_p;
    *p_d = p;

    return true;
}

extern "C" void destroy_filter(float (***filter_d)[1], float **p_d)
{
    dealloc_float_array_cuda((float ***) filter_d, p_d);
}

__global__ void k_fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width)
{
    unsigned int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    if (threadId > 7)
        return;

    unsigned int i, j, c;

    /* Fill borders with outer image data. */

    // south
    if (threadId == 0)
        for (i = height + B; i < height + 2 * B; i++)
            for (j = B; j < B + width; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[B + height - 1][j][c];

    // north
    if (threadId == 1)
        for (i = 0; i < B; i++) // north
            for (j = B; j < B + width; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[B][j][c];

    // east
    if (threadId == 2)
        for (i = B; i < B + height; i++)
            for (j = width + B; j < width + 2 * B; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[i][B + width - 1][c];

    // west
    if (threadId == 3)
        for (i = B; i < B + height; i++)
            for (j = 0; j < B; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[i][B][c];

    // se
    if (threadId == 4)
        for (i = height + B; i < height + 2 * B; i++)
            for (j = width + B; j < width + 2 * B; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[B + height - 1][B + width - 1][c];

    // nw
    if (threadId == 5)
        for (i = 0; i < B; i++)
            for (j = 0; j < B; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[B][B][c];

    // sw
    if (threadId == 6)
        for (i = height + B; i < height + 2 * B; i++)
            for (j = 0; j < B; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[B + height - 1][B][c];

    // ne
    if (threadId == 7)
        for (i = 0; i < B; i++)
            for (j = width + B; j < width + 2 * B; j++)
                for (c = 0; c < CHANNELS; c++)
                    curr_image_d[i][j][c] = curr_image_d[B][B + width - 1][c];
}

__global__ void k_apply_filter_cuda(float (**output_image_d)[CHANNELS], float (**input_image_d)[CHANNELS], float (**filter_d)[1])
{
    unsigned int threadsPerBlock = blockDim.x * blockDim.y;
    unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int globalId = blockId * threadsPerBlock + threadId;

    unsigned int i, j, c;
    int p, q;

    i = globalId / WIDTH;
    j = globalId % WIDTH;

    if (i > HEIGHT - 1)
        return;

    i += B;
    j += B;

    for (c = 0; c < CHANNELS; c++)
    {
        float value = 0.0f;

        for (p = -B; p <= B; p++)
            for (q = -B; q <= B; q++)
                value += input_image_d[i - p][j - q][c] * filter_d[p + B][q + B][0];

        output_image_d[i][j][c] = value;
    }
}

extern "C" void fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width)
{
    dim3 dimBl(8);
    dim3 dimGr(1);
    
    k_fill_borders<<<dimGr, dimBl>>>(curr_image_d, height, width);
}

extern "C" void apply_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], float (**filter_d)[1], unsigned int block_size, unsigned int grid_dim)
{
    dim3 dimBl(block_size);
    dim3 dimGr(grid_dim, grid_dim);

    k_apply_filter_cuda<<<dimGr, dimBl>>>(output_image, input_image, filter_d);
}
