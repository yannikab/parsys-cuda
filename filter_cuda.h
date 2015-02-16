/* 
 * File:   filter_cuda.h
 * Author: John
 *
 * Created on February 16, 2015, 1:55 AM
 */

#ifndef FILTER_CUDA_H
#define	FILTER_CUDA_H

#include <stdbool.h>

#include "settings.h"

#define B 1

static const float filter[2 * B + 1][2 * B + 1] = {
    {0.0625f, 0.125f, 0.0625f},
    {0.1250f, 0.250f, 0.1250f},
    {0.0625f, 0.125f, 0.0625f},
};

extern "C" {
    bool init_filter(float (***filter_d)[1], float **p_d, const float filter[2 * B + 1][2 * B + 1]);
    void destroy_filter(float (***filter_d)[1], float **p_d);
    void fill_borders(float (**curr_image_d)[CHANNELS], unsigned int height, unsigned int width);
    void apply_filter_cuda(float (**output_image)[CHANNELS], float (**input_image)[CHANNELS], float (**filter_d)[1], unsigned int block_size, unsigned int grid_dim);
}

#endif	/* FILTER_CUDA_H */
