/* 
 * File:   2d_cuda_malloc.h
 * Author: John
 *
 * Created on February 13, 2015, 10:18 PM
 */

#ifndef TWOD_CUDA_MALLOC_H
#define	TWOD_CUDA_MALLOC_H

#include <stdbool.h>

extern "C" {
    bool alloc_uchar_array_cuda(unsigned char ***array_d, unsigned char **p_d, int rows, int columns, int channels);
    void dealloc_uchar_array_cuda(unsigned char ***array);
    bool alloc_float_array_cuda(float ***array_d, float **p_d, int rows, int columns, int channels);
    void dealloc_float_array_cuda(float ***array);
}

#endif	/* TWOD_CUDA_MALLOC_H */
