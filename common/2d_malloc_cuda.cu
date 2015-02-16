/* 
 * File:   2d_malloc_cuda.c
 * Author: John
 *
 * Created on February 13, 2015, 10:19 PM
 */

#include <stdio.h>
#include <stdlib.h>

#include <stdbool.h>

#include "2d_cuda_malloc.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void k_assign_uchar_ptrs(unsigned char **array, unsigned char *p, int rows, int columns, int channels)
{
    int i;
    for (i = 0; i < rows; i++)
        array[i] = &(p[i * columns * channels]);
}

__global__ void k_assign_float_ptrs(float **array, float *p, int rows, int columns, int channels)
{
    int i;
    for (i = 0; i < rows; i++)
        array[i] = &(p[i * columns * channels]);
}

extern "C" bool alloc_uchar_array_cuda(unsigned char ***array_d, unsigned char **p_d, int rows, int columns, int channels)
{
    unsigned char *p;
    cudaMalloc((void **) &p, rows * columns * channels * sizeof (unsigned char));
    if (p == NULL)
    {
        fprintf(stderr, "cudaMalloc: could not allocate device memory (1)\n");
        return false;
    }

    *p_d = p;

    unsigned char **array_p;
    cudaMalloc((void **) &array_p, rows * sizeof (unsigned char *));
    if (array_p == NULL)
    {
        fprintf(stderr, "cudaMalloc: could not allocate device memory (2)\n");
        cudaFree(p);
        return false;
    }

    dim3 dimBl(1);
    dim3 dimGr(1);
    k_assign_uchar_ptrs<<<dimGr, dimBl>>>(array_p, p, rows, columns, channels);

    *array_d = array_p;

    return true;
}

extern "C" bool alloc_float_array_cuda(float ***array_d, float **p_d, int rows, int columns, int channels)
{
    float *p;
    cudaMalloc((void **) &p, rows * columns * channels * sizeof (float));
    if (p == NULL)
    {
        fprintf(stderr, "cudaMalloc: could not allocate device memory (1)\n");
        return false;
    }

    *p_d = p;

    float **array_p;
    cudaMalloc((void **) &array_p, rows * sizeof (float *));
    if (array_p == NULL)
    {
        fprintf(stderr, "cudaMalloc: could not allocate device memory (2)\n");
        cudaFree(p);
        return false;
    }

    dim3 dimBl(1);
    dim3 dimGr(1);
    k_assign_float_ptrs<<<dimGr, dimBl>>>(array_p, p, rows, columns, channels);

    *array_d = array_p;

    return true;
}

extern "C" void dealloc_uchar_array_cuda(unsigned char ***array_d, unsigned char **p_d)
{
    cudaFree(p_d);
    p_d = NULL;
    cudaFree(*array_d);
    *array_d = NULL;
}

extern "C" void dealloc_float_array_cuda(float ***array_d, float **p_d)
{
    cudaFree(p_d);
    p_d = NULL;
    cudaFree(*array_d);
    *array_d = NULL;
}
