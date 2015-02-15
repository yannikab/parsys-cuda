/* 
 * File:   main.c
 * Author: jester
 *
 * Created on February 13, 2015, 10:22 PM
 */

#include <stdio.h>
#include <stdlib.h>

// #include "common/2d_cuda_malloc.h"

// int main_add(int argc, char** argv);
int main_cuda(int argc, char** argv);

/*
 * 
 */
int main(int argc, char** argv)
{
    // unsigned char **buffer;
    // alloc_uchar_array_cuda(&buffer, 100, 100, 3);

    // return main_add(argc, argv);
    return main_cuda(argc, argv);
}
