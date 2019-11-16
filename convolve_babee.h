#ifndef CONVOLVE_BABEE
#define CONVOLVE_BABEE

#include <stdio.h>

typedef struct tensor_2d {
  int ** tensor, width, height;
} tensor_2d;

typedef struct tensor_3d {
  int *** tensor;
  int width, height, depth;
} tensor_3d;

typedef struct maxpool_dims_3d{
  int width, height, depth, stride;
} maxpool_dims_3d;

typedef struct maxpool_dims_2d {
  int width, height, stride;
} maxpool_dims_2d;

#endif
