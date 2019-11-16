#include "convolve_babee.h"
#include "convolve_tests_helpers.c"

#define max(a, b) ( a > b ? a : b )

// ALL FUNCTIONS RETURN DUMB VALUE REMOVE IT.

tensor_3d * maxpool_3d(tensor_3d * tensor, maxpool_dims_3d * dims) {
  return new_3d_value_tensor(1,1,1,1);
}

tensor_3d * convolution_3d(tensor_3d * tensor, tensor_3d * convolution_tensor) {
  return new_3d_value_tensor(1,1,1,1);;
}

tensor_2d * maxpool_2d(tensor_2d * tensor,  maxpool_dims_2d * dims) {
  return new_2d_value_tensor(1,1,1);;
}

tensor_2d * convolution_2d(tensor_2d * tensor, tensor_2d * convolution_tensor) {
  return new_2d_value_tensor(1,1,1);;
}
