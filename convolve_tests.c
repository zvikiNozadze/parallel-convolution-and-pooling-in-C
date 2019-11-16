#include "convolve_babee.c"

bool test_dims_no_operation_conv_2d() {
  tensor_2d * tensor = new_2d_rand_tensor(5, 5);
  tensor_2d * conv_tensor = new_2d_value_tensor(1, 1, 1);
  tensor_2d * conv_output = convolution_2d(tensor, conv_tensor);

  bool is_correct = equals_2d(conv_output, tensor);

  free_2d_tensor(tensor);
  free_2d_tensor(conv_tensor);
  free_2d_tensor(conv_output);

  return is_correct;
}

bool test_operation_conv_2d() {
  tensor_2d * tensor = new_2d_value_tensor(5, 5, 1);
  tensor_2d * conv_tensor = new_2d_value_tensor(2, 2, 1);
  tensor_2d * correct_output = new_2d_value_tensor(4, 4, 4);
  tensor_2d * conv_output = convolution_2d(tensor, conv_tensor);

  bool is_correct = equals_2d(conv_output, correct_output);

  free_2d_tensor(tensor);
  free_2d_tensor(conv_tensor);
  free_2d_tensor(correct_output);
  free_2d_tensor(conv_output);

  return is_correct;
}

bool test_nonsquare_conv_2d() {
  tensor_2d * tensor = new_2d_value_tensor(10, 5, 3);
  tensor_2d * conv_tensor = new_2d_value_tensor(3, 4, 2);
  tensor_2d * correct_output = new_2d_value_tensor(8, 2, 3*4*6);
  tensor_2d * conv_output = convolution_2d(tensor, conv_tensor);

  bool is_correct = equals_2d(conv_output, correct_output);

  free_2d_tensor(tensor);
  free_2d_tensor(conv_tensor);
  free_2d_tensor(correct_output);
  free_2d_tensor(conv_output);

  return is_correct;
}

bool test_edge_dims_negs_conv_2d() {
  tensor_2d * tensor = new_2d_value_tensor(19, 2, 13);
  tensor_2d * conv_tensor = new_2d_value_tensor(1, 2, -2);
  tensor_2d * correct_output = new_2d_value_tensor(19, 1, -52);
  tensor_2d * conv_output = convolution_2d(tensor, conv_tensor);

  bool is_correct = equals_2d(conv_output, correct_output);

  free_2d_tensor(tensor);
  free_2d_tensor(conv_tensor);
  free_2d_tensor(correct_output);
  free_2d_tensor(conv_output);

  return is_correct;
}

bool test_dims_no_stride_maxpool_2d() {
  tensor_2d * tensor = new_2d_rand_tensor(50, 50);
  maxpool_dims_2d * dims = (maxpool_dims_2d *)malloc(sizeof(maxpool_dims_2d));
  dims->width = dims->height = dims->stride = 1;
  tensor_2d * maxpool_output = maxpool_2d(tensor, dims);

  bool is_correct = equals_2d(maxpool_output, tensor);

  free_2d_tensor(tensor);
  free_2d_tensor(maxpool_output);
  free(dims);
  return is_correct;
}

bool test_dims_stride_maxpool_2d() {
  tensor_2d * tensor = new_2d_value_tensor(4, 4, 3);
  maxpool_dims_2d * dims = (maxpool_dims_2d *)malloc(sizeof(maxpool_dims_2d));
  dims->width = dims->height = dims->stride = 2;
  tensor_2d * correct_output = new_2d_value_tensor(2, 2, 3);
  tensor_2d * maxpool_output = maxpool_2d(tensor, dims);

  bool is_correct = equals_2d(maxpool_output, correct_output);

  free_2d_tensor(tensor);
  free_2d_tensor(correct_output);
  free_2d_tensor(maxpool_output);
  free(dims);
  return is_correct;
}

bool test_nonsquare_computational_maxpool_2d() {
  tensor_2d * tensor = new_2d_incremental_tensor(11, 8);

  maxpool_dims_2d * dims = (maxpool_dims_2d *)malloc(sizeof(maxpool_dims_2d));

  dims->width = 5;
  dims->height = 2;
  dims->stride = 3;

  tensor_2d * correct_output = new_2d_value_tensor(3, 3, 0);
  correct_output->tensor[0][0] = 33;
  correct_output->tensor[1][0] = 57;
  correct_output->tensor[2][0] = 81;
  for (int i = 0; i < 3; i++)
    for (int j = 1; j < 3; j++)
      correct_output->tensor[i][j] = correct_output->tensor[i][j-1] + 3;

  tensor_2d * maxpool_output = maxpool_2d(tensor, dims);

  bool is_correct = equals_2d(maxpool_output, correct_output);

  free_2d_tensor(tensor);
  free_2d_tensor(correct_output);
  free_2d_tensor(maxpool_output);
  free(dims);
  return is_correct;
}

bool test_dims_simple_conv_3d() {
  tensor_3d * tensor = new_3d_incremental_tensor(4, 4, 4);
  tensor_3d * conv_tensor = new_3d_value_tensor(1, 1, 1, 1);
  tensor_3d * conv_output = convolution_3d(tensor, conv_tensor);

  bool is_correct = equals_3d(conv_output, tensor);

  free_3d_tensor(tensor);
  free_3d_tensor(conv_tensor);
  free_3d_tensor(conv_output);

  return is_correct;
}

bool test_cubic_conv_3d() {
  tensor_3d * tensor = new_3d_value_tensor(40, 40, 40, 3);
  tensor_3d * conv_tensor = new_3d_value_tensor(2, 2, 2, -2);
  tensor_3d * correct_output = new_3d_value_tensor(49, 49, 49, -48);
  tensor_3d * conv_output = convolution_3d(tensor, conv_tensor);

  bool is_correct = equals_3d(conv_output, correct_output);

  free_3d_tensor(tensor);
  free_3d_tensor(conv_tensor);
  free_3d_tensor(correct_output);
  free_3d_tensor(conv_output);

  return is_correct;
}

bool test_noncubuc_conv_3d() {
  tensor_3d * tensor = new_3d_value_tensor(40, 30, 20, 10);
  tensor_3d * conv_tensor = new_3d_value_tensor(2, 3, 4, 5);
  tensor_3d * correct_output = new_3d_value_tensor(39, 28, 17, 10*2*3*4*5);
  tensor_3d * conv_output = convolution_3d(tensor, conv_tensor);

  bool is_correct = equals_3d(conv_output, correct_output);

  free_3d_tensor(tensor);
  free_3d_tensor(conv_tensor);
  free_3d_tensor(correct_output);
  free_3d_tensor(conv_output);

  return is_correct;
}

bool test_dims_simple_stride_maxpool_3d() {
  maxpool_dims_3d * dims = (maxpool_dims_3d *)malloc(sizeof(maxpool_dims_3d));
  dims->width = dims->height = dims->depth = dims->stride = 2;
  tensor_3d * tensor = new_3d_value_tensor(4, 4, 2, 1);
  tensor_3d * correct_output = new_3d_value_tensor(2, 2, 1, 1);
  tensor_3d * maxpool_output = maxpool_3d(tensor, dims);

  bool is_correct = equals_3d(maxpool_output, correct_output);

  free_3d_tensor(tensor);
  free_3d_tensor(correct_output);
  free_3d_tensor(maxpool_output);
  free(dims);
  return is_correct;
}

bool test_noncubuc_stride_maxpool_3d() {
  maxpool_dims_3d * dims = (maxpool_dims_3d *)malloc(sizeof(maxpool_dims_3d));
  dims->width = 2;
  dims->height = 4;
  dims->depth = 2;
  dims->stride = 2;

  tensor_3d * tensor = new_3d_incremental_tensor(16, 8, 4);
  tensor_3d * correct_output = new_3d_value_tensor(8, 3, 2, 2);

  // ????? HERE I M TAKING ADVANTAGE OF KNOLEDGE
  // EVERY MATRIX LAYER ON WIDTH AXIS ARE THE SAME
  // AND INCREMENTAL SO BOTTOM RIGHTMOST ELEMENT WILL BE
  // ANSWER ON EACH OUTPUT CELL. AND AVERY LAYER OF OUTPUT MATRIX
  // ON AXIS WIDTH ARE GOING TO BE SAME.
  for (int i = 0; i < 8; i++) {
    int value = 13;
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 2; k++) {
        correct_output->tensor[i][j][k] = value;
        value += 2;
      }
      value += 4;
    }
  }

  tensor_3d * maxpool_output = maxpool_3d(tensor, dims);

  bool is_correct = equals_3d(maxpool_output, correct_output);

  free_3d_tensor(tensor);
  free_3d_tensor(correct_output);
  free_3d_tensor(maxpool_output);
  free(dims);
  return is_correct;
}

void test_maxpool_3d() {
  bool test_results[2] = {
    test_dims_simple_stride_maxpool_3d(),
    test_noncubuc_stride_maxpool_3d()
  };

  printf("=====test_maxpool_3d=====\n");
  for (int i = 0; i < 2; i++)
    if (test_results[i])  printf("%d PASS\n", i);
    else                  printf("%d FAIL\n", i);
}

void test_convolution_3d() {
  bool test_results[3] = {
    test_dims_simple_conv_3d(),
    test_cubic_conv_3d(),
    test_noncubuc_conv_3d()
  };

  printf("===test_convolution_3d===\n");
  for (int i = 0; i < 3; i++)
    if (test_results[i])  printf("%d PASS\n", i);
    else                  printf("%d FAIL\n", i);

}

void test_maxpool_2d() {
  bool test_results[3] = {
    test_dims_no_stride_maxpool_2d(),
    test_dims_stride_maxpool_2d(),
    test_nonsquare_computational_maxpool_2d()
  };

  printf("=====test_maxpool_2d=====\n");
  for (int i = 0; i < 3; i++)
    if (test_results[i])  printf("%d PASS\n", i);
    else                  printf("%d FAIL\n", i);

}

void test_convolution_2d() {
  bool test_results[4] = {
    test_dims_no_operation_conv_2d(),
    test_operation_conv_2d(),
    test_nonsquare_conv_2d(),
    test_edge_dims_negs_conv_2d()
  };

  printf("===test_convolution_2d===\n");
  for (int i = 0; i < 4; i++)
    if (test_results[i])  printf("%d PASS\n", i);
    else                  printf("%d FAIL\n", i);

}

int main(int argc, char const *argv[]) {
  test_convolution_2d();
  test_maxpool_2d();
  test_convolution_3d();
  test_maxpool_3d();
  return 0;
}
