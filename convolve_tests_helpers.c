#ifndef CONVOLVE_TESTS_HELPERS
#define CONVOLVE_TESTS_HELPERS

#include <stdlib.h>
#include <stdbool.h>

#define new_array(length, elem_size)    malloc(length * elem_size)

// ---------------------2D--------------------- //

bool equals_2d(tensor_2d * a, tensor_2d * b) {
  if (a->width != b->width || a->height != b->height)
    return false;

  for (int i = 0; i < a->width; i++)
    for (int j = 0; j < a->height; j++)
      if (a->tensor[i][j] != b->tensor[i][j])
        return false;
  return true;
}

tensor_2d * new_2d_rand_tensor(int w, int h) {
  tensor_2d * t = (tensor_2d *)malloc(sizeof(tensor_2d));
  t->tensor = (int **)new_array(w, sizeof(int*));
  t->width = w;
  t->height = h;
  for (int i = 0; i < w; i++) {
    t->tensor[i] = (int *)new_array(h, sizeof(int));
    for (int j = 0; j < h; j++)
      t->tensor[i][j] = rand();
  }
  return t;
}

tensor_2d * new_2d_value_tensor(int w, int h, int value) {
  tensor_2d * t = (tensor_2d *)malloc(sizeof(tensor_2d));
  t->tensor = (int **)new_array(w, sizeof(int*));
  t->width = w;
  t->height = h;
  for (int i = 0; i < w; i++) {
    t->tensor[i] = (int *)new_array(h, sizeof(int));
    for (int j = 0; j < h; j++)
      t->tensor[i][j] = value;
  }
  return t;
}

tensor_2d * new_2d_incremental_tensor(int w, int h) {
  tensor_2d * t = (tensor_2d *)malloc(sizeof(tensor_2d));
  t->tensor = (int **)new_array(w, sizeof(int*));
  t->width = w;
  t->height = h;
  for (int i = 0; i < w; i++) {
    t->tensor[i] = (int *)new_array(h, sizeof(int));
    for (int j = 0; j < h; j++)
      t->tensor[i][j] = i*h + j;
  }
  return t;
}

void free_2d_tensor(tensor_2d * t) {
  for (int i = 0; i < t->width; i++)
      free(t->tensor[i]);
  free(t->tensor);
  free(t);
}

// ---------------------3D--------------------- //

bool equals_3d(tensor_3d * a, tensor_3d * b) {
  if (a->width != b->width || a->height != b->height || a->depth != b->depth)
    return false;

  for (int i = 0; i < a->width; i++)
    for (int j = 0; j < a->height; j++)
      for (int k = 0; k < a->depth; k++)
        if (a->tensor[i][j][k] != b->tensor[i][j][k])
          return false;
  return true;
}

// returns 3d tensor each including 2d tensors values
// are increasing within it. all including 2d tensors are same.
tensor_3d * new_3d_incremental_tensor(int w, int h, int d) {
  tensor_3d * t = (tensor_3d *)malloc(sizeof(tensor_3d));
  t->width = w;
  t->height = h;
  t->depth = d;
  t->tensor = (int ***)new_array(w, sizeof(int**));
  for (int i = 0; i < w; i++) {
    t->tensor[i] = (int **)new_array(h, sizeof(int*));
    for (int j = 0; j < h; j++) {
      t->tensor[i][j] = (int *)new_array(d, sizeof(int));
      for (int k = 0; k < d; k++)
        t->tensor[i][j][k] = j*d + k;
    }
  }
  return t;
}

tensor_3d * new_3d_value_tensor(int w, int h, int d, int value) {
  tensor_3d * t = (tensor_3d *)malloc(sizeof(tensor_3d));
  t->width = w;
  t->height = h;
  t->depth = d;
  t->tensor = (int ***)new_array(w, sizeof(int**));
  for (int i = 0; i < w; i++) {
    t->tensor[i] = (int **)new_array(h, sizeof(int*));
    for (int j = 0; j < h; j++) {
      t->tensor[i][j] = (int *)new_array(d, sizeof(int));
      for (int k = 0; k < d; k++)
        t->tensor[i][j][k] = value;
    }
  }
  return t;
}

void free_3d_tensor(tensor_3d * t) {
  for (int i = 0; i < t->width; i++) {
    for (int j = 0; j < t->height; j++)
      free(t->tensor[i][j]);
    free(t->tensor[i]);
  }
  free(t->tensor);
  free(t);
}

#endif
