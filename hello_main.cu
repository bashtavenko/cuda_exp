#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_utils.h"

__global__ void dummy_kernel(void) {}

__device__ int add_device(int a, int b) { return a + b; }

__global__ void add(int a, int b, int* c) { *c = add_device(a, b); }

int main(void) {
  // Check for GPU
  int device_count;
  RETURN_IF_ERROR(cudaGetDeviceCount(&device_count));

  printf("CUDA Device Count: %d\n", device_count);

  // Get device properties
  cudaDeviceProp device_prop;
  RETURN_IF_ERROR(cudaGetDeviceProperties(&device_prop, 0));

  // Launch basic kernel
  dummy_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("Launched dummy kernel\n");

  // Launch addition
  int c;
  int* dev_c;
  RETURN_IF_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
  add<<<1, 1>>>(2, 7, dev_c);
  RETURN_IF_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
  printf("2 + 7 = %d\n", c);
  RETURN_IF_ERROR(cudaFree(dev_c));

  return EXIT_SUCCESS;
}
