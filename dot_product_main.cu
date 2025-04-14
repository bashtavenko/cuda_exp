#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_utils.h"

constexpr int N = 10;

__global__ void add(int* a, int* b, int* c) {
  int tid = blockIdx.x;  // this thread handles the data at its thread id
  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main(void) {
  int a[N];
  int b[N];
  int c[N];
  int* dev_a;
  int* dev_b;
  int* dev_c;

  // Allocate the memory on the GPU
  RETURN_IF_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
  RETURN_IF_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
  RETURN_IF_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

  // fill the arrays 'a' and 'b' on the CPU
  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i * i;
  }

  // Copy 'a' and 'b' to GPU
  RETURN_IF_ERROR(
      cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  RETURN_IF_ERROR(
      cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  // N blocks, 1 thread
  // N blocks x 1 thread = N parallel threads
  // We could have launched N/2 per block and 2 threads
  // or N/4 with 4, or 1 block and N threads
  add<<<N, 1>>>(dev_a, dev_b, dev_c);

  RETURN_IF_ERROR(
      cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  // display the results
  for (int i = 0; i < N; ++i) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  RETURN_IF_ERROR(cudaFree(dev_a));
  RETURN_IF_ERROR(cudaFree(dev_b));
  RETURN_IF_ERROR(cudaFree(dev_c));

  return EXIT_SUCCESS;
}