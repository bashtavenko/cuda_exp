#include <stdio.h>
#include <cuda_runtime.h>
#include "hello_kernel.cuh"

// CUDA kernel implementation
__global__ void hello_world_kernel() {
  printf("Block %d, Thread %d: Hello world\n", blockIdx.x, threadIdx.x);
}

// Host function that launches the kernel
void launch_hello_kernel() {
  // Launch kernel with 2 blocks and 4 threads per block
  hello_world_kernel<<<2, 4>>>();

  // Wait for GPU to finish
  cudaDeviceSynchronize();
}