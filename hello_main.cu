#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(void) {
  // Just a dummy kernel
}

int main(void) {
  // Check for GPU
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    printf("Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }
  printf("CUDA Device Count: %d\n", device_count);

  // Get device properties
  cudaDeviceProp device_prop;
  err = cudaGetDeviceProperties(&device_prop, 0);
  if (err != cudaSuccess) {
    printf("Failed to get CUDA device properties: %s", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  // Launch kernel
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("Launched kernel\n");
  return EXIT_SUCCESS;
}
