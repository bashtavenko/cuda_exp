#ifndef CUDA_EXP__CUDA_UTILS_H_
#define CUDA_EXP__CUDA_UTILS_H_
#include <stdio.h>

static void ReturnIfError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define RETURN_IF_ERROR(err) (ReturnIfError(err, __FILE__, __LINE__))

#endif  // CUDA_EXP__CUDA_UTILS_H_
