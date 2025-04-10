#include <cuda_runtime.h>
#include "hello_kernel.cuh"
#include "glog/logging.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"

absl::Status Run() {
  // Print basic info
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return absl::InternalError(absl::StrFormat(
        "Failed to get CUDA device count: %s", cudaGetErrorString(err)));
  }

  LOG(INFO) << absl::StreamFormat("CUDA Device Count: %d", device_count);

  if (device_count == 0) {
    return absl::InternalError("No CUDA capable devices found.");
  }

  // Get device properties
  cudaDeviceProp device_prop;
  err = cudaGetDeviceProperties(&device_prop, 0);
  if (err != cudaSuccess) {
    return absl::InternalError(absl::StrFormat(
        "Failed to get CUDA device properties: %s", cudaGetErrorString(err)));
  }

  LOG(INFO) << absl::StreamFormat("Using device 0: %s", device_prop.name);

  // Launch the CUDA kernel
  launch_hello_kernel();

  LOG(INFO) << "CUDA program executed successfully";
  return absl::OkStatus();
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  auto status = Run();
  if (!status.ok()) LOG(ERROR) << status.message();
  return status.ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}