#ifndef HELLO_KERNEL_CUH
#define HELLO_KERNEL_CUH

// CUDA kernel function declaration
__global__ void hello_world_kernel();

// Host-callable function to launch the kernel
void launch_hello_kernel();

#endif //HELLO_KERNEL_CUH
