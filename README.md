# CUDA and TensorRT ramp up and snippets

## Resources
[Programming Massively Parallel Processors](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)

[CUDA by Example](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685)  
[Source code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-.git)

### NVIDIA
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[CUDA samples](https://github.com/NVIDIA/cuda-samples/)  
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
[Tensor RT](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)

## Local or cloud machine
GTX 1650 - entry level   
Alternatives     
[Koyeb](https://www.koyeb.com/)  
[Vast.ai](https://vast.ai/)

## Local debugging

Set `CMAKE_CUDA_ARCHITECTURES` to the machine graphic card. For GeForce RTX 5070 the architecture is `70`.

On CLion add `cuda-gdb` as a custom debugger in Settings | Build, Execution... | Toolchain and set Debugger to `/usr/bin/cuda-gdb`

This works in cuda-gdb but not in CLion

```bash
nvcc -g -G -o cmake-build-debug-cuda-debug/hello_main hello_main.cu
cuda-gdb ./cmake-build-debug-cuda-debug/hello_main
```
