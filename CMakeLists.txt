cmake_minimum_required(VERSION 3.30)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_exp CUDA)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

add_executable(hello_main hello_main.cu)

set_target_properties(hello_main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

