cmake_minimum_required(VERSION 3.30)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_exp CUDA)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

add_executable(hello_main hello_main.cu
        cuda_utils.h)

add_executable(dot_product dot_product_main.cu cuda_utils.h)

set_target_properties(hello_main dot_product PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

