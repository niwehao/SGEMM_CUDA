cmake_minimum_required(VERSION 3.19)
project(H20_K10 LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90-real" CACHE STRING "Build native code for H20 (SM90)")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
list(GET CUDAToolkit_INCLUDE_DIRS 0 K10_CUDA_INCLUDE_DIR)
# CMake treats the toolkit include as an implicit system path and emits it after
# ordinary -I roots. Use a build-tree symlink so Toolkit headers can be first.
set(K10_TOOLKIT_INCLUDE_LINK "${CMAKE_CURRENT_BINARY_DIR}/cuda_toolkit_include")
if(EXISTS "${K10_TOOLKIT_INCLUDE_LINK}" OR IS_SYMLINK "${K10_TOOLKIT_INCLUDE_LINK}")
  file(REMOVE "${K10_TOOLKIT_INCLUDE_LINK}")
endif()
file(CREATE_LINK "${K10_CUDA_INCLUDE_DIR}" "${K10_TOOLKIT_INCLUDE_LINK}" SYMBOLIC)
set(CUBLAS13_ROOT "" CACHE PATH "Root of NVIDIA cuBLAS 13 headers and libraries")
if(NOT CUBLAS13_ROOT)
  message(FATAL_ERROR "Set -DCUBLAS13_ROOT=/path/to/nvidia/cu13")
endif()
set(CUBLAS13_INCLUDE "${CUBLAS13_ROOT}/include")
set(CUBLAS13_LIBRARY "${CUBLAS13_ROOT}/lib/libcublas.so.13")
set(CUBLASLT13_LIBRARY "${CUBLAS13_ROOT}/lib/libcublasLt.so.13")
if(NOT EXISTS "${CUBLAS13_INCLUDE}/cublas_v2.h")
  message(FATAL_ERROR "cuBLAS 13 header not found: ${CUBLAS13_INCLUDE}/cublas_v2.h")
endif()
if(NOT EXISTS "${CUBLAS13_LIBRARY}" OR NOT EXISTS "${CUBLASLT13_LIBRARY}")
  message(FATAL_ERROR "cuBLAS 13 shared libraries not found under ${CUBLAS13_ROOT}/lib")
endif()

set(K10_KERNEL "${PROJECT_SOURCE_DIR}/src/kernel/10_kernel_warptiling.cuh")
set(K10_EXPECTED_SHA256 "6fa8ee47cd61200de30745485b2727f526a64baf10b9f2066e8e37857c016cdb")
file(SHA256 "${K10_KERNEL}" K10_KERNEL_SHA256)
if(NOT K10_KERNEL_SHA256 STREQUAL K10_EXPECTED_SHA256)
  message(FATAL_ERROR "kernel10 SHA mismatch: ${K10_KERNEL_SHA256}")
endif()
file(SHA256 "${PROJECT_SOURCE_DIR}/src/candidate_manifest.hpp" K10_CANDIDATES_SHA256)
file(SHA256 "${PROJECT_SOURCE_DIR}/src/main.cpp" K10_MAIN_SHA256)
file(SHA256 "${PROJECT_SOURCE_DIR}/src/kernel/launch_kernel10.cu" K10_LAUNCH_SHA256)
file(SHA256 "${PROJECT_SOURCE_DIR}/cuda_stub/cublas_v2.h" K10_CUDA_STUB_SHA256)
string(SHA256 K10_HARNESS_SHA256 "${K10_MAIN_SHA256}${K10_LAUNCH_SHA256}${K10_CANDIDATES_SHA256}${K10_CUDA_STUB_SHA256}")
set(K10_SOURCE_REVISION "f465bcf4675c73b8603cf418883d5886b68b918d")
configure_file(src/build_metadata.hpp.in build_metadata.hpp @ONLY)

add_executable(k10_h20
  src/main.cpp
  src/kernel/launch_kernel10.cu
)
# The launcher TU uses only CUDA headers and the local cuBLAS include shim.
# The host TU gets the real cuBLAS 13 header. CMake supplies the toolkit's CUDA
# headers as compiler system includes for both language modes.
target_include_directories(k10_h20 BEFORE PRIVATE
  "${K10_TOOLKIT_INCLUDE_LINK}"
  "$<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_SOURCE_DIR}/cuda_stub>"
  "$<$<COMPILE_LANGUAGE:CXX>:${CUBLAS13_INCLUDE}>"
  "${PROJECT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(k10_h20 PRIVATE
  CUDA::cudart
  "${CUBLAS13_LIBRARY}"
  "${CUBLASLT13_LIBRARY}"
)
target_compile_options(k10_h20 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(k10_h20 PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  BUILD_RPATH "${CUDAToolkit_ROOT}/lib64;${CUDAToolkit_ROOT}/lib;${CUBLAS13_ROOT}/lib"
)
