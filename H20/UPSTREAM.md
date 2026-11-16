# Extraction record

- User-fork source revision: `f465bcf4675c73b8603cf418883d5886b68b918d`
- Original device source: `src/kernels/10_kernel_warptiling.cuh`
- H20 device source: `H20/src/kernel/10_kernel_warptiling.cuh`
- SHA-256 for both files: `6fa8ee47cd61200de30745485b2727f526a64baf10b9f2066e8e37857c016cdb`
- Device source diff: none; the file is copied byte-for-byte.
- New host pieces: a one-candidate manifest, a deterministic correctness/benchmark driver, and a CUDA 13/cuBLAS 13 CMake build. `src/kernel/launch_kernel10.cu` contains the K10_0520 launcher and compile-time configuration checks. `cuda_stub/cublas_v2.h` is used only by that CUDA translation unit, whose kernel code does not call cuBLAS; `src/main.cpp` uses the real cuBLAS 13 header.
- The fork's `LICENSE` is copied unchanged.
