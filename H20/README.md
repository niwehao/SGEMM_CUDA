# H20 SGEMM kernel10

This is a small, standalone project for the fastest validated kernel10 configuration found on the qjy003 H20 node. It contains only the unchanged kernel10 device header, one host driver, and one launcher.

## Fixed workload and parameters

- `M=N=K=4096`, FP32
- `alpha=0.5`, `beta=3.0`
- Kernel: `NUM_THREADS=128`, `BM=64`, `BN=128`, `BK=16`, `WM=64`, `WN=32`, `WNITER=1`, `TM=4`, `TN=8`
- cuBLAS baseline: `cublasGemmEx`, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- Input: deterministic seed `20261116`; the executable prints the input hashes and kernel source SHA.
- Correctness compares every kernel output element against the cuBLAS result, with absolute tolerance `0.01` and a finite-value check.
- Timing: 50 warmup calls, then three batches of 50 consecutive calls. `dC` is reset to the same `C0` before each warmup or timed batch. CUDA events bracket only the 50 calls. The executable reports all batch times and median/mean GFLOPS for cuBLAS and kernel10.

## Build on qjy003

The node has CUDA Toolkit 13.0.88 and the cuBLAS 13 headers/libraries in a separate package root. Set these paths, then build a native SM90 binary:

```bash
export CUDA_HOME=/home/weihaoni/.local/share/mok-compact-20261104/environment/cuda-13.0.88
export CUBLAS13_ROOT=/home/weihaoni/.venvs/myenv/lib/python3.12/site-packages/nvidia/cu13
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=90-real \
  -DCUBLAS13_ROOT="$CUBLAS13_ROOT"
cmake --build build --parallel 2
```

The Release CUDA build uses `-O3 -lineinfo`; it does not use `-G`. CMake checks the copied kernel header SHA before compilation and links the versioned `.so.13` libraries explicitly.

The cuBLAS package also contains CUDA runtime headers. CMake adds CUDA Toolkit 13's include directory as an ordinary `-I` before the cuBLAS include path, so both translation units take `cuda_runtime.h` and `crt` headers from the Toolkit. The original kernel header includes `cublas_v2.h` but its device implementation does not call cuBLAS. The CUDA launcher TU gets only the tiny `cuda_stub/cublas_v2.h` shim; the host `main.cpp` gets the real cuBLAS 13 header and links the real cuBLAS 13 libraries.

Confirm the dynamic libraries before running:

```bash
LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUBLAS13_ROOT/lib" ldd build/k10_h20 | grep -E 'libcudart|libcublas'
readelf -d build/k10_h20 | grep -E 'NEEDED|RUNPATH'
```

Expected: `libcudart.so.13`, `libcublas.so.13`, and `libcublasLt.so.13` resolve from the configured CUDA/cuBLAS 13 roots. A `libcublas.so.11` result is wrong.

## Run on physical GPU 0

After confirming GPU 0 is reserved and idle, pass its approved UUID. The wrapper verifies the physical index-to-UUID mapping and refuses to run if a compute process is already present.

```bash
GPU0_UUID=GPU-... ./run_gpu0.sh
```

The executable writes `results.tsv` in this directory. It does not collect profiler traces.

## Source and license

`src/kernel/10_kernel_warptiling.cuh` is copied byte-for-byte from `src/kernels/10_kernel_warptiling.cuh` in this user's fork at source revision `f465bcf4675c73b8603cf418883d5886b68b918d`. Its SHA-256 is `6fa8ee47cd61200de30745485b2727f526a64baf10b9f2066e8e37857c016cdb`. `src/kernel/launch_kernel10.cu` contains only the fixed launcher and host-side static assertions. The minimal host driver replaces the original all-kernel dispatch; the templated kernel body is unchanged. The fork's MIT license is included as `LICENSE`.
