# Kernel 11 correctness harness

`kernel11_correctness.cu` compares kernel 11 with cuBLAS FP32. Its 18 cases cover tile boundaries, partial K tiles, non-square sizes, and deliberately misaligned device pointers.

Run these commands from the repository root with CUDA 12.8 on an H20 (`sm_90`):

```bash
export PATH=/usr/local/cuda-12.8/bin:"$PATH"
mkdir -p build
nvcc -std=c++20 -O2 -lineinfo -arch=sm_90 -I src \
  tests/kernel11_correctness.cu src/runner.cu -lcublas \
  -o build/kernel11_correctness

CUDA_VISIBLE_DEVICES=0 build/kernel11_correctness
CUDA_VISIBLE_DEVICES=0 build/kernel11_correctness --case misaligned_full_tile
```

For another GPU, change `-arch=sm_90` to that GPU's compute capability.

Optional memory check for the baseline case:

```bash
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --padding 4096 \
  --error-exitcode 99 build/kernel11_correctness --case baseline_128
```
