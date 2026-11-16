# H20 K10_0520 profile

## Workload

- Host: `qjhs-sh-lab-04`; physical GPU0: NVIDIA H20, UUID `GPU-4ae56f60-49c6-6499-0960-fe62fca591b3`.
- Workload: FP32 GEMM, M=N=K=4096, alpha=0.5, beta=3; full harness performs correctness, 50 warmup calls, then 3 batches of 50 event-timed calls.
- Candidate: `K10_0520`, threads=128, BM=64, BN=128, BK=16, WM=64, WN=32, WNITER=1, TM=4, TN=8.
- Source revision: `f465bcf4675c73b8603cf418883d5886b68b918d`; K10 source SHA-256: `6fa8ee47cd61200de30745485b2727f526a64baf10b9f2066e8e37857c016cdb`.
- Binary SHA-256: `45f87eb458ef3bb09e08dc1e258870eef2c44d72b0578a15ed9e8d1d36123f87`.
- Toolchain: CUDA 13.0.88, cuBLAS 13.1.1.3, Nsight Systems 2024.6.2.225-246235244400v0, Nsight Compute 2025.3.1.0.

## Capture scope and result

- Unprofiled run: 2026-11-16 09:16:59–09:17:02 UTC; exit 0. Correctness PASS, max absolute error `7.629e-6`; K10_0520 median `24579.157 GFLOPS`, cuBLAS median `30161.821 GFLOPS`.
- Nsight Systems: 09:17:52–09:18:00 UTC; CUDA trace only (`--sample=none --cpuctxsw=none`); exit 0. `nsys stats` exit 0 and found 201 matching K10 launches across the full harness.
- Nsight Compute: 09:23:09–09:23:18 UTC; `--set full`, kernel replay, filter `regex:.*sgemmWarptiling.*`, `--launch-skip 51 --launch-count 1`, source import enabled from the H20 `src/` tree; exit 0. The imported report was verified to show GPU device 0, the K10_0520 template, grid `(32,64,1)`, block `128`, and CUDA source.
- All runs exposed only physical GPU0 through `CUDA_VISIBLE_DEVICES` and used CUDA ordinal `DEVICE=0`.

## Files

- `kernel10_best.nsys-rep`: complete Nsight Systems native report.
- `kernel10_best.ncu-rep`: one selected K10_0520 launch with full counters and imported source.
- `kernel10_best.results.tsv`: unprofiled correctness and timing results.
- `SHA256SUMS`: SHA-256 manifest for the three files above. The files were copied from the remote run directory and their local hashes matched the remote hashes.
- `profile_all_gpu0.sh`: reruns the unprofiled benchmark, Nsight Systems, and Nsight Compute sequentially. `profile_ncu_gpu0.sh` is its Nsight Compute stage.

After building as described in `../README.md`, run the workflow on the GPU host:

```bash
GPU0_UUID=GPU-4ae56f60-49c6-6499-0960-fe62fca591b3 \
H20_RUN_DIR="$PWD/runs/k10-$(date -u +%Y%m%dT%H%M%SZ)" \
./trace/profile_all_gpu0.sh
```

The script checks physical GPU 0 before each stage. The Nsight Compute stage uses `sudo` and may prompt for authorization; it does not store the password. Set `CUDA_HOME`, `CUBLAS13_ROOT`, `NSYS_BIN`, or `NCU_BIN` to override the detected or recorded paths.

Remote run directory: `/home/weihaoni/gemm_k10_tune_20261116.E2PEFi/K10_0520_profile.RklJGV`.
