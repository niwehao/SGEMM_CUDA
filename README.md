# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus), then pass it to CMake:
    ```bash
    cmake -S . -B build -DCUDA_COMPUTE_CAPABILITY=80
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Capture profiling results: `make profile KERNEL=<kernel number>` (details below).

## Reproducible profiling runs

Run `make profile` on the Linux machine that has the NVIDIA GPU, driver, CUDA
toolkit, `nsys`, and `ncu`. A Mac can be used only as the analysis host for
already-collected report files; this wrapper does not profile from macOS.

`make profile KERNEL=11` creates one unique run directory under
`benchmark_results/profiled_experiments/`. It builds a Release binary from a
snapshot of the current source tree, then runs that same binary and kernel
argument three separate times:

1. A normal run saves the program output and its reported timings.
2. Nsight Systems saves a native `.nsys-rep` timeline.
3. Nsight Compute saves a native `.ncu-rep` or `.ncu-repz` report.

The two profilers do not wrap each other. The `nsys` and `ncu` runs are separate
process launches. Nsight Compute uses kernel replay, so its run can execute a
selected kernel more than once while collecting counters. Use the normal run's
timings for performance comparisons; profiler timings include collection and
replay overhead.

By default, Nsight Compute collects the `basic` set for one matching kernel
launch. For kernels 1–12, the filter selects that SGEMM kernel; for kernel 0,
no name filter is applied, so the first CUDA kernel may be an initialization
kernel rather than the intended cuBLAS GEMM. Set `NCU_KERNEL_NAME` if the
specific cuBLAS kernel is known. The program currently tests
sizes 128 through 4096 in one invocation, so the first matching launch is from
the 128 case. Change `NCU_LAUNCH_SKIP` and `NCU_LAUNCH_COUNT` to select a
different launch range. Set `NCU_KERNEL_NAME` to replace the default name
filter, or `NCU_SET=full` to request the full metric set. A larger set or more
launches can take much longer. The script validates each native report by
opening it with the matching Nsight CLI. A missing, empty, unreadable report or
nonzero command status makes the run fail and remains recorded in its directory.
The wrapper detects which extension the installed `ncu` actually wrote, records
that path and format in `ncu/report_status.txt`, then validates that exact file.
`.ncu-rep` is uncompressed; `.ncu-repz` is the zstd-compressed native format.

The wrapper reads the installed NVIDIA CLI help and checks the options it will
use. It queries physical GPU 0's UUID and compute capability with
`nvidia-smi`, then exposes only that UUID to the program and Nsight Compute. The
program uses CUDA ordinal 0 after this mapping. For a source build, the wrapper
converts capability `9.0` to CMake architecture `90`, passes it to CMake, then
checks `CMakeCache.txt` and the verbose NVCC command for `sgemm.cu`. A failed
query or mismatch stops the run. For a supplied `SGEMM_BIN`, the binary's
architecture is recorded as unknown because its compile command is not
available. The wrapper saves GPU utilization, memory use, and compute-process
state. It checks for other compute processes before each run phase and stops
when GPU 0 is busy. The run directory records that UUID, model, PCI bus ID,
driver, CUDA toolkit, Nsight versions, commands, UTC times, exit statuses,
source commit, tracked diff, and a snapshot of tracked plus untracked source and
configuration files. It also saves the built or supplied binary's SHA-256.
Each run has separate `baseline/`, `nsys/`, and `ncu/` subdirectories. Nothing
in an earlier run is overwritten or automatically removed.

The source snapshot is built in a new directory for each run. If you already
have the exact binary you want to profile, you can use it directly and place
artifacts on a persistent volume:

```bash
make profile KERNEL=11 \
  SGEMM_BIN=/path/to/sgemm \
  PROFILE_ROOT=/path/to/persistent/profile-runs
```

Other supported settings are `NSYS_BIN`, `NCU_BIN`, `NVIDIA_SMI_BIN`,
`NCU_SET`, `NCU_KERNEL_NAME`, `NCU_LAUNCH_SKIP`, `NCU_LAUNCH_COUNT`, and
`BUILD_JOBS`. If Nsight Compute needs sudo and a cached authorization works,
set `NCU_SUDO=1`; it uses noninteractive `sudo -n`. If sudo requires a fresh
password, set `NCU_SUDO_TTY=1` (this selects the TTY path even if `NCU_SUDO=1`
is also set). Just before NCU starts, the wrapper checks for a controlling
terminal and prints `PROFILE_NCU_SUDO_AUTH_REQUIRED` to stdout.
Then only the NCU sudo command reads from `/dev/tty`; all earlier commands and
the later report validation still use `/dev/null`. A person can enter the sudo
password in that terminal after the signal. Do not type it into a command,
redirect it to a log, or leave terminal echo enabled in an automation client.
The script never reads a password file or stores the password. The root child
sets `umask 022` so an NCU report is readable by the user who owns the run
directory, then closes standard input before launching Nsight Compute and the
workload. Any permission failure is kept in `ncu/combined.log` and causes a
nonzero result.

This capture path covers only `make profile KERNEL=...`. Running
`./build/sgemm ...`, `make bench`, or an autotuner script directly does not
automatically create Nsight reports. Those commands remain separate workflows.

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
