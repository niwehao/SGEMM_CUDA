#!/usr/bin/env bash
set -euo pipefail

# Run from H20, for example:
# GPU0_UUID=GPU-... H20_RUN_DIR=/path/to/new/run CUDA_HOME=/path/to/cuda \
# CUBLAS13_ROOT=/path/to/cu13 ./trace/profile_all_gpu0.sh
# Set NSYS_SUDO=1 if Nsight Systems needs sudo. It prompts in this terminal;
# the script never stores or logs a password. Existing sudo authorization works too.

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=${H20_PROJECT_DIR:-$(cd "$here/.." && pwd)}
run_dir=${H20_RUN_DIR:?Set H20_RUN_DIR to a new, unused experiment directory}
gpu_uuid=${GPU0_UUID:?Set GPU0_UUID to the approved physical GPU 0 UUID}
binary=$project/build/k10_h20
cuda_home=${CUDA_HOME:-/home/weihaoni/.local/share/mok-compact-20261104/environment/cuda-13.0.88}
cublas_root=${CUBLAS13_ROOT:-/home/weihaoni/.venvs/myenv/lib/python3.12/site-packages/nvidia/cu13}
nsys=${NSYS_BIN:-$(command -v nsys || printf '%s' /opt/nvidia/nsight-systems/2024.6.2/bin/nsys)}
ncu=${NCU_BIN:-/opt/nvidia/nsight-compute/2025.3.1/ncu}
profile_ncu=$here/profile_ncu_gpu0.sh

die() { echo "$*" >&2; exit 2; }
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
[[ -x "$binary" ]] || die "H20 binary not executable: $binary"
[[ -x "$nsys" ]] || die "Nsight Systems executable not found: $nsys"
[[ -x "$ncu" ]] || die "Nsight Compute executable not found: $ncu"
[[ -f "$profile_ncu" ]] || die "Missing profiler script: $profile_ncu"

parent=$(dirname "$run_dir")
name=$(basename "$run_dir")
mkdir -p "$parent"
run_dir="$(cd "$parent" && pwd)/$name"
mkdir "$run_dir" || die "Run directory already exists; refusing to overwrite: $run_dir"

export H20_PROJECT_DIR="$project" H20_RUN_DIR="$run_dir" GPU0_UUID="$gpu_uuid"
export CUDA_HOME="$cuda_home" CUBLAS13_ROOT="$cublas_root" NCU_BIN="$ncu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$cuda_home/lib64:$cuda_home/lib:$cublas_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf '%s\n' "$(date -u +%FT%TZ)" >"$run_dir/workflow_start_utc.txt"
finish() {
  local status=$?
  printf '%s\n' "$(date -u +%FT%TZ)" >"$run_dir/workflow_end_utc.txt"
  printf '%s\n' "$status" >"$run_dir/workflow_exit_code.txt"
}
trap finish EXIT

preflight() {
  local stage=$1 actual apps
  actual=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]') || return 2
  apps=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) || return 2
  {
    printf 'nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader\n%s\n' "$actual"
    printf 'nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader\n%s\n' "$apps"
  } >"$stage/gpu0_preflight.txt"
  if [[ "$actual" != "$gpu_uuid" || -n "$(printf '%s' "$apps" | tr -d '[:space:]')" ]]; then
    echo "GPU 0 UUID mismatch or GPU 0 is busy; refusing this stage." >&2
    return 3
  fi
}

run_stage() {
  local stage=$1; shift
  mkdir "$stage"
  preflight "$stage" || return $?
  printf '%q ' "$@" >"$stage/command.txt"
  printf '\n' >>"$stage/command.txt"
  date -u +%FT%TZ >"$stage/start_utc.txt"
  local status=0
  "$@" >"$stage/capture.log" 2>&1 || status=$?
  date -u +%FT%TZ >"$stage/end_utc.txt"
  printf '%s\n' "$status" >"$stage/exit_code.txt"
  return "$status"
}

run_stage "$run_dir/unprofiled" \
  env CUDA_VISIBLE_DEVICES="$gpu_uuid" "$binary" --mode full --ids K10_0520 \
  --output "$run_dir/unprofiled/results.tsv" || exit $?

nsys_cmd=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_uuid" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$nsys" profile --trace=cuda \
  --sample=none --cpuctxsw=none \
  --output "$run_dir/nsys/cuda_trace" \
  "$binary" --mode full --ids K10_0520 --output "$run_dir/nsys/results.tsv")
if [[ "${NSYS_SUDO:-0}" == 1 ]]; then
  [[ -t 0 && -t 2 ]] || die "NSYS_SUDO=1 needs a terminal for normal sudo authorization"
  printf '%s\n' 'sudo -v (password prompt, if needed, is interactive and is not recorded)' \
    >"$run_dir/nsys_sudo_command.txt"
  sudo -v
  nsys_cmd=(sudo -n "${nsys_cmd[@]}")
fi
run_stage "$run_dir/nsys" "${nsys_cmd[@]}" || exit $?

ncu_cmd=(env H20_PROJECT_DIR="$project" H20_RUN_DIR="$run_dir" \
  GPU0_UUID="$gpu_uuid" CUDA_HOME="$cuda_home" CUBLAS13_ROOT="$cublas_root" \
  NCU_BIN="$ncu" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" bash "$profile_ncu")
if [[ "${NCU_SUDO:-1}" != 0 ]]; then
  if [[ -t 0 && -t 2 ]]; then sudo -v; else sudo -n -v; fi
  ncu_cmd=(sudo -n "${ncu_cmd[@]}")
fi
run_stage "$run_dir/ncu_stage" "${ncu_cmd[@]}" || exit $?

echo "All stages completed. Results: $run_dir"
