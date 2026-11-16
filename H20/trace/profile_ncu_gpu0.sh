#!/usr/bin/env bash
set -uo pipefail
exec </dev/null

project=${H20_PROJECT_DIR:?Set H20_PROJECT_DIR}
run_dir=${H20_RUN_DIR:?Set H20_RUN_DIR to a new experiment directory}
gpu_uuid=${GPU0_UUID:-GPU-4ae56f60-49c6-6499-0960-fe62fca591b3}
cuda_home=${CUDA_HOME:-/home/weihaoni/.local/share/mok-compact-20261104/environment/cuda-13.0.88}
cublas_root=${CUBLAS13_ROOT:-/home/weihaoni/.venvs/myenv/lib/python3.12/site-packages/nvidia/cu13}
ncu=${NCU_BIN:-/opt/nvidia/nsight-compute/2025.3.1/ncu}
out=$run_dir/ncu

actual=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
apps=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
if [[ $actual != "$gpu_uuid" || -n $apps ]]; then
  echo "GPU 0 UUID mismatch or GPU 0 is busy" >&2
  exit 2
fi
mkdir "$out" || exit 2
nvidia-smi -i 0 --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv,noheader >"$out/gpu0.before.txt"
date -u +%FT%TZ >"$out/start_utc.txt"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_uuid" \
LD_LIBRARY_PATH="$cuda_home/lib64:$cuda_home/lib:$cublas_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$ncu" --set full --replay-mode kernel --devices 0 --launch-skip 51 --launch-count 1 \
  --kernel-name-base demangled --kernel-name 'regex:.*sgemmWarptiling.*' \
  --import-source yes --source-folders "$project/src" \
  --export "$out/kernel10_best" "$project/build/k10_h20" \
  --mode full --ids K10_0520 --output "$out/results.tsv" >"$out/capture.log" 2>&1
status=$?
date -u +%FT%TZ >"$out/end_utc.txt"
printf '%s\n' "$status" >"$out/exit_code.txt"
chmod -R a+rX "$out"
cat "$out/start_utc.txt" "$out/end_utc.txt" "$out/exit_code.txt"
exit "$status"
