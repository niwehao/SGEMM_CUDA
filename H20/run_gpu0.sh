#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [[ -z "${GPU0_UUID:-}" ]]; then
  echo "Set GPU0_UUID to the approved physical GPU 0 UUID; refusing to launch." >&2
  exit 2
fi
physical0_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F, '$1+0==0 {gsub(/[[:space:]]/,"",$2); print $2}')
if [[ "$physical0_uuid" != "$GPU0_UUID" ]]; then
  echo "GPU0 UUID mismatch: requested $GPU0_UUID, found $physical0_uuid." >&2
  exit 2
fi
compute_apps=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$compute_apps" ]]; then
  echo "Physical GPU 0 is busy (PIDs: $compute_apps); refusing to launch." >&2
  exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
export LD_LIBRARY_PATH="${CUDA_HOME:?Set CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUBLAS13_ROOT:?Set CUBLAS13_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$here"
exec "$here/build/k10_h20" "$@"
