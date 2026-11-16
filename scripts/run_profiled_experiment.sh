#!/usr/bin/env bash

# Run one SGEMM workload three times: unprofiled, under Nsight Systems, and
# under Nsight Compute. Each invocation runs in its own directory so files
# written by sgemm.cu on a validation failure cannot replace another run's data.

set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage: KERNEL=<0..12> [SGEMM_BIN=/path/to/sgemm] [PROFILE_ROOT=/path] \
       [NSYS_BIN=nsys] [NCU_BIN=ncu] [NCU_SET=basic] \
       [NCU_KERNEL_NAME=<regex>] [NCU_LAUNCH_SKIP=0] \
       [NCU_LAUNCH_COUNT=1] [NCU_SUDO=0|1] \
       [NCU_SUDO_TTY=0|1] bash scripts/run_profiled_experiment.sh

If SGEMM_BIN is unset, this script builds sgemm from a source snapshot inside
the new run directory. PROFILE_ROOT defaults to benchmark_results/profiled_experiments.
EOF
}

KERNEL=${KERNEL:-}
if [[ ! "$KERNEL" =~ ^([0-9]|1[0-2])$ ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
PROFILE_ROOT=${PROFILE_ROOT:-"$REPO_ROOT/benchmark_results/profiled_experiments"}
NSYS_BIN=${NSYS_BIN:-nsys}
NCU_BIN=${NCU_BIN:-ncu}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}
CMAKE_BIN=${CMAKE_BIN:-cmake}
NVCC_BIN=${NVCC_BIN:-nvcc}
BUILD_JOBS=${BUILD_JOBS:-2}
NCU_SET=${NCU_SET:-basic}
NCU_LAUNCH_SKIP=${NCU_LAUNCH_SKIP:-0}
NCU_LAUNCH_COUNT=${NCU_LAUNCH_COUNT:-1}
NCU_SUDO=${NCU_SUDO:-0}
NCU_SUDO_TTY=${NCU_SUDO_TTY:-0}
SGEMM_BIN=${SGEMM_BIN:-}

if [[ "$PROFILE_ROOT" != /* ]]; then
  PROFILE_ROOT="$REPO_ROOT/$PROFILE_ROOT"
fi
if ! mkdir -p "$PROFILE_ROOT"; then
  printf 'Cannot create PROFILE_ROOT: %s\n' "$PROFILE_ROOT" >&2
  exit 1
fi
PROFILE_ROOT=$(cd "$PROFILE_ROOT" && pwd -P) || exit 1

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
if ! RUN_DIR=$(mktemp -d "$PROFILE_ROOT/${timestamp}_kernel_${KERNEL}.XXXXXX"); then
  printf 'Cannot create a unique run directory under %s\n' "$PROFILE_ROOT" >&2
  exit 1
fi

umask 077
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/build" \
  "$RUN_DIR/build_logs" "$RUN_DIR/baseline" "$RUN_DIR/nsys" "$RUN_DIR/ncu"

OVERALL_RC=0
GPU_UUID=
SGEMM_ABS=
SUMMARY="$RUN_DIR/summary.txt"

record_failure() {
  OVERALL_RC=1
  printf 'FAIL: %s\n' "$1" | tee -a "$SUMMARY" >&2
}

write_status() {
  local file=$1 phase=$2 rc=$3 started=$4 finished=$5
  {
    printf 'phase=%s\n' "$phase"
    printf 'started_utc=%s\n' "$started"
    printf 'finished_utc=%s\n' "$finished"
    printf 'exit_code=%s\n' "$rc"
    if (( rc == 0 )); then
      printf 'status=ok\n'
    else
      printf 'status=failed\n'
    fi
  } > "$file"
}

run_logged() {
  local phase=$1 phase_dir=$2
  shift 2
  local started finished start_epoch finish_epoch rc stdin_path=/dev/null
  if [[ ${1:-} == --stdin-tty ]]; then
    stdin_path=/dev/tty
    shift
  fi
  mkdir -p "$phase_dir" || return 1
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  start_epoch=$(date +%s)
  {
    printf 'cwd=%s\n' "$phase_dir"
    printf 'command=cd '
    printf '%q' "$phase_dir"
    printf ' &&'
    printf ' %q' "$@"
    printf '\n'
    printf 'started_utc=%s\n' "$started"
    printf 'stdin=%s\n' "$stdin_path"
  } > "$phase_dir/command.txt"

  if (cd "$phase_dir" && "$@" < "$stdin_path") > "$phase_dir/combined.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  finish_epoch=$(date +%s)
  write_status "$phase_dir/status.txt" "$phase" "$rc" "$started" "$finished"
  printf 'elapsed_seconds=%s\n' "$((finish_epoch - start_epoch))" >> "$phase_dir/status.txt"
  printf 'finished_utc=%s\n' "$finished" >> "$phase_dir/command.txt"
  printf 'elapsed_seconds=%s\n' "$((finish_epoch - start_epoch))" >> "$phase_dir/command.txt"
  return "$rc"
}

check_gpu0_idle() {
  local phase=$1 check_dir="$RUN_DIR/$1/gpu_preflight" rc process_count
  if run_logged "gpu0_idle_check_${phase}" "$check_dir" "$NVIDIA_SMI_BIN" -i 0 \
    --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader; then
    rc=0
  else
    rc=$?
  fi
  if (( rc != 0 )); then
    printf 'physical_gpu_index=0\nstatus=unknown\nquery_exit_code=%s\n' "$rc" > "$check_dir/idle_status.txt"
    record_failure "Could not verify GPU 0 is idle before $phase; see ${phase}/gpu_preflight/combined.log."
    return 1
  fi

  process_count=$(awk 'NF && tolower($0) !~ /no running processes found/ { n++ } END { print n+0 }' "$check_dir/combined.log")
  {
    printf 'physical_gpu_index=0\n'
    printf 'physical_gpu_uuid=%s\n' "$GPU_UUID"
    printf 'compute_process_count=%s\n' "$process_count"
    if (( process_count == 0 )); then
      printf 'status=idle\n'
    else
      printf 'status=busy\n'
    fi
  } > "$check_dir/idle_status.txt"
  if (( process_count != 0 )); then
    record_failure "GPU 0 has a compute process before $phase; see ${phase}/gpu_preflight/combined.log."
    return 1
  fi
  return 0
}

write_not_run() {
  local phase=$1 phase_dir=$2 reason=$3
  mkdir -p "$phase_dir"
  {
    printf 'phase=%s\n' "$phase"
    printf 'status=not_run\n'
    printf 'exit_code=125\n'
    printf 'reason=%s\n' "$reason"
  } > "$phase_dir/status.txt"
  printf 'command_not_run=%s\n' "$reason" > "$phase_dir/command.txt"
}

record_tool_version() {
  local label=$1 tool=$2 rc
  {
    printf '[%s]\n' "$label"
    printf 'command='
    printf '%q --version\n' "$tool"
  } >> "$RUN_DIR/environment.txt"
  if command -v "$tool" >/dev/null 2>&1; then
    if "$tool" --version </dev/null >> "$RUN_DIR/environment.txt" 2>&1; then
      rc=0
    else
      rc=$?
    fi
  else
    printf 'tool_not_found=%s\n' "$tool" >> "$RUN_DIR/environment.txt"
    rc=127
  fi
  printf 'exit_code=%s\n\n' "$rc" >> "$RUN_DIR/environment.txt"
  return "$rc"
}

copy_file_to_snapshot() {
  local rel=$1 source="$REPO_ROOT/$1" destination="$RUN_DIR/source_snapshot/$1"
  mkdir -p "$(dirname "$destination")" || return 1
  cp -p "$source" "$destination"
}

is_source_or_config_file() {
  case "$1" in
    *.cu|*.cuh|*.c|*.cc|*.cpp|*.h|*.hh|*.hpp|*.S|*.s|*.sh|*.cmake|*.yml|*.yaml|*.toml|*.json|*.md|*.txt|Makefile|CMakeLists.txt|.gitignore)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

write_summary_header() {
  cat > "$SUMMARY" <<EOF
run_directory=$RUN_DIR
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
kernel=$KERNEL
workload=one invocation of sgemm <kernel>; the current program runs its built-in sizes 128,256,512,1024,2048,4096
profile_root=$PROFILE_ROOT
EOF
}

write_summary_header
printf 'PROFILE_RUN_DIR=%s\n' "$RUN_DIR"

if [[ ! "$NCU_LAUNCH_SKIP" =~ ^[0-9]+$ || ! "$NCU_LAUNCH_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  record_failure 'NCU_LAUNCH_SKIP must be >= 0 and NCU_LAUNCH_COUNT must be > 0.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  record_failure 'BUILD_JOBS must be a positive integer.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
if [[ "$NCU_SUDO" != 0 && "$NCU_SUDO" != 1 ]]; then
  record_failure 'NCU_SUDO must be 0 or 1.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
if [[ "$NCU_SUDO_TTY" != 0 && "$NCU_SUDO_TTY" != 1 ]]; then
  record_failure 'NCU_SUDO_TTY must be 0 or 1.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi

# Record the source revision, complete tracked diff, current status, and a copy
# of tracked files plus untracked source/configuration files. The snapshot is
# also the input to the default build below.
if ! git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt" 2> "$RUN_DIR/git_error.log"; then
  record_failure 'Cannot read the repository commit; see git_error.log.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD > "$RUN_DIR/source_branch.txt" 2>> "$RUN_DIR/git_error.log" || true
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all > "$RUN_DIR/git_status.txt" 2>> "$RUN_DIR/git_error.log"
git_status_rc=$?
git -C "$REPO_ROOT" diff --binary HEAD -- > "$RUN_DIR/tracked_worktree.diff" 2>> "$RUN_DIR/git_error.log"
git_diff_rc=$?
if (( git_status_rc != 0 || git_diff_rc != 0 )); then
  record_failure 'Could not capture git status or tracked diff; see git_error.log.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi

git -C "$REPO_ROOT" ls-files -z > "$RUN_DIR/tracked_files.nul" 2>> "$RUN_DIR/git_error.log"
tracked_list_rc=$?
git -C "$REPO_ROOT" ls-files --others --exclude-standard -z > "$RUN_DIR/untracked_files.nul" 2>> "$RUN_DIR/git_error.log"
untracked_list_rc=$?
if (( tracked_list_rc != 0 || untracked_list_rc != 0 )); then
  record_failure 'Could not enumerate repository files; see git_error.log.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi

snapshot_rc=0
: > "$RUN_DIR/source_snapshot/tracked_files.txt"
: > "$RUN_DIR/source_snapshot/untracked_source_files.txt"
while IFS= read -r -d '' rel; do
  if [[ -f "$REPO_ROOT/$rel" ]]; then
    if copy_file_to_snapshot "$rel"; then
      printf '%s\n' "$rel" >> "$RUN_DIR/source_snapshot/tracked_files.txt"
    else
      printf 'copy_failed=%s\n' "$rel" >> "$RUN_DIR/source_snapshot/copy_errors.txt"
      snapshot_rc=1
    fi
  fi
done < "$RUN_DIR/tracked_files.nul"

while IFS= read -r -d '' rel; do
  case "$rel" in
    logs/*|*/logs/*|benchmark_results/*|*/benchmark_results/*|experiment_artifacts/*|*/experiment_artifacts/*|*.log|*.nsys-rep|*.ncu-rep|*.ncu-repz)
      continue
      ;;
  esac
  if is_source_or_config_file "$rel" && [[ -f "$REPO_ROOT/$rel" ]]; then
    if copy_file_to_snapshot "$rel"; then
      printf '%s\n' "$rel" >> "$RUN_DIR/source_snapshot/untracked_source_files.txt"
    else
      printf 'copy_failed=%s\n' "$rel" >> "$RUN_DIR/source_snapshot/copy_errors.txt"
      snapshot_rc=1
    fi
  fi
done < "$RUN_DIR/untracked_files.nul"

if (( snapshot_rc != 0 )); then
  record_failure 'Source snapshot copy failed; see source_snapshot/copy_errors.txt.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi

{
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf unknown)"
  printf 'uname='; uname -a 2>&1
  printf 'repo_root=%s\n' "$REPO_ROOT"
  printf 'git_commit=%s\n' "$(cat "$RUN_DIR/source_commit.txt")"
  printf 'git_branch=%s\n' "$(cat "$RUN_DIR/source_branch.txt" 2>/dev/null || printf unknown)"
  printf 'profile_root=%s\n' "$PROFILE_ROOT"
  printf 'requested_sgemm_bin=%s\n' "${SGEMM_BIN:-<build from source snapshot>}"
  printf 'ncu_capture=kernel replay; launch_skip=%s; launch_count=%s; set=%s\n' "$NCU_LAUNCH_SKIP" "$NCU_LAUNCH_COUNT" "$NCU_SET"
  printf 'ncu_sudo=%s\n' "$NCU_SUDO"
  printf 'ncu_sudo_tty=%s\n' "$NCU_SUDO_TTY"
  printf 'ncu_kernel_name_override=%s\n' "${NCU_KERNEL_NAME:-<default kernel-name filter>}"
  printf 'CUDA_VISIBLE_DEVICES will be set to physical GPU 0 UUID; DEVICE=0 after remapping.\n'
} > "$RUN_DIR/run_info.txt"

: > "$RUN_DIR/environment.txt"
printf '[host]\n' >> "$RUN_DIR/environment.txt"
printf 'command=date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ\nvalue=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/environment.txt"
printf 'command=hostname\nvalue=%s\n\n' "$(hostname 2>/dev/null || printf unknown)" >> "$RUN_DIR/environment.txt"
record_tool_version 'nvidia-smi' "$NVIDIA_SMI_BIN" || true
record_tool_version 'cmake' "$CMAKE_BIN" || true
record_tool_version 'nvcc' "$NVCC_BIN" || true
record_tool_version 'nsys' "$NSYS_BIN" || true
record_tool_version 'ncu' "$NCU_BIN" || true

for required_tool in "$NVIDIA_SMI_BIN" "$NSYS_BIN" "$NCU_BIN"; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    record_failure "Required tool is unavailable: $required_tool. See environment.txt."
  fi
done
if [[ "$NCU_SUDO" == 1 || "$NCU_SUDO_TTY" == 1 ]] && ! command -v sudo >/dev/null 2>&1; then
  record_failure 'NCU_SUDO or NCU_SUDO_TTY requires sudo; the script never reads or logs a password.'
fi
if (( OVERALL_RC != 0 )); then
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi

# Capture the installed CLIs' own help before starting a workload. This catches
# option drift on the target host and leaves the exact help text with the run.
run_logged 'nvidia_smi_query_help' "$RUN_DIR/cli_help/nvidia_smi" "$NVIDIA_SMI_BIN" --help-query-gpu
nvidia_smi_help_rc=$?
if (( nvidia_smi_help_rc != 0 )) || ! grep -Fq 'compute_cap' "$RUN_DIR/cli_help/nvidia_smi/combined.log"; then
  record_failure 'Installed nvidia-smi help does not confirm the compute_cap query field; refusing to run.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
run_logged 'nsys_profile_help' "$RUN_DIR/cli_help/nsys_profile" "$NSYS_BIN" profile --help
nsys_help_rc=$?
run_logged 'ncu_help' "$RUN_DIR/cli_help/ncu" "$NCU_BIN" --help
ncu_help_rc=$?
if (( nsys_help_rc != 0 || ncu_help_rc != 0 )); then
  record_failure 'Could not read the installed Nsight CLI help; refusing to start a workload.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
for option in --trace --sample --cpuctxsw --output --force-overwrite; do
  if ! grep -Fq -- "$option" "$RUN_DIR/cli_help/nsys_profile/combined.log"; then
    record_failure "Installed nsys profile help does not list $option."
  fi
done
for option in --set --replay-mode --devices --launch-skip --launch-count --export --kernel-name --kernel-name-base --import; do
  if ! grep -Fq -- "$option" "$RUN_DIR/cli_help/ncu/combined.log"; then
    record_failure "Installed ncu help does not list $option."
  fi
done
if (( OVERALL_RC != 0 )); then
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi

# nvidia-smi index 0 is the physical GPU selected for the run. Passing its UUID
# to CUDA_VISIBLE_DEVICES avoids inheriting an index remapping from the shell;
# the application then selects CUDA ordinal 0, which maps only to that UUID.
if "$NVIDIA_SMI_BIN" -i 0 --query-gpu=uuid --format=csv,noheader > "$RUN_DIR/gpu0_uuid.txt" 2> "$RUN_DIR/gpu_probe.log"; then
  GPU_UUID=$(awk 'NF { print $1; exit }' "$RUN_DIR/gpu0_uuid.txt")
else
  gpu_probe_rc=$?
  printf 'physical_gpu_index=0\nexit_code=%s\n' "$gpu_probe_rc" > "$RUN_DIR/gpu0_status.txt"
  cat "$RUN_DIR/gpu_probe.log" >> "$RUN_DIR/gpu0_status.txt"
  record_failure 'Could not query physical GPU 0; see gpu0_status.txt.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
if [[ ! "$GPU_UUID" =~ ^GPU- ]]; then
  record_failure "GPU 0 did not return a physical GPU UUID (got '${GPU_UUID:-empty}'); refusing to run."
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
if "$NVIDIA_SMI_BIN" -i 0 --query-gpu=compute_cap --format=csv,noheader > "$RUN_DIR/gpu0_compute_capability_raw.txt" 2>> "$RUN_DIR/gpu_probe.log"; then
  GPU_COMPUTE_CAPABILITY_RAW=$(awk 'NF { print $1; exit }' "$RUN_DIR/gpu0_compute_capability_raw.txt")
else
  gpu_cap_query_rc=$?
  printf 'query=compute_cap\nexit_code=%s\n' "$gpu_cap_query_rc" > "$RUN_DIR/gpu0_compute_capability.txt"
  cat "$RUN_DIR/gpu_probe.log" >> "$RUN_DIR/gpu0_compute_capability.txt"
  record_failure 'Cannot query compute_cap for physical GPU 0; refusing to build or run. See gpu0_compute_capability.txt.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY_RAW//./}
if [[ ! "$GPU_COMPUTE_CAPABILITY" =~ ^[0-9]{2,3}$ ]]; then
  printf 'raw_compute_cap=%s\nparsed_architecture=%s\n' "$GPU_COMPUTE_CAPABILITY_RAW" "$GPU_COMPUTE_CAPABILITY" > "$RUN_DIR/gpu0_compute_capability.txt"
  record_failure "Unexpected compute_cap value '${GPU_COMPUTE_CAPABILITY_RAW:-empty}' for physical GPU 0; refusing to run."
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
{
  printf 'physical_gpu_index=0\nquery=compute_cap\n'
  printf 'raw_compute_cap=%s\ncmake_cuda_compute_capability=%s\n' "$GPU_COMPUTE_CAPABILITY_RAW" "$GPU_COMPUTE_CAPABILITY"
} > "$RUN_DIR/gpu0_compute_capability.txt"
{
  printf 'physical_gpu_index=0\n'
  printf 'cuda_visible_devices=%s\n' "$GPU_UUID"
  printf 'runtime_device_ordinal=0\n'
  printf 'command='
  printf '%q -i 0 --query-gpu=index,uuid,name,compute_cap,driver_version,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader\n' "$NVIDIA_SMI_BIN"
} > "$RUN_DIR/gpu0_status.txt"
if "$NVIDIA_SMI_BIN" -i 0 --query-gpu=index,uuid,name,compute_cap,driver_version,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader > "$RUN_DIR/gpu0.csv" 2>> "$RUN_DIR/gpu_probe.log"; then
  cat "$RUN_DIR/gpu0.csv" >> "$RUN_DIR/gpu0_status.txt"
else
  gpu_info_rc=$?
  printf 'gpu_details_exit_code=%s\n' "$gpu_info_rc" >> "$RUN_DIR/gpu0_status.txt"
  cat "$RUN_DIR/gpu_probe.log" >> "$RUN_DIR/gpu0_status.txt"
  record_failure 'Could not capture GPU model/driver details; see gpu0_status.txt.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
if "$NVIDIA_SMI_BIN" -i 0 -q > "$RUN_DIR/gpu0_full.txt" 2>&1; then
  :
else
  gpu_full_rc=$?
  printf 'gpu_full_query_exit_code=%s\n' "$gpu_full_rc" >> "$RUN_DIR/gpu0_status.txt"
fi

# Refuse to run into a shared workload; retain the process query for auditing.
printf 'command=%q -i 0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader\n' "$NVIDIA_SMI_BIN" > "$RUN_DIR/gpu0_compute_processes_before.txt"
if "$NVIDIA_SMI_BIN" -i 0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >> "$RUN_DIR/gpu0_compute_processes_before.txt" 2> "$RUN_DIR/gpu0_compute_processes_before.err"; then
  gpu_process_count=$(awk 'NF && $0 !~ /^command=/ && tolower($0) !~ /no running processes found/ { n++ } END { print n+0 }' "$RUN_DIR/gpu0_compute_processes_before.txt")
  printf 'compute_process_count=%s\n' "$gpu_process_count" >> "$RUN_DIR/gpu0_compute_processes_before.txt"
  if (( gpu_process_count != 0 )); then
    record_failure 'Physical GPU 0 already has a compute process; no workload was started. See gpu0_compute_processes_before.txt.'
    printf 'overall_status=failed\n' >> "$SUMMARY"
    exit 1
  fi
else
  gpu_process_query_rc=$?
  printf 'query_exit_code=%s\n' "$gpu_process_query_rc" >> "$RUN_DIR/gpu0_compute_processes_before.txt"
  cat "$RUN_DIR/gpu0_compute_processes_before.err" >> "$RUN_DIR/gpu0_compute_processes_before.txt"
  record_failure 'Could not verify that physical GPU 0 is idle; no workload was started.'
  printf 'overall_status=failed\n' >> "$SUMMARY"
  exit 1
fi
printf 'physical_gpu_uuid=%s\n' "$GPU_UUID" >> "$RUN_DIR/run_info.txt"
printf 'physical_gpu_index=0\n' >> "$RUN_DIR/run_info.txt"
printf 'gpu_compute_capability_raw=%s\n' "$GPU_COMPUTE_CAPABILITY_RAW" >> "$RUN_DIR/run_info.txt"
printf 'cmake_cuda_compute_capability=%s\n' "$GPU_COMPUTE_CAPABILITY" >> "$RUN_DIR/run_info.txt"

# Resolve the program. By default compile from the exact copied source tree;
# callers can point at a separately built binary using SGEMM_BIN.
if [[ -n "$SGEMM_BIN" ]]; then
  if [[ "$SGEMM_BIN" != /* ]]; then
    SGEMM_BIN="$REPO_ROOT/$SGEMM_BIN"
  fi
  if [[ ! -x "$SGEMM_BIN" ]]; then
    record_failure "SGEMM_BIN is not executable: $SGEMM_BIN"
    printf 'overall_status=failed\n' >> "$SUMMARY"
    exit 1
  fi
  SGEMM_ABS=$SGEMM_BIN
  printf 'binary_source=provided\n' >> "$RUN_DIR/run_info.txt"
  printf 'external_binary_architecture=unknown (no compile_commands.json for provided binary)\n' > "$RUN_DIR/cuda_architecture_validation.txt"
else
  printf 'binary_source=source_snapshot_build\n' >> "$RUN_DIR/run_info.txt"
  cmake_configure_rc=0
  run_logged 'cmake_configure' "$RUN_DIR/build_logs/configure" "$CMAKE_BIN" -S "$RUN_DIR/source_snapshot" -B "$RUN_DIR/build" -DCMAKE_BUILD_TYPE=Release -DCUDA_COMPUTE_CAPABILITY="$GPU_COMPUTE_CAPABILITY"
  cmake_configure_rc=$?
  if (( cmake_configure_rc == 0 )); then
    run_logged 'cmake_build' "$RUN_DIR/build_logs/build" "$CMAKE_BIN" --build "$RUN_DIR/build" --target sgemm --parallel "$BUILD_JOBS" --verbose
    cmake_build_rc=$?
  else
    cmake_build_rc=125
    printf 'phase=cmake_build\nstatus=not_run\nreason=cmake_configure_failed\nexit_code=125\n' > "$RUN_DIR/build/status.txt"
  fi
  printf 'configure_exit_code=%s\nbuild_exit_code=%s\n' "$cmake_configure_rc" "$cmake_build_rc" > "$RUN_DIR/build_status.txt"
  if (( cmake_configure_rc != 0 || cmake_build_rc != 0 )); then
    record_failure 'Build failed; see build_logs/configure/combined.log or build_logs/build/combined.log.'
    printf 'overall_status=failed\n' >> "$SUMMARY"
    exit 1
  fi
  cache_arch=$(sed -n 's/^CUDA_COMPUTE_CAPABILITY:STRING=//p' "$RUN_DIR/build/CMakeCache.txt" 2>/dev/null)
  sgemm_compile_line=$(grep -F "$RUN_DIR/source_snapshot/sgemm.cu" "$RUN_DIR/build_logs/build/combined.log" 2>/dev/null | grep -F "compute_${GPU_COMPUTE_CAPABILITY}" | grep -F "sm_${GPU_COMPUTE_CAPABILITY}" | head -n 1)
  {
    printf 'gpu_compute_capability_raw=%s\n' "$GPU_COMPUTE_CAPABILITY_RAW"
    printf 'expected_cmake_architecture=%s\n' "$GPU_COMPUTE_CAPABILITY"
    printf 'cmake_cache_architecture=%s\n' "${cache_arch:-missing}"
    printf 'sgemm_compile_command_architecture=%s\n' "$sgemm_compile_line"
  } > "$RUN_DIR/cuda_architecture_validation.txt"
  if [[ "$cache_arch" != "$GPU_COMPUTE_CAPABILITY" || -z "$sgemm_compile_line" || "$sgemm_compile_line" != *"compute_${GPU_COMPUTE_CAPABILITY}"* || "$sgemm_compile_line" != *"sm_${GPU_COMPUTE_CAPABILITY}"* ]]; then
    record_failure 'Built binary architecture did not match physical GPU 0 compute capability; see cuda_architecture_validation.txt.'
    printf 'overall_status=failed\n' >> "$SUMMARY"
    exit 1
  fi
  printf 'architecture_validation=passed\n' >> "$RUN_DIR/cuda_architecture_validation.txt"
  SGEMM_ABS="$RUN_DIR/build/sgemm"
  if [[ ! -x "$SGEMM_ABS" ]]; then
    record_failure "Build returned success but did not produce executable $SGEMM_ABS"
    printf 'overall_status=failed\n' >> "$SUMMARY"
    exit 1
  fi
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$SGEMM_ABS" > "$RUN_DIR/sgemm_binary.sha256"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$SGEMM_ABS" > "$RUN_DIR/sgemm_binary.sha256"
else
  printf 'binary_hash=unavailable (neither sha256sum nor shasum found)\n' > "$RUN_DIR/sgemm_binary.sha256"
fi
printf 'actual_sgemm_bin=%s\n' "$SGEMM_ABS" >> "$RUN_DIR/run_info.txt"

# Check the device immediately before every workload phase. A busy check stops
# this and all later GPU work while preserving earlier reports and status files.
baseline_rc=125
nsys_rc=125
nsys_validate_rc=125
ncu_rc=125
ncu_validate_rc=125
can_continue=1

if check_gpu0_idle baseline; then
  run_logged 'baseline' "$RUN_DIR/baseline" env CUDA_VISIBLE_DEVICES="$GPU_UUID" DEVICE=0 "$SGEMM_ABS" "$KERNEL"
  baseline_rc=$?
  awk '/Average elapsed time:/ { print }' "$RUN_DIR/baseline/combined.log" > "$RUN_DIR/baseline/results.txt"
  if (( baseline_rc != 0 )); then
    record_failure "Unprofiled workload exited with $baseline_rc; its output is retained in baseline/combined.log."
  fi
else
  baseline_rc=125
  write_not_run 'baseline' "$RUN_DIR/baseline" 'GPU0_busy_before_baseline'
  : > "$RUN_DIR/baseline/results.txt"
  can_continue=0
fi

nsys_base="$RUN_DIR/nsys/trace"
nsys_report="$nsys_base.nsys-rep"
if (( can_continue )); then
  if check_gpu0_idle nsys; then
    run_logged 'nsys' "$RUN_DIR/nsys" env CUDA_VISIBLE_DEVICES="$GPU_UUID" DEVICE=0 "$NSYS_BIN" profile \
      --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=false \
      --output "$nsys_base" "$SGEMM_ABS" "$KERNEL"
    nsys_rc=$?
    if [[ -s "$nsys_report" ]]; then
      run_logged 'nsys_validate' "$RUN_DIR/nsys/validation" "$NSYS_BIN" stats "$nsys_report"
      nsys_validate_rc=$?
    else
      nsys_validate_rc=1
      printf 'phase=nsys_validate\nstatus=failed\nexit_code=1\nreason=missing_or_empty_nsys_rep\n' > "$RUN_DIR/nsys/validation_status.txt"
      record_failure 'Nsight Systems did not create a non-empty .nsys-rep; see nsys/combined.log.'
    fi
    if (( nsys_rc != 0 )); then
      record_failure "Nsight Systems command exited with $nsys_rc; see nsys/combined.log."
    fi
    if (( nsys_validate_rc != 0 )); then
      record_failure "Nsight Systems report validation failed; see nsys/validation_status.txt or nsys/validation/combined.log."
    fi
    if [[ -s "$nsys_report" ]]; then
      printf 'report=%s\nreport_bytes=%s\nvalidation_exit_code=%s\n' "$nsys_report" "$(wc -c < "$nsys_report" | tr -d ' ')" "$nsys_validate_rc" > "$RUN_DIR/nsys/report_status.txt"
    else
      printf 'report=%s\nreport_bytes=0\nvalidation_exit_code=%s\n' "$nsys_report" "$nsys_validate_rc" > "$RUN_DIR/nsys/report_status.txt"
    fi
  else
    write_not_run 'nsys' "$RUN_DIR/nsys" 'GPU0_busy_before_nsys'
    write_not_run 'nsys_validate' "$RUN_DIR/nsys/validation" 'nsys_not_run'
    printf 'report=%s\nreport_bytes=0\nvalidation_exit_code=125\nstatus=not_run\n' "$nsys_report" > "$RUN_DIR/nsys/report_status.txt"
    can_continue=0
  fi
else
  write_not_run 'nsys' "$RUN_DIR/nsys" 'baseline_not_run_due_to_GPU0_preflight'
  write_not_run 'nsys_validate' "$RUN_DIR/nsys/validation" 'nsys_not_run'
  printf 'report=%s\nreport_bytes=0\nvalidation_exit_code=125\nstatus=not_run\n' "$nsys_report" > "$RUN_DIR/nsys/report_status.txt"
fi

if [[ -n "${NCU_KERNEL_NAME:-}" ]]; then
  ncu_kernel_name=$NCU_KERNEL_NAME
else
  case "$KERNEL" in
    0) ncu_kernel_name= ;;
    1) ncu_kernel_name='regex:.*sgemm_naive.*' ;;
    2) ncu_kernel_name='regex:.*sgemm_global_mem_coalesce.*' ;;
    3) ncu_kernel_name='regex:.*sgemm_shared_mem_block.*' ;;
    4) ncu_kernel_name='regex:.*sgemm1DBlocktiling.*' ;;
    5) ncu_kernel_name='regex:.*sgemm2DBlocktiling.*' ;;
    6) ncu_kernel_name='regex:.*sgemmVectorize.*' ;;
    7) ncu_kernel_name='regex:.*sgemmResolveBankConflicts.*' ;;
    8) ncu_kernel_name='regex:.*sgemmResolveBankExtraCol.*' ;;
    9) ncu_kernel_name='regex:.*sgemmAutotuned.*' ;;
    10) ncu_kernel_name='regex:.*sgemmWarptiling.*' ;;
    11) ncu_kernel_name='regex:.*sgemmDoubleBuffering.*' ;;
    12) ncu_kernel_name='regex:.*runSgemmDoubleBuffering2.*' ;;
  esac
fi

ncu_base="$RUN_DIR/ncu/kernel_${KERNEL}"
ncu_report=
ncu_args=(--set "$NCU_SET" --replay-mode kernel --devices 0 --launch-skip "$NCU_LAUNCH_SKIP" --launch-count "$NCU_LAUNCH_COUNT" --export "$ncu_base")
if [[ -n "$ncu_kernel_name" ]]; then
  ncu_args+=(--kernel-name-base demangled --kernel-name "$ncu_kernel_name")
fi
if [[ "$NCU_SUDO_TTY" == 1 ]]; then
  ncu_command=(sudo -S -k -p '' -- /bin/sh -c 'umask 022; exec 0</dev/null; exec "$@"' sh env CUDA_VISIBLE_DEVICES="$GPU_UUID" DEVICE=0 "$NCU_BIN")
elif [[ "$NCU_SUDO" == 1 ]]; then
  ncu_command=(sudo -n -- /bin/sh -c 'umask 022; exec 0</dev/null; exec "$@"' sh env CUDA_VISIBLE_DEVICES="$GPU_UUID" DEVICE=0 "$NCU_BIN")
else
  ncu_command=(env CUDA_VISIBLE_DEVICES="$GPU_UUID" DEVICE=0 "$NCU_BIN")
fi
if (( can_continue )); then
  if check_gpu0_idle ncu; then
    tty_auth_ready=1
    if [[ "$NCU_SUDO_TTY" == 1 ]]; then
      if ( : </dev/tty ) 2>/dev/null; then
        printf 'PROFILE_NCU_SUDO_AUTH_REQUIRED\n'
      else
        tty_auth_ready=0
        ncu_rc=125
        ncu_validate_rc=125
        record_failure 'NCU_SUDO_TTY=1 requires an available controlling TTY; NCU was not started.'
        write_not_run 'ncu' "$RUN_DIR/ncu" 'no_controlling_tty_for_ncu_sudo'
        write_not_run 'ncu_validate' "$RUN_DIR/ncu/validation" 'ncu_not_run'
        printf 'report=<none>\nexpected_candidates=%s.ncu-rep,%s.ncu-repz\nreport_bytes=0\nvalidation_exit_code=125\nstatus=not_run\nreason=no_controlling_tty\n' "$ncu_base" "$ncu_base" > "$RUN_DIR/ncu/report_status.txt"
      fi
    fi
    if (( tty_auth_ready )); then
      if [[ "$NCU_SUDO_TTY" == 1 ]]; then
        run_logged 'ncu' "$RUN_DIR/ncu" --stdin-tty "${ncu_command[@]}" "${ncu_args[@]}" "$SGEMM_ABS" "$KERNEL"
      else
        run_logged 'ncu' "$RUN_DIR/ncu" "${ncu_command[@]}" "${ncu_args[@]}" "$SGEMM_ABS" "${KERNEL}"
      fi
      ncu_rc=$?
    ncu_report_candidates=("$ncu_base.ncu-rep" "$ncu_base.ncu-repz")
    ncu_report_empty=
    ncu_report_multiple=0
    for candidate in "${ncu_report_candidates[@]}"; do
      if [[ -s "$candidate" ]]; then
        if [[ -z "$ncu_report" ]]; then
          ncu_report=$candidate
        else
          ncu_report_multiple=1
        fi
      elif [[ -e "$candidate" ]]; then
        ncu_report_empty=$candidate
      fi
    done
    if (( ncu_report_multiple != 0 )); then
      record_failure 'Nsight Compute produced both supported report extensions; see ncu/report_status.txt.'
    fi
    if [[ -s "$ncu_report" ]]; then
      run_logged 'ncu_validate' "$RUN_DIR/ncu/validation" "$NCU_BIN" --import "$ncu_report" --page details
      ncu_validate_rc=$?
    else
      ncu_validate_rc=1
      printf 'phase=ncu_validate\nstatus=failed\nexit_code=1\nreason=missing_or_empty_ncu_report\n' > "$RUN_DIR/ncu/validation_status.txt"
      record_failure 'Nsight Compute did not create a non-empty .ncu-rep or .ncu-repz; see ncu/combined.log.'
    fi
    if (( ncu_rc != 0 )); then
      record_failure "Nsight Compute command exited with $ncu_rc; its full error log is retained in ncu/combined.log."
    fi
    if (( ncu_validate_rc != 0 )); then
      record_failure "Nsight Compute report validation failed; see ncu/validation_status.txt or ncu/validation/combined.log."
    fi
    if [[ -n "$ncu_report" && -s "$ncu_report" ]]; then
      if [[ "$ncu_report" == *.ncu-repz ]]; then
        ncu_report_format=zstd_compressed_ncu_repz
      else
        ncu_report_format=uncompressed_ncu_rep
      fi
      printf 'report=%s\nformat=%s\nreport_bytes=%s\nvalidation_exit_code=%s\nmultiple_supported_reports=%s\n' "$ncu_report" "$ncu_report_format" "$(wc -c < "$ncu_report" | tr -d ' ')" "$ncu_validate_rc" "$ncu_report_multiple" > "$RUN_DIR/ncu/report_status.txt"
    else
      printf 'report=<none>\nempty_candidate=%s\nexpected_candidates=%s,%s\nreport_bytes=0\nvalidation_exit_code=%s\n' "${ncu_report_empty:-<none>}" "${ncu_report_candidates[0]}" "${ncu_report_candidates[1]}" "$ncu_validate_rc" > "$RUN_DIR/ncu/report_status.txt"
    fi
    fi
  else
    write_not_run 'ncu' "$RUN_DIR/ncu" 'GPU0_busy_before_ncu'
    write_not_run 'ncu_validate' "$RUN_DIR/ncu/validation" 'ncu_not_run'
    printf 'report=<none>\nexpected_candidates=%s.ncu-rep,%s.ncu-repz\nreport_bytes=0\nvalidation_exit_code=125\nstatus=not_run\n' "$ncu_base" "$ncu_base" > "$RUN_DIR/ncu/report_status.txt"
    can_continue=0
  fi
else
  write_not_run 'ncu' "$RUN_DIR/ncu" 'an earlier GPU0 preflight failed'
  write_not_run 'ncu_validate' "$RUN_DIR/ncu/validation" 'ncu_not_run'
  printf 'report=<none>\nexpected_candidates=%s.ncu-rep,%s.ncu-repz\nreport_bytes=0\nvalidation_exit_code=125\nstatus=not_run\n' "$ncu_base" "$ncu_base" > "$RUN_DIR/ncu/report_status.txt"
fi

{
  printf 'baseline_exit_code=%s\n' "$baseline_rc"
  printf 'nsys_exit_code=%s\n' "$nsys_rc"
  printf 'nsys_report_validation_exit_code=%s\n' "$nsys_validate_rc"
  printf 'ncu_exit_code=%s\n' "$ncu_rc"
  printf 'ncu_report_validation_exit_code=%s\n' "$ncu_validate_rc"
  if (( OVERALL_RC == 0 )); then
    printf 'overall_status=ok\noverall_exit_code=0\n'
  else
    printf 'overall_status=failed\noverall_exit_code=1\n'
  fi
} >> "$SUMMARY"

printf 'PROFILE_STATUS=%s\n' "$([[ $OVERALL_RC == 0 ]] && printf ok || printf failed)"
printf 'PROFILE_SUMMARY=%s\n' "$SUMMARY"
exit "$OVERALL_RC"
