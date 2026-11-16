.PHONY: all build debug clean profile bench cuobjdump

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<0..12> [SGEMM_BIN=/path/to/sgemm] [PROFILE_ROOT=/path] [NCU_SUDO_TTY=1]
SGEMM_BIN ?=
PROFILE_ROOT ?=
NSYS_BIN ?= nsys
NCU_BIN ?= ncu
NVIDIA_SMI_BIN ?= nvidia-smi
NCU_SET ?= basic
NCU_KERNEL_NAME ?=
NCU_LAUNCH_SKIP ?= 0
NCU_LAUNCH_COUNT ?= 1
NCU_SUDO ?= 0
NCU_SUDO_TTY ?= 0
BUILD_JOBS ?= 2

profile:
	@KERNEL="$(KERNEL)" \
	 SGEMM_BIN="$(SGEMM_BIN)" \
	 PROFILE_ROOT="$(PROFILE_ROOT)" \
	 NSYS_BIN="$(NSYS_BIN)" \
	 NCU_BIN="$(NCU_BIN)" \
	 NVIDIA_SMI_BIN="$(NVIDIA_SMI_BIN)" \
	 NCU_SET="$(NCU_SET)" \
	 NCU_KERNEL_NAME="$(NCU_KERNEL_NAME)" \
	 NCU_LAUNCH_SKIP="$(NCU_LAUNCH_SKIP)" \
	 NCU_LAUNCH_COUNT="$(NCU_LAUNCH_COUNT)" \
	 NCU_SUDO="$(NCU_SUDO)" \
	 NCU_SUDO_TTY="$(NCU_SUDO_TTY)" \
	 BUILD_JOBS="$(BUILD_JOBS)" \
	 bash scripts/run_profiled_experiment.sh

bench: build
	@bash gen_benchmark_results.sh
