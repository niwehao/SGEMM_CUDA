#include "runner.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct TestCase {
  const char *name;
  int M;
  int N;
  int K;
  float alpha;
  float beta;
  std::size_t pointer_offset_floats = 0;
};

// The first case reproduces the project's smallest benchmark. The remaining
// cases exercise tile edges, multiple tiles, K tails, and non-vector widths.
const TestCase kCases[] = {
    {"baseline_128", 128, 128, 128, 0.5f, 3.0f},
    // Full tiles, but every logical device pointer is one float past cudaMalloc.
    {"misaligned_full_tile", 128, 256, 128, 0.5f, 3.0f, 1},
    {"edge_127", 127, 127, 127, 1.25f, -0.5f},
    {"edge_129", 129, 129, 129, -0.75f, 0.25f},
    {"edge_255", 255, 255, 255, 0.25f, 0.75f},
    {"edge_256", 256, 256, 256, 1.0f, -1.0f},
    {"edge_257", 257, 257, 257, 0.5f, 0.5f},
    {"rect_Ktail_Nnot4", 133, 131, 31, -0.625f, 0.125f},
    {"multitile_Ktail_Nnot4", 257, 259, 33, 0.875f, -0.25f},
    {"multitile_Kaligned_Nnot4", 257, 131, 48, -0.5f, 0.75f},
    {"small_rect_Ktail", 65, 67, 19, 0.375f, -0.625f},
    {"smallK_1", 17, 19, 1, 0.5f, 0.25f},
    {"smallK_15", 17, 19, 15, -0.75f, 0.5f},
    {"smallK_16", 17, 19, 16, 1.0f, -0.5f},
    {"smallK_17", 17, 19, 17, 0.25f, 0.75f},
    {"smallK_31", 33, 35, 31, -0.625f, 0.125f},
    {"smallK_32", 33, 35, 32, 0.875f, -0.25f},
    {"smallK_33", 33, 35, 33, -0.5f, 0.75f},
};

constexpr float kAbsTolerance = 1.0e-3f;
constexpr float kRelTolerance = 2.0e-4f;

struct DeviceBuffer {
  // allocation is the cudaMalloc base; ptr may point inside it for alignment
  // tests. Always release allocation, never the shifted logical pointer.
  float *allocation = nullptr;
  float *ptr = nullptr;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (allocation != nullptr) {
      cudaFree(allocation);
    }
  }
};

enum class CaseResult { Pass, NumericFailure, RuntimeFailure };

bool check_cuda(cudaError_t status, const char *operation,
                const TestCase &test) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "[" << test.name << "] FAIL: " << operation << ": "
            << cudaGetErrorString(status) << "\n";
  return false;
}

bool allocate_matrix_buffer(DeviceBuffer &buffer, std::size_t elements,
                            const char *label, const TestCase &test,
                            std::size_t offset_floats) {
  const cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(&buffer.allocation),
                 (elements + offset_floats) * sizeof(float));
  if (status == cudaSuccess) {
    buffer.ptr = buffer.allocation + offset_floats;
    return true;
  }
  std::cerr << "[" << test.name << "] FAIL: cudaMalloc(" << label << "): "
            << cudaGetErrorString(status) << "\n";
  return false;
}

float input_value(std::size_t index, std::uint32_t salt) {
  // Integer mixing makes the inputs reproducible without using Python or RNG
  // state. The result stays finite and in approximately [-0.5, 0.5].
  std::uint32_t value = static_cast<std::uint32_t>(index);
  value = value * 1664525u + 1013904223u + salt * 2246822519u;
  value ^= value >> 15;
  value *= 3266489917u;
  value ^= value >> 13;
  const int centered = static_cast<int>(value % 2049u) - 1024;
  return static_cast<float>(centered) / 2048.0f;
}

std::vector<float> make_matrix(std::size_t elements, std::uint32_t salt) {
  std::vector<float> values(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    values[i] = input_value(i, salt);
  }
  return values;
}

bool check_cublas(cublasStatus_t status, const char *operation,
                  const TestCase &test) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << "[" << test.name << "] FAIL: " << operation
            << " returned cuBLAS status " << static_cast<int>(status)
            << "\n";
  return false;
}

CaseResult run_case(cublasHandle_t handle, const TestCase &test) {
  const std::size_t a_elements = static_cast<std::size_t>(test.M) * test.K;
  const std::size_t b_elements = static_cast<std::size_t>(test.K) * test.N;
  const std::size_t c_elements = static_cast<std::size_t>(test.M) * test.N;

  const std::vector<float> A = make_matrix(a_elements, 1u);
  const std::vector<float> B = make_matrix(b_elements, 7u);
  const std::vector<float> C_initial = make_matrix(c_elements, 19u);
  std::vector<float> C_result(c_elements);
  std::vector<float> C_reference(c_elements);

  DeviceBuffer dA, dB, dC, dC_reference;
  if (!allocate_matrix_buffer(dA, a_elements, "A", test,
                              test.pointer_offset_floats) ||
      !allocate_matrix_buffer(dB, b_elements, "B", test,
                              test.pointer_offset_floats) ||
      !allocate_matrix_buffer(dC, c_elements, "C", test,
                              test.pointer_offset_floats) ||
      !allocate_matrix_buffer(dC_reference, c_elements, "C_reference", test,
                              test.pointer_offset_floats)) {
    return CaseResult::RuntimeFailure;
  }

  if (!check_cuda(cudaMemcpy(dA.ptr, A.data(), a_elements * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "copy A to device", test) ||
      !check_cuda(cudaMemcpy(dB.ptr, B.data(), b_elements * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "copy B to device", test) ||
      !check_cuda(cudaMemcpy(dC.ptr, C_initial.data(),
                             c_elements * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "copy C to device", test) ||
      !check_cuda(cudaMemcpy(dC_reference.ptr, C_initial.data(),
                             c_elements * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "copy reference C to device", test)) {
    return CaseResult::RuntimeFailure;
  }

  // This is the row-major-to-column-major cuBLAS mapping used by
  // runCublasFP32 in src/runner.cu:129-131. Matrix storage is exact-sized.
  if (!check_cublas(
          cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, test.N, test.M,
                       test.K, &test.alpha, dB.ptr, CUDA_R_32F, test.N,
                       dA.ptr, CUDA_R_32F, test.K, &test.beta,
                       dC_reference.ptr, CUDA_R_32F, test.N,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "cublasGemmEx FP32", test)) {
    return CaseResult::RuntimeFailure;
  }
  if (!check_cuda(cudaDeviceSynchronize(), "synchronize cuBLAS reference",
                  test)) {
    return CaseResult::RuntimeFailure;
  }

  run_kernel(11, test.M, test.N, test.K, test.alpha, dA.ptr, dB.ptr,
             test.beta, dC.ptr, handle);
  if (!check_cuda(cudaGetLastError(), "kernel launch", test) ||
      !check_cuda(cudaDeviceSynchronize(), "kernel execution", test)) {
    return CaseResult::RuntimeFailure;
  }

  if (!check_cuda(cudaMemcpy(C_result.data(), dC.ptr,
                             c_elements * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy kernel output to host", test) ||
      !check_cuda(cudaMemcpy(C_reference.data(), dC_reference.ptr,
                             c_elements * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy reference output to host", test)) {
    return CaseResult::RuntimeFailure;
  }

  std::size_t bad_count = 0;
  std::size_t first_bad = c_elements;
  float max_abs_error = 0.0f;
  float max_allowed_error = 0.0f;
  for (std::size_t i = 0; i < c_elements; ++i) {
    const float expected = C_reference[i];
    const float actual = C_result[i];
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      if (first_bad == c_elements) {
        first_bad = i;
      }
      ++bad_count;
      continue;
    }

    const float abs_error = std::fabs(expected - actual);
    const float allowed_error =
        kAbsTolerance + kRelTolerance * std::fabs(expected);
    max_abs_error = std::max(max_abs_error, abs_error);
    max_allowed_error = std::max(max_allowed_error, allowed_error);
    if (abs_error > allowed_error) {
      if (first_bad == c_elements) {
        first_bad = i;
      }
      ++bad_count;
    }
  }

  if (bad_count != 0) {
    const std::size_t row = first_bad / static_cast<std::size_t>(test.N);
    const std::size_t col = first_bad % static_cast<std::size_t>(test.N);
    std::cerr << std::scientific << std::setprecision(6)
              << "[" << test.name << "] FAIL: " << bad_count << "/"
              << c_elements << " elements differ; first at C[" << row << ","
              << col << "] ref=" << C_reference[first_bad]
              << " got=" << C_result[first_bad]
              << " max_abs_error=" << max_abs_error
              << " max_observed_allowed=" << max_allowed_error << "\n";
    return CaseResult::NumericFailure;
  }

  std::cout << std::scientific << std::setprecision(6)
            << "[" << test.name << "] PASS: M=" << test.M << " N=" << test.N
            << " K=" << test.K << " alpha=" << test.alpha
            << " beta=" << test.beta
            << " pointer_offset_floats=" << test.pointer_offset_floats
            << " max_abs_error=" << max_abs_error
            << "\n";
  return CaseResult::Pass;
}

const TestCase *find_case(const std::string &name) {
  for (const TestCase &test : kCases) {
    if (name == test.name) {
      return &test;
    }
  }
  return nullptr;
}

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " [--case NAME | --list]\n"
            << "Default: run all cases.\n"
            << "Cases:\n";
  for (const TestCase &test : kCases) {
    std::cerr << "  " << test.name << " (M=" << test.M << ", N=" << test.N
              << ", K=" << test.K << ")\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  const TestCase *selected_case = nullptr;
  if (argc == 2 && std::string(argv[1]) == "--list") {
    for (const TestCase &test : kCases) {
      std::cout << test.name << " M=" << test.M << " N=" << test.N
                << " K=" << test.K << " alpha=" << test.alpha
                << " beta=" << test.beta
                << " pointer_offset_floats=" << test.pointer_offset_floats
                << "\n";
    }
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--case") {
    selected_case = find_case(argv[2]);
    if (selected_case == nullptr) {
      std::cerr << "Unknown case: " << argv[2] << "\n";
      print_usage(argv[0]);
      return 2;
    }
  } else if (argc != 1) {
    print_usage(argv[0]);
    return 2;
  }

  int device = 0;
  if (!check_cuda(cudaGetDevice(&device), "get current device",
                  kCases[0])) {
    return 2;
  }
  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, device),
                  "get device properties", kCases[0])) {
    return 2;
  }
  std::cout << "Running kernel11 correctness cases on " << properties.name
            << " (device " << device << ")\n";
  std::cout << "Tolerance: abs <= " << kAbsTolerance << " + "
            << kRelTolerance << " * abs(reference)\n";

  cublasHandle_t handle = nullptr;
  const cublasStatus_t create_status = cublasCreate(&handle);
  if (create_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "FAIL: cublasCreate returned status "
              << static_cast<int>(create_status) << "\n";
    return 2;
  }

  int passed = 0;
  int failed = 0;
  bool runtime_failure = false;
  const auto run_and_count = [&](const TestCase &test) {
    const CaseResult result = run_case(handle, test);
    if (result == CaseResult::Pass) {
      ++passed;
    } else {
      ++failed;
    }
    if (result == CaseResult::RuntimeFailure) {
      runtime_failure = true;
    }
  };

  if (selected_case != nullptr) {
    run_and_count(*selected_case);
  } else {
    for (const TestCase &test : kCases) {
      run_and_count(test);
      // An illegal access can leave the CUDA context unusable. Stop after a
      // runtime failure; --case allows each failure to be isolated in a fresh
      // process when comparing the original and repaired kernels.
      if (runtime_failure) {
        break;
      }
    }
  }

  const cublasStatus_t destroy_status = cublasDestroy(handle);
  if (destroy_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "FAIL: cublasDestroy returned status "
              << static_cast<int>(destroy_status) << "\n";
    ++failed;
  }

  std::cout << "SUMMARY: " << passed << " passed, " << failed << " failed\n";
  return failed == 0 ? 0 : 1;
}
