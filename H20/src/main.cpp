#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gnu/libc-version.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "candidate_manifest.hpp"
#include "build_metadata.hpp"

namespace {
constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr float kAlpha = 0.5f;
constexpr float kBeta = 3.0f;
constexpr uint32_t kSeed = 20261116u;
constexpr double kOps = 2.0 * static_cast<double>(kM) * kN * kK;
constexpr double kTolerance = 0.01;
constexpr uint32_t kSaltA = 0x9e3779b9u;
constexpr uint32_t kSaltB = 0x85ebca6bu;
constexpr uint32_t kSaltC = 0xc2b2ae35u;

void checkCuda(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::ostringstream os;
    os << where << ": " << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
    throw std::runtime_error(os.str());
  }
}
void checkCublas(cublasStatus_t e, const char* where) {
  if (e != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream os;
    os << where << ": cuBLAS status " << static_cast<int>(e);
    throw std::runtime_error(os.str());
  }
}

uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

uint64_t fnv1a64(const float* p, size_t n) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < n * sizeof(float); ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

void fillInput(std::vector<float>& out, uint32_t salt) {
  for (size_t i = 0; i < out.size(); ++i) {
    const uint32_t h = mix32(kSeed ^ salt ^ (static_cast<uint32_t>(i) * 0x27d4eb2du));
    float v = static_cast<float>(h % 5u) + 0.01f * static_cast<float>((h >> 8) % 5u);
    if ((h >> 16) & 1u) v = -v;
    out[i] = v;
  }
}

std::string hex64(uint64_t x) {
  std::ostringstream os;
  os << std::hex << std::setfill('0') << std::setw(16) << x;
  return os.str();
}

cublasStatus_t cublasCall(cublasHandle_t h, float* A, float* B, float* C) {
  return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, kN, kM, kK, &kAlpha,
                      B, CUDA_R_32F, kN, A, CUDA_R_32F, kK, &kBeta, C,
                      CUDA_R_32F, kN, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

struct Compare {
  bool finite = true;
  double max_abs = 0.0;
  size_t bad_index = 0;
  float got = 0.0f;
  float ref = 0.0f;
  bool pass = true;
};

Compare compareOutput(const std::vector<float>& ref, const std::vector<float>& got) {
  Compare c;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
      c.finite = false;
      c.pass = false;
      c.bad_index = i;
      c.got = got[i];
      c.ref = ref[i];
      return c;
    }
    const double d = std::fabs(static_cast<double>(ref[i]) - got[i]);
    if (d > c.max_abs) {
      c.max_abs = d;
      c.bad_index = i;
      c.got = got[i];
      c.ref = ref[i];
    }
    if (d > kTolerance) c.pass = false;
  }
  return c;
}

bool allFinite(const std::vector<float>& a, size_t* bad = nullptr) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i])) {
      if (bad) *bad = i;
      return false;
    }
  }
  return true;
}

struct Options {
  std::string mode = "full";
  std::string output = "results.tsv";
  std::vector<std::string> ids;
};

Options parseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) o.mode = argv[++i];
    else if (a == "--output" && i + 1 < argc) o.output = argv[++i];
    else if (a == "--ids" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      std::string id;
      while (std::getline(ss, id, ',')) if (!id.empty()) o.ids.push_back(id);
    } else if (a == "--help") {
      std::cout << "usage: k10bench [--mode full|screen] [--output results.tsv] [--ids id,id,...]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete option: " + a);
    }
  }
  if (o.mode != "full" && o.mode != "screen") throw std::runtime_error("mode must be full or screen");
  return o;
}

struct Measurement {
  bool ok = true;
  std::string reason;
  std::vector<double> batch_ms;
};

bool callN(bool baseline, const Candidate* candidate, cublasHandle_t handle,
           float* dA, float* dB, float* dC, cudaStream_t stream, int iters,
           std::string* reason) {
  for (int i = 0; i < iters; ++i) {
    if (baseline) {
      const cublasStatus_t s = cublasCall(handle, dA, dB, dC);
      if (s != CUBLAS_STATUS_SUCCESS) {
        *reason = "cuBLAS call status " + std::to_string(static_cast<int>(s));
        return false;
      }
    } else {
      candidate->launch(kM, kN, kK, kAlpha, dA, dB, kBeta, dC, stream);
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        *reason = std::string("kernel launch: ") + cudaGetErrorName(e) + ": " + cudaGetErrorString(e);
        return false;
      }
    }
  }
  return true;
}

bool resetC(float* dC, const float* dC0, size_t bytes, cudaStream_t stream,
            std::string* reason) {
  const cudaError_t e = cudaMemcpyAsync(dC, dC0, bytes, cudaMemcpyDeviceToDevice, stream);
  if (e != cudaSuccess) {
    *reason = std::string("reset dC: ") + cudaGetErrorName(e) + ": " + cudaGetErrorString(e);
    return false;
  }
  return true;
}

bool checkFiniteOnDevice(float* dC, std::vector<float>& hostOut,
                         cudaStream_t stream, size_t* bad, std::string* reason) {
  const cudaError_t e = cudaMemcpyAsync(hostOut.data(), dC, hostOut.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream);
  if (e != cudaSuccess) {
    *reason = std::string("copy output: ") + cudaGetErrorName(e) + ": " + cudaGetErrorString(e);
    return false;
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    *reason = std::string("stream sync: ") + cudaGetErrorName(sync) + ": " + cudaGetErrorString(sync);
    return false;
  }
  if (!allFinite(hostOut, bad)) {
    *reason = "non-finite output at index " + std::to_string(*bad);
    return false;
  }
  return true;
}

Measurement measure(bool baseline, const Candidate* candidate,
                    cublasHandle_t handle, float* dA, float* dB, float* dC,
                    const float* dC0, size_t bytes, cudaStream_t stream,
                    int warmupIters, int timedIters, int batches,
                    std::vector<float>& hostOut) {
  Measurement m;
  cudaEvent_t start{}, stop{};
  std::string reason;
  try {
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    if (!resetC(dC, dC0, bytes, stream, &reason) ||
        !callN(baseline, candidate, handle, dA, dB, dC, stream, warmupIters, &reason)) {
      m.ok = false; m.reason = "warmup: " + reason;
      goto done;
    }
    {
      size_t bad = 0;
      if (!checkFiniteOnDevice(dC, hostOut, stream, &bad, &reason)) {
        m.ok = false; m.reason = "warmup: " + reason;
        goto done;
      }
    }
    if (!resetC(dC, dC0, bytes, stream, &reason)) {
      m.ok = false; m.reason = "warmup reset: " + reason;
      goto done;
    }
    for (int b = 0; b < batches; ++b) {
      if (!resetC(dC, dC0, bytes, stream, &reason)) {
        m.ok = false; m.reason = "batch reset: " + reason;
        goto done;
      }
      checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
      if (!callN(baseline, candidate, handle, dA, dB, dC, stream, timedIters, &reason)) {
        m.ok = false; m.reason = "timed calls: " + reason;
        goto done;
      }
      checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
      checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
      float elapsed = 0.0f;
      checkCuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
      m.batch_ms.push_back(elapsed);
      size_t bad = 0;
      if (!checkFiniteOnDevice(dC, hostOut, stream, &bad, &reason)) {
        m.ok = false; m.reason = "batch " + std::to_string(b + 1) + ": " + reason;
        goto done;
      }
    }
  } catch (const std::exception& e) {
    m.ok = false;
    m.reason = e.what();
  }
done:
  if (start) cudaEventDestroy(start);
  if (stop) cudaEventDestroy(stop);
  return m;
}

std::vector<double> perCallMs(const std::vector<double>& batchMs, int iters) {
  std::vector<double> v;
  for (double x : batchMs) v.push_back(x / iters);
  return v;
}
double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n & 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}
double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
std::string clean(const std::string& x) {
  std::string y = x;
  for (char& c : y) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
  return y;
}

void writeHeader(std::ostream& os) {
  os << "mode\ttype\tid\tthreads\tBM\tBN\tBK\tWM\tWN\tWNITER\tTM\tTN\tWMITER\tSMEM_bytes"
     << "\tcorrect\tfinite\tmax_abs\tbad_index\twarmup_calls\tcalls_per_batch\tbatch_count"
     << "\tbatch1_ms\tbatch2_ms\tbatch3_ms\tmedian_call_ms\tmean_call_ms\tmedian_GFLOPS\tmean_GFLOPS\treason\n";
}
void writeRow(std::ostream& os, const std::string& mode, const std::string& type,
              const std::string& id, const Candidate* c, const std::string& correct,
              const std::string& finite, double maxAbs, size_t badIndex,
              int warmup, int calls, int batches, const Measurement& m,
              const std::string& reason) {
  int threads=0,bm=0,bn=0,bk=0,wm=0,wn=0,wi=0,tm=0,tn=0,wmi=0,smem=0;
  if (c) {
    threads=c->threads; bm=c->BM; bn=c->BN; bk=c->BK; wm=c->WM; wn=c->WN;
    wi=c->WNITER; tm=c->TM; tn=c->TN; wmi=c->WMITER; smem=c->SMEM_bytes;
  }
  const auto pc=perCallMs(m.batch_ms,calls);
  std::vector<double> gf;
  for (double ms : pc) if (ms > 0.0) gf.push_back(kOps/(ms*1.0e6));
  os << mode << '\t' << type << '\t' << id << '\t' << threads << '\t' << bm << '\t' << bn << '\t'
     << bk << '\t' << wm << '\t' << wn << '\t' << wi << '\t' << tm << '\t' << tn << '\t'
     << wmi << '\t' << smem << '\t' << correct << '\t' << finite << '\t'
     << std::setprecision(9) << maxAbs << '\t' << badIndex << '\t' << warmup << '\t'
     << calls << '\t' << batches << '\t';
  for (int i=0;i<3;++i) {
    if (i < static_cast<int>(m.batch_ms.size())) os << std::fixed << std::setprecision(6) << m.batch_ms[i];
    os << '\t';
  }
  os << std::fixed << std::setprecision(6) << median(pc) << '\t' << mean(pc) << '\t'
     << std::setprecision(3) << median(gf) << '\t' << mean(gf) << '\t' << clean(reason) << '\n';
}

void printCompare(const std::string& id, const Compare& c) {
  std::cout << id << " correctness=" << (c.pass ? "PASS" : "FAIL")
            << " finite=" << (c.finite ? "yes" : "no")
            << " max_abs=" << std::setprecision(10) << c.max_abs;
  if (!c.pass) std::cout << " index=" << c.bad_index << " got=" << c.got << " ref=" << c.ref;
  std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parseOptions(argc, argv);
    const int warmupIters = opt.mode == "full" ? 50 : 10;
    const int timedIters = opt.mode == "full" ? 50 : 10;
    const int batches = opt.mode == "full" ? 3 : 2;
    const size_t elems = static_cast<size_t>(kM) * kN;
    const size_t bytes = elems * sizeof(float);

    int deviceCount=0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount != 1) throw std::runtime_error("requires exactly one visible GPU; set CUDA_VISIBLE_DEVICES to the approved physical GPU0 UUID");
    checkCuda(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int runtimeVersion=0, driverVersion=0;
    checkCuda(cudaRuntimeGetVersion(&runtimeVersion), "cudaRuntimeGetVersion");
    checkCuda(cudaDriverGetVersion(&driverVersion), "cudaDriverGetVersion");

    std::vector<float> hA(elems), hB(elems), hC0(elems), hRef(elems), hOut(elems);
    fillInput(hA, kSaltA);
    fillInput(hB, kSaltB);
    fillInput(hC0, kSaltC);

    std::cout << "mode=" << opt.mode << " M=N=K=4096 alpha=0.5 beta=3"
              << " warmup_calls=" << warmupIters << " calls_per_batch=" << timedIters
              << " batches=" << batches << '\n';
    std::cout << "input_seed=" << kSeed
              << " generator=mix32(seed^matrix_salt^index*0x27d4eb2d),"
                 " value=sign*((hash%5)+0.01*((hash>>8)%5))"
              << " A_fnv1a64=" << hex64(fnv1a64(hA.data(), elems))
              << " B_fnv1a64=" << hex64(fnv1a64(hB.data(), elems))
              << " C0_fnv1a64=" << hex64(fnv1a64(hC0.data(), elems)) << '\n';
    std::cout << "gpu=" << prop.name << " cc=" << prop.major << '.' << prop.minor
              << " visible_gpu_count=" << deviceCount << " CUDA_runtime=" << runtimeVersion
              << " CUDA_driver=" << driverVersion << " glibc=" << gnu_get_libc_version() << '\n';
    std::cout << "kernel10_sha256=" << K10_KERNEL_SHA256
              << " candidates_sha256=" << K10_CANDIDATES_SHA256
              << " harness_sha256=" << K10_HARNESS_SHA256 << '\n';
    std::cout << "cuBLAS call: cublasGemmEx FP32 CUBLAS_COMPUTE_32F CUBLAS_GEMM_DEFAULT_TENSOR_OP; tolerance=0.01\n";

    std::ofstream out(opt.output, std::ios::app);
    if (!out) throw std::runtime_error("cannot open results output: " + opt.output);
    if (out.tellp() == 0) writeHeader(out);
    out << std::setprecision(9);

    float *dA=nullptr,*dB=nullptr,*dC0=nullptr,*dC=nullptr;
    checkCuda(cudaMalloc(&dA, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&dB, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc(&dC0, bytes), "cudaMalloc C0");
    checkCuda(cudaMalloc(&dC, bytes), "cudaMalloc C");
    cudaStream_t stream{};
    cublasHandle_t handle{};
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetStream(handle, stream), "cublasSetStream");
    int blasVersion=0;
    checkCublas(cublasGetVersion(handle, &blasVersion), "cublasGetVersion");
    std::cout << "cuBLAS_version=" << blasVersion << '\n';

    checkCuda(cudaMemcpyAsync(dA, hA.data(), bytes, cudaMemcpyHostToDevice, stream), "copy A");
    checkCuda(cudaMemcpyAsync(dB, hB.data(), bytes, cudaMemcpyHostToDevice, stream), "copy B");
    checkCuda(cudaMemcpyAsync(dC0, hC0.data(), bytes, cudaMemcpyHostToDevice, stream), "copy C0");
    checkCuda(cudaStreamSynchronize(stream), "sync input copies");

    checkCuda(cudaMemcpyAsync(dC, dC0, bytes, cudaMemcpyDeviceToDevice, stream), "reset reference C");
    checkCublas(cublasCall(handle, dA, dB, dC), "cuBLAS reference GEMM");
    checkCuda(cudaMemcpyAsync(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost, stream), "copy reference");
    checkCuda(cudaStreamSynchronize(stream), "sync reference");
    size_t badFinite=0;
    if (!allFinite(hRef, &badFinite)) throw std::runtime_error("cuBLAS reference is non-finite at " + std::to_string(badFinite));

    Compare baselineCompare;
    {
      checkCuda(cudaMemcpyAsync(dC, dC0, bytes, cudaMemcpyDeviceToDevice, stream), "reset baseline correctness");
      checkCublas(cublasCall(handle, dA, dB, dC), "cuBLAS baseline correctness");
      checkCuda(cudaMemcpyAsync(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost, stream), "copy baseline correctness");
      checkCuda(cudaStreamSynchronize(stream), "sync baseline correctness");
      baselineCompare=compareOutput(hRef,hOut);
      printCompare("cuBLAS13",baselineCompare);
    }

    Candidate baseline{};
    Measurement baselineM;
    if (baselineCompare.pass) {
      baselineM=measure(true,&baseline,handle,dA,dB,dC,dC0,bytes,stream,warmupIters,timedIters,batches,hOut);
      if (!baselineM.ok) std::cout << "cuBLAS timing failure: " << baselineM.reason << '\n';
    } else {
      baselineM.ok=false;
      baselineM.reason="cuBLAS self-check exceeded tolerance";
    }
    writeRow(out,opt.mode,"baseline","cuBLAS13",nullptr,
             baselineCompare.pass?"PASS":"FAIL",baselineCompare.finite?"yes":"no",
             baselineCompare.max_abs,baselineCompare.bad_index,warmupIters,timedIters,batches,
             baselineM,baselineM.reason);
    out.flush();
    const bool baselineOkay = baselineCompare.pass && baselineM.ok;
    if (!baselineOkay) {
      throw std::runtime_error("cuBLAS baseline correctness or timing failed: " + baselineM.reason);
    }

    int selected=0;
    bool allCandidatesOkay=true;
    for (const Candidate& c : kCandidates) {
      if (!opt.ids.empty() && std::find(opt.ids.begin(),opt.ids.end(),c.id)==opt.ids.end()) continue;
      ++selected;
      std::cout << "candidate=" << c.id << " threads=" << c.threads << " BM=" << c.BM
                << " BN=" << c.BN << " BK=" << c.BK << " WM=" << c.WM
                << " WN=" << c.WN << " WNITER=" << c.WNITER << " TM=" << c.TM
                << " TN=" << c.TN << " WMITER=" << c.WMITER << " SMEM_bytes=" << c.SMEM_bytes << '\n';
      bool correctnessOkay=true;
      std::string failReason;
      Compare cmp;
      bool compared=false;
      try {
        checkCuda(cudaMemcpyAsync(dC,dC0,bytes,cudaMemcpyDeviceToDevice,stream),"reset candidate correctness");
        c.launch(kM,kN,kK,kAlpha,dA,dB,kBeta,dC,stream);
        const cudaError_t launchError=cudaGetLastError();
        if (launchError!=cudaSuccess) throw std::runtime_error(std::string("kernel launch: ")+cudaGetErrorName(launchError)+": "+cudaGetErrorString(launchError));
        checkCuda(cudaMemcpyAsync(hOut.data(),dC,bytes,cudaMemcpyDeviceToHost,stream),"copy candidate correctness");
        checkCuda(cudaStreamSynchronize(stream),"sync candidate correctness");
        cmp=compareOutput(hRef,hOut);
        compared=true;
        printCompare(c.id,cmp);
        correctnessOkay=cmp.pass;
        if (!cmp.pass) failReason=cmp.finite ? "max_abs exceeds 0.01" : "non-finite output or reference";
      } catch (const std::exception& e) {
        correctnessOkay=false;
        failReason=e.what();
        std::cerr << c.id << " correctness failure: " << failReason << '\n';
        cudaGetLastError();
      }

      Measurement m;
      if (correctnessOkay) {
        m=measure(false,&c,handle,dA,dB,dC,dC0,bytes,stream,warmupIters,timedIters,batches,hOut);
        if (!m.ok) std::cout << c.id << " timing failure: " << m.reason << '\n';
      } else {
        m.ok=false;
        m.reason=failReason;
      }
      writeRow(out,opt.mode,"kernel10",c.id,&c,correctnessOkay?"PASS":"FAIL",
               compared?(cmp.finite?"yes":"no"):"unknown",cmp.max_abs,cmp.bad_index,warmupIters,timedIters,batches,m,m.reason);
      out.flush();
      if (!correctnessOkay || !m.ok) allCandidatesOkay=false;
      const auto pc=perCallMs(m.batch_ms,timedIters);
      std::vector<double> gf;
      for (double ms : pc) if (ms>0.0) gf.push_back(kOps/(ms*1.0e6));
      if (m.ok) {
        std::cout << c.id << " batch_ms=";
        for (double v:m.batch_ms) std::cout << std::fixed << std::setprecision(3) << v << ',';
        std::cout << " median_ms_per_call=" << std::setprecision(6) << median(pc)
                  << " mean_ms_per_call=" << mean(pc)
                  << " median_GFLOPS=" << std::setprecision(2) << median(gf)
                  << " mean_GFLOPS=" << mean(gf) << '\n';
      }
    }
    if (selected==0) throw std::runtime_error("no matching candidates selected");
    checkCuda(cudaStreamSynchronize(stream),"final stream sync");
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA); cudaFree(dB); cudaFree(dC0); cudaFree(dC);
    return allCandidatesOkay ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << '\n';
    return 2;
  }
}
