#include <cuda_runtime.h>
#include "10_kernel_warptiling.cuh"

extern "C" void launch_K10_0520(int M, int N, int K, float alpha,
                                float *A, float *B, float beta, float *C,
                                cudaStream_t stream) {
  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int WM = 64;
  constexpr int WN = 32;
  constexpr int WNITER = 1;
  constexpr int TM = 4;
  constexpr int TN = 8;
  constexpr int NUM_THREADS = 128;
  constexpr int WARPSIZE = 32;
  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  static_assert(BN % WN == 0 && BM % WM == 0);
  static_assert((BN / WN) * (BM / WM) == NUM_THREADS / WARPSIZE);
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
  static_assert(WM % WMITER == 0 && WN % WNITER == 0);
  static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);
  static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE);
  static_assert((NUM_THREADS * 4) % BK == 0);
  static_assert((NUM_THREADS * 4) % BN == 0);
  static_assert(BN % (16 * TN) == 0 && BM % (16 * TM) == 0);
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0);
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0);
  static_assert(NUM_THREADS % WARPSIZE == 0);
  static_assert(BK % 4 == 0 && BN % 4 == 0);
  static_assert(4096 % BM == 0 && 4096 % BN == 0 && 4096 % BK == 0);
  static_assert(BM == 64 && BN == 128 && BK == 16);
  static_assert(BM * BK + BK * BN <= 48 * 1024 / sizeof(float));
  static_assert(TN % 4 == 0, "float4 output requires TN divisible by four");

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
      <<<grid, NUM_THREADS, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}
