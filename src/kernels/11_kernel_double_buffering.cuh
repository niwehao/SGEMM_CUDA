#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace db {

template <const int BM, const int BN, const int BK, const int NUM_THREADS>
__device__ void loadFromGmem(const int M, const int N, const int K,
                             const int blockRow, const int blockCol,
                             const int kStart, const float *A, const float *B,
                             float *As, float *Bs) {
  // The first and second half of the block load separate buffers. Map each
  // half's thread IDs to [0, NUM_THREADS / 2) so either half can fill a tile.
  const uint loaderTid = threadIdx.x % (NUM_THREADS / 2);
  constexpr uint loaderThreads = NUM_THREADS / 2;

  const bool fullKTile = kStart + BK <= K;
  const bool vectorizeA = blockRow + BM <= M && fullKTile && K % 4 == 0 &&
                          (reinterpret_cast<std::uintptr_t>(
                               &A[static_cast<size_t>(blockRow) * K + kStart]) &
                           0xF) == 0;
  if (vectorizeA) {
    constexpr uint rowStrideA = (loaderThreads * 4) / BK;
    const uint innerRowA = loaderTid / (BK / 4);
    const uint innerColA = loaderTid % (BK / 4);
    for (uint offset = 0; offset + rowStrideA <= BM;
         offset += rowStrideA) {
      const float4 tmp = reinterpret_cast<const float4 *>(
          &A[static_cast<size_t>(blockRow + innerRowA + offset) * K + kStart +
             innerColA * 4])[0];
      // Transpose A while storing it in shared memory.
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }
  } else {
    // Fill every shared-memory element, including partial M/K tiles. As is
    // transposed to [K, M] for the compute loop.
    for (uint idx = loaderTid; idx < BM * BK; idx += loaderThreads) {
      const uint k = idx / BM;
      const uint row = idx % BM;
      const int globalRow = blockRow + row;
      const int globalK = kStart + k;
      As[k * BM + row] =
          (globalRow < M && globalK < K)
              ? A[static_cast<size_t>(globalRow) * K + globalK]
              : 0.0f;
    }
  }

  const bool vectorizeB = blockCol + BN <= N && fullKTile && N % 4 == 0 &&
                          (reinterpret_cast<std::uintptr_t>(
                               &B[static_cast<size_t>(kStart) * N + blockCol]) &
                           0xF) == 0;
  if (vectorizeB) {
    constexpr uint rowStrideB = loaderThreads / (BN / 4);
    const uint innerRowB = loaderTid / (BN / 4);
    const uint innerColB = loaderTid % (BN / 4);
    for (uint offset = 0; offset + rowStrideB <= BK;
         offset += rowStrideB) {
      const float4 tmp = reinterpret_cast<const float4 *>(
          &B[static_cast<size_t>(kStart + innerRowB + offset) * N + blockCol +
             innerColB * 4])[0];
      const uint sharedIdx = (innerRowB + offset) * BN + innerColB * 4;
      Bs[sharedIdx + 0] = tmp.x;
      Bs[sharedIdx + 1] = tmp.y;
      Bs[sharedIdx + 2] = tmp.z;
      Bs[sharedIdx + 3] = tmp.w;
    }
  } else {
    // Fill every shared-memory element, including partial N/K tiles.
    for (uint idx = loaderTid; idx < BK * BN; idx += loaderThreads) {
      const uint k = idx / BN;
      const uint col = idx % BN;
      const int globalK = kStart + k;
      const int globalCol = blockCol + col;
      Bs[k * BN + col] =
          (globalK < K && globalCol < N)
              ? B[static_cast<size_t>(globalK) * N + globalCol]
              : 0.0f;
    }
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmDoubleBuffering(const int M, const int N, const int K,
                         const float alpha, float *A, float *B, float beta,
                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // setup double buffering split
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  const int blockRow = cRow * BM;
  const int blockCol = cCol * BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // load first (B0)
    db::loadFromGmem<BM, BN, BK, NUM_THREADS>(M, N, K, blockRow, blockCol, 0,
                                               A, B, As, Bs);
  }
  __syncthreads();

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // process current (B0)
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
    } else {
      // load current + 1 (B1)
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, NUM_THREADS>(
            M, N, K, blockRow, blockCol, bkIdx + BK, A, B, As + (BM * BK),
            Bs + (BK * BN));
      }
    }
    // Both half-block roles rendezvous at the same block-wide barrier. The
    // first half has consumed B0; the second half has finished loading B1.
    __syncthreads();

    if (doubleBufferIdx == 0) {
      // The first half contributes its warp-owned outputs for B1.
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
    } else {
      // The second half contributes its warp-owned outputs for B0 while the
      // first half contributes for B1.
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
    }
    __syncthreads();

    if (doubleBufferIdx == 0) {
      // load current + 2 (B0), after all threads finish consuming B0
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, NUM_THREADS>(
            M, N, K, blockRow, blockCol, bkIdx + 2 * BK, A, B, As, Bs);
      }
    } else if (bkIdx + BK < K) {
      // The second half contributes its warp-owned outputs for B1.
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As + (BM * BK),
                              Bs + (BK * BN), warpRow, warpCol,
                              threadRowInWarp, threadColInWarp);
    }
    // Complete the load/compute pair before either shared buffer is reused.
    __syncthreads();
  }

  const int warpBlockRow = blockRow + warpRow * WM;
  const int warpBlockCol = blockCol + warpCol * WN;
  const size_t warpCOffset =
      static_cast<size_t>(warpBlockRow) * N + warpBlockCol;
  constexpr bool vectorStoreLayout = (WN % 4 == 0) && (WSUBN % 4 == 0) &&
                                     (TN % 4 == 0);
  const bool vectorizeC = vectorStoreLayout && warpBlockRow + WM <= M &&
                          warpBlockCol + WN <= N && N % 4 == 0 &&
                          (reinterpret_cast<std::uintptr_t>(C + warpCOffset) &
                           0xF) == 0;

  // Write out results. Complete, aligned warp tiles retain float4 stores;
  // boundary or unaligned tiles use guarded scalar accesses.
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        if (vectorizeC) {
          float *C_interim =
              C + warpCOffset + (wSubRowIdx * WSUBM) * N +
              wSubColIdx * WSUBN;
          for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            const size_t cOffset =
                static_cast<size_t>(threadRowInWarp * TM + resIdxM) * N +
                threadColInWarp * TN + resIdxN;
            float4 tmp =
                reinterpret_cast<float4 *>(C_interim + cOffset)[0];
            const int resultIdx =
                (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                wSubColIdx * TN + resIdxN;
            tmp.x = alpha * threadResults[resultIdx + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[resultIdx + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resultIdx + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resultIdx + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(C_interim + cOffset)[0] = tmp;
          }
        } else {
          const int localRow = warpRow * WM + wSubRowIdx * WSUBM +
                               threadRowInWarp * TM + resIdxM;
          const int globalRow = blockRow + localRow;
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            const int localCol = warpCol * WN + wSubColIdx * WSUBN +
                                 threadColInWarp * TN + resIdxN;
            const int globalCol = blockCol + localCol;
            if (globalRow < M && globalCol < N) {
              const size_t cIdx =
                  static_cast<size_t>(globalRow) * N + globalCol;
              const int resultIdx =
                  (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                  wSubColIdx * TN + resIdxN;
              C[cIdx] = alpha * threadResults[resultIdx] + beta * C[cIdx];
            }
          }
        }
      }
    }
  }
}
