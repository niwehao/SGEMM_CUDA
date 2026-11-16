#pragma once
#include <cuda_runtime.h>

struct Candidate {
  const char *id;
  int threads, BM, BN, BK, WM, WN, WNITER, TM, TN, WMITER, SMEM_bytes;
  void (*launch)(int, int, int, float, float *, float *, float, float *, cudaStream_t);
};

extern "C" void launch_K10_0520(int, int, int, float, float *, float *, float, float *, cudaStream_t);

static const Candidate kCandidates[] = {
    {"K10_0520", 128, 64, 128, 16, 64, 32, 1, 4, 8, 2, 12288, launch_K10_0520},
};
static const int kCandidateCount = sizeof(kCandidates) / sizeof(kCandidates[0]);
