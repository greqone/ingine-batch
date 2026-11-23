// Dense (non-expert) GEMMs of the batched decode step (src/model/batch.cpp) and the message-gate epilogue that
// consumes the fused forget|gain GEMM.
#pragma once

#include "kernels/prefill/prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

struct cublasLtContext;
typedef struct cublasLtContext* cublasLtHandle_t;

namespace ingine::kernels::batch {

// count products at once: product j reads A + j * a, B + j * b and writes C + j * c (elements)
struct Strided {
  int count = 1;
  std::int64_t a = 0, b = 0, c = 0;
};

// The step's own BF16 GEMM (mma.m16n8k16, FP32 accumulation), TunedGemm's candidates for the head besides
// cuBLASLt's: C = A . B^T as TunedGemm::run (one product), CTA tiles of 64 x 128, 64 x 64 or 32 x 128 (cfg 0, 1, 2;
// rows of A x rows of B), K in 64-element stages, the first weight stages loaded before the wait for the previous
// kernel (programmatic dependent launch). Requires K % 64 == 0, lda and ldb multiples of 8, N and ldc even, A and B
// 16-byte aligned, C 8-byte aligned.
inline constexpr int kMmaConfigs = 3;
void mma_gemm(int cfg, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32,
              int M, int N, int K, cudaStream_t stream);

// BF16 GEMM, the arithmetic of prefill::Gemm: C[M, N] (row-major, ldc) = A[M, K] (bf16, lda) . B[N, K]^T (bf16,
// ldb), FP32 accumulation, split-K partials reduced in FP32 only; C is bf16 (rounded once) or fp32.
// The algorithm is chosen by measurement in place. Shapes are grouped by bucket(M); a group's candidates are the
// cuBLASLt heuristic's first kCandidates at the bucket's M and configurations of cuBLASLt's configurable kernels, or
// for N >= kMmaMinN (the head) the heuristic's first kHeadCandidates and the mma_gemm configs (see group()).
// A group that has run in kWarmSteps steps (collect() calls; shapes that come and go, as the prefill's, keep the
// heuristic's first) and has candidates with fewer than kSamples timings runs them in turn between CUDA events, and
// collect() (after the stream synchronised) adds the timings; a candidate slower than kDrop times the group's
// fastest (twice, after a single timing) stops being run. Every other call runs the group's fastest candidate
// (least time measured). Descriptors and validity are cached per exact shape.
class TunedGemm {
 public:
  static constexpr int kCandidates = 6, kHeadCandidates = 2, kSamples = 2, kWarmSteps = 8, kMmaMinN = 65536;
  static constexpr float kDrop = 1.25f;
  static constexpr std::size_t kWorkspaceBytes = std::size_t(32) << 20;

  TunedGemm();
  ~TunedGemm();
  TunedGemm(const TunedGemm&) = delete;
  TunedGemm& operator=(const TunedGemm&) = delete;

  void run(const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32, int M,
           int N, int K, cudaStream_t stream, const Strided& batch = {});
  // Folds in the timings of the calls since the last collect() (waits for their events); ends a step.
  void collect();

  // the M an algorithm group is measured at (M rounded up: 8, 16, 32, 48, 64, 96, ..., 256, 320, ..., multiples of
  // 64 from 256)
  static int bucket(int M);

 private:
  struct Cand;
  struct Group;
  struct Plan;
  struct Timing {
    Group* group;
    int cand;
    cudaEvent_t start, stop;
  };
  // (M, N, K, lda, ldb, ldc, fp32 C, count, stride a, stride b, stride c)
  using Key = std::tuple<int, int, int, int, int, int, bool, int, std::int64_t, std::int64_t, std::int64_t>;
  Plan& plan(const Key& key);
  Group& group(const Key& key, const Plan& p);
  int configurable(bool c_fp32);

  cublasLtHandle_t handle_ = nullptr;
  void* workspace_ = nullptr;
  int sms_ = 0;
  int configurable_[2] = {-2, -2};  // the configurable algorithm id per C type (-2: not looked up, -1: none)
  std::map<Key, Plan*> plans_;
  std::map<Key, Group*> groups_;  // keyed with bucket(M), ldc and the strides of A and C zeroed
  std::vector<Group*> used_;  // since the last collect()
  std::vector<Timing> pending_;
  std::vector<cudaEvent_t> events_;  // idle
};

// prefill::gate_apply of G gates at once: fg fp32 [G, T, 2H] (gate g's row t: the forget products, then the gain
// products), norm_w fp32 [G, H]; out bf16 + g * out_stride [T, H] = gate g's rows (rows.h[g], rows.msg[g]).
void gate_apply(const prefill::GateRows& rows, int G, const float* fg, int T, const float* norm_w,
                __nv_bfloat16* out, std::size_t out_stride, cudaStream_t stream);

}  // namespace ingine::kernels::batch
