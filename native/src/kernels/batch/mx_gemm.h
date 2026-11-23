// MXFP8 GEMM through cuBLASLt (block-scaled FP8 tensor cores): C bf16 [M, N] (row stride ldc) = A [M, K] . W [N, K]^T,
// both float8_e4m3fn with an E8M0 scale per 32 consecutive K elements (kernels/batch/draft.h Mx layout), FP32
// accumulation, C rounded once. One heuristic algorithm cached per shape; run() is CUDA-graph capturable.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>

struct cublasLtContext;
typedef struct cublasLtContext* cublasLtHandle_t;

namespace ingine::kernels::bdraft {

class MxGemm {
 public:
  MxGemm();
  ~MxGemm();
  MxGemm(const MxGemm&) = delete;
  MxGemm& operator=(const MxGemm&) = delete;

  // a, a_sf: M rows; w, w_sf: N rows (K % 128 == 0, M and N any)
  void run(const std::uint8_t* a, const std::uint8_t* a_sf, const std::uint8_t* w, const std::uint8_t* w_sf,
           __nv_bfloat16* c, int ldc, int M, int N, int K, cudaStream_t stream);

  static constexpr std::size_t WORKSPACE_BYTES = std::size_t(32) << 20;

 private:
  struct Plan;
  Plan& plan(int M, int N, int K, int ldc);

  cublasLtHandle_t handle_ = nullptr;
  void* workspace_ = nullptr;
  std::map<std::tuple<int, int, int, int>, Plan*> plans_;
};

}  // namespace ingine::kernels::bdraft
