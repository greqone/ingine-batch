#include "kernels/batch/mx_gemm.h"

#include <cublasLt.h>

#include <stdexcept>
#include <string>

namespace ingine::kernels::bdraft {

namespace {

void check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("MxGemm: cuBLASLt ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
}

}  // namespace

struct MxGemm::Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t w = nullptr, a = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
  ~Plan() {
    if (w) cublasLtMatrixLayoutDestroy(w);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

MxGemm::MxGemm() {
  check(cublasLtCreate(&handle_), "create");
  if (cudaMalloc(&workspace_, WORKSPACE_BYTES) != cudaSuccess) {
    cublasLtDestroy(handle_);
    throw std::runtime_error("MxGemm: workspace allocation failed");
  }
}

MxGemm::~MxGemm() {
  for (auto& [k, p] : plans_) delete p;
  cudaFree(workspace_);
  cublasLtDestroy(handle_);
}

// Row-major C[M, N] = A[M, K] . W[N, K]^T is column-major C^T[N, M] = W^T (op T on the K x N matrix W) . A
// (the K x M matrix A): cuBLAS's A is W, its B is A.
MxGemm::Plan& MxGemm::plan(int M, int N, int K, int ldc) {
  const auto key = std::make_tuple(M, N, K, ldc);
  if (auto it = plans_.find(key); it != plans_.end()) return *it->second;
  auto* p = new Plan;
  try {
    check(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)), "a scale mode");
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)), "b scale mode");
    check(cublasLtMatrixLayoutCreate(&p->w, CUDA_R_8F_E4M3, K, N, K), "layout w");
    check(cublasLtMatrixLayoutCreate(&p->a, CUDA_R_8F_E4M3, K, M, K), "layout a");
    check(cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16BF, N, M, ldc), "layout c");
    // the heuristic wants scale pointers; run() sets the call's
    const void* any = workspace_;
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &any, sizeof(any)), "a scale");
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &any, sizeof(any)), "b scale");
    cublasLtMatmulPreference_t pref = nullptr;
    check(cublasLtMatmulPreferenceCreate(&pref), "preference create");
    const std::size_t ws = WORKSPACE_BYTES;
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    cublasStatus_t st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    if (st == CUBLAS_STATUS_SUCCESS)
      st = cublasLtMatmulAlgoGetHeuristic(handle_, p->op, p->w, p->a, p->c, p->c, pref, 1, &result, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    check(st, "heuristic");
    if (found == 0) throw std::runtime_error("MxGemm: no cuBLASLt algorithm");
    p->algo = result.algo;
  } catch (...) {
    delete p;
    throw;
  }
  plans_.emplace(key, p);
  return *p;
}

void MxGemm::run(const std::uint8_t* a, const std::uint8_t* a_sf, const std::uint8_t* w, const std::uint8_t* w_sf,
                 __nv_bfloat16* c, int ldc, int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || K % 128 != 0) throw std::invalid_argument("MxGemm: bad shape");
  Plan& p = plan(M, N, K, ldc);
  check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sf, sizeof(w_sf)), "a scale");
  check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_sf, sizeof(a_sf)), "b scale");
  const float alpha = 1.f, beta = 0.f;
  check(cublasLtMatmul(handle_, p.op, &alpha, w, p.w, a, p.a, &beta, c, p.c, c, p.c, &p.algo, workspace_,
                       WORKSPACE_BYTES, stream),
        "matmul");
}

}  // namespace ingine::kernels::bdraft
