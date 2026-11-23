#include "kernels/batch/dense.h"

#include <cublasLt.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ingine::kernels::batch {

namespace {

void check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("batch TunedGemm: cuBLASLt ") + what + " failed (" +
                             std::to_string(int(status)) + ")");
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("batch TunedGemm: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

// a cuBLASLt algorithm
struct TunedGemm::Cand {
  cublasLtMatmulAlgo_t algo{};
};

// one algorithm group: the candidates and their timings
struct TunedGemm::Group {
  std::vector<Cand> cands;
  std::vector<double> best;  // least time measured, ms
  std::vector<int> timed, issued;
  std::vector<bool> dropped;
  int version = 0;  // bumped when timings change
  int steps = 0;    // collect() calls since its first run
  bool used = false;
};

// one exact shape: descriptors, the candidates valid for it, the algorithm in use
struct TunedGemm::Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  Group* group = nullptr;
  std::vector<bool> valid;  // per candidate (cublasLtMatmulAlgoCheck)
  int best = -1, best_version = -1, first = -1;  // first: the first valid
  ~Plan() {
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

int TunedGemm::bucket(int M) {
  if (M <= 8) return 8;
  if (M <= 64) return (M + 15) / 16 * 16;
  if (M <= 256) return (M + 31) / 32 * 32;
  return (M + 63) / 64 * 64;
}

TunedGemm::TunedGemm() {
  check(cublasLtCreate(&handle_), "create");
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaMalloc(&workspace_, kWorkspaceBytes) != cudaSuccess) {
    cublasLtDestroy(handle_);
    throw std::runtime_error("batch TunedGemm: workspace allocation failed");
  }
}

TunedGemm::~TunedGemm() {
  for (const Timing& t : pending_) cudaEventDestroy(t.start), cudaEventDestroy(t.stop);
  for (cudaEvent_t e : events_) cudaEventDestroy(e);
  for (auto& [k, p] : plans_) delete p;
  for (auto& [k, g] : groups_) delete g;
  cudaFree(workspace_);
  cublasLtDestroy(handle_);
}

namespace {

// Row-major C[M, N] = A[M, K] . B[N, K]^T is column-major C^T[N, M] = B^T (op T on the K x N matrix B) . A (the
// K x M matrix A): cuBLAS's A is B, its B is A.
void make_layouts(int M, int N, int K, int lda, int ldb, int ldc, bool c_fp32, int count, std::int64_t sa,
                  std::int64_t sb, std::int64_t sc, cublasLtMatrixLayout_t& la, cublasLtMatrixLayout_t& lb,
                  cublasLtMatrixLayout_t& lc) {
  check(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, ldb), "layout a");
  check(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, lda), "layout b");
  check(cublasLtMatrixLayoutCreate(&lc, c_fp32 ? CUDA_R_32F : CUDA_R_16BF, N, M, ldc), "layout c");
  if (count > 1) {
    const std::pair<cublasLtMatrixLayout_t, std::int64_t> ls[] = {{la, sb}, {lb, sa}, {lc, sc}};
    for (const auto& [l, stride] : ls) {
      check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)), "count");
      check(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
            "stride");
    }
  }
}

}  // namespace

TunedGemm::Plan& TunedGemm::plan(const Key& key) {
  if (auto it = plans_.find(key); it != plans_.end()) return *it->second;
  const auto [M, N, K, lda, ldb, ldc, c_fp32, count, sa, sb, sc] = key;
  auto* p = new Plan;
  try {
    check(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    make_layouts(M, N, K, lda, ldb, ldc, c_fp32, count, sa, sb, sc, p->la, p->lb, p->lc);
    p->group = &group(key, *p);
    for (const Cand& c : p->group->cands) {
      cublasLtMatmulHeuristicResult_t r{};
      const bool ok = cublasLtMatmulAlgoCheck(handle_, p->op, p->la, p->lb, p->lc, p->lc, &c.algo, &r) ==
                          CUBLAS_STATUS_SUCCESS &&
                      r.workspaceSize <= kWorkspaceBytes;
      if (ok && p->first < 0) p->first = int(p->valid.size());
      p->valid.push_back(ok);
    }
    if (p->first < 0) throw std::runtime_error("batch TunedGemm: no valid cuBLASLt algorithm");
  } catch (...) {
    delete p;
    throw;
  }
  plans_.emplace(key, p);
  return *p;
}

// cuBLASLt's configurable kernel family: the first algorithm id with split-K and the 64x64 and 256x64 tiles
int TunedGemm::configurable(bool c_fp32) {
  int& id = configurable_[c_fp32 ? 1 : 0];
  if (id != -2) return id;
  id = -1;
  const cudaDataType_t ct = c_fp32 ? CUDA_R_32F : CUDA_R_16BF;
  int ids[64], n = 0;
  if (cublasLtMatmulAlgoGetIds(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, 64, ids, &n) !=
      CUBLAS_STATUS_SUCCESS)
    return id;
  for (int i = 0; i < n && id < 0; ++i) {
    cublasLtMatmulAlgo_t a;
    if (cublasLtMatmulAlgoInit(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, ids[i], &a) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    std::size_t bytes = 0;
    int split = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &split, sizeof(split), &bytes);
    if (!split || cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &bytes) !=
                      CUBLAS_STATUS_SUCCESS)
      continue;
    std::vector<int> tiles(bytes / sizeof(int));
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), bytes, &bytes);
    auto has = [&](int t) { return std::find(tiles.begin(), tiles.end(), t) != tiles.end(); };
    if (has(CUBLASLT_MATMUL_TILE_64x64) && has(CUBLASLT_MATMUL_TILE_256x64)) id = ids[i];
  }
  return id;
}

// The heuristic's candidates at bucket(M), with the plan's descriptors otherwise (strides of whole M-row
// matrices scaled to the bucket's M); the configurable family's 64x64, 128x64, 128x128 and 256x64 tiles (weight
// rows x activation rows), split to fill the GPU once when unsplit they leave half of it idle, each with the first
// stage count of a fixed preference that it supports.
TunedGemm::Group& TunedGemm::group(const Key& key, const Plan& p) {
  auto [M, N, K, lda, ldb, ldc, c_fp32, count, sa, sb, sc] = key;
  const int Mb = bucket(M);
  const Key gkey{Mb, N, K, lda, ldb, 0, c_fp32, count, 0, sb, 0};  // the same work for any C rows' stride
  if (auto it = groups_.find(gkey); it != groups_.end()) return *it->second;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> results(kCandidates);
  int found = 0;
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
  auto scaled = [&](std::int64_t s, int ld) { return s == std::int64_t(M) * ld ? std::int64_t(Mb) * ld : s; };
  try {
    make_layouts(Mb, N, K, lda, ldb, ldc, c_fp32, count, scaled(sa, lda), sb, scaled(sc, ldc), la, lb, lc);
    st = cublasLtMatmulPreferenceCreate(&pref);
    const std::size_t ws = kWorkspaceBytes;
    const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;  // FP32 split-K sums
    if (st == CUBLAS_STATUS_SUCCESS)
      st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    if (st == CUBLAS_STATUS_SUCCESS)
      st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                                sizeof(reduction));
    if (st == CUBLAS_STATUS_SUCCESS)
      st = cublasLtMatmulAlgoGetHeuristic(handle_, p.op, la, lb, lc, lc, pref, kCandidates, results.data(), &found);
  } catch (...) {
    st = CUBLAS_STATUS_INTERNAL_ERROR;
  }
  auto* g = new Group;
  for (int i = 0; i < found; ++i) g->cands.push_back({results[std::size_t(i)].algo});
  if (const int id = st == CUBLAS_STATUS_SUCCESS ? configurable(c_fp32) : -1; id >= 0) {
    const cudaDataType_t ct = c_fp32 ? CUDA_R_32F : CUDA_R_16BF;
    struct Tile {
      int id, rows_b, rows_a;
    };
    for (const Tile t : {Tile{CUBLASLT_MATMUL_TILE_64x64, 64, 64}, Tile{CUBLASLT_MATMUL_TILE_128x64, 128, 64},
                         Tile{CUBLASLT_MATMUL_TILE_128x128, 128, 128}, Tile{CUBLASLT_MATMUL_TILE_256x64, 256, 64}}) {
      const std::int64_t base =
          std::int64_t((N + t.rows_b - 1) / t.rows_b) * ((Mb + t.rows_a - 1) / t.rows_a) * count;
      const int splits = 2 * base <= sms_ ? int(std::min<std::int64_t>(std::min(16, K / 256), sms_ / base)) : 1;
      for (const int stages : {CUBLASLT_MATMUL_STAGES_64x3, CUBLASLT_MATMUL_STAGES_32x4, CUBLASLT_MATMUL_STAGES_32x6,
                               CUBLASLT_MATMUL_STAGES_64x4, CUBLASLT_MATMUL_STAGES_64x2}) {
        cublasLtMatmulAlgo_t a;
        if (cublasLtMatmulAlgoInit(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, id, &a) !=
            CUBLAS_STATUS_SUCCESS)
          break;
        const std::uint32_t scheme = splits > 1 ? CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE : CUBLASLT_REDUCTION_SCHEME_NONE;
        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &t.id, sizeof(t.id));
        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof(splits));
        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme, sizeof(scheme));
        cublasLtMatmulHeuristicResult_t r{};
        if (cublasLtMatmulAlgoCheck(handle_, p.op, la, lb, lc, lc, &a, &r) == CUBLAS_STATUS_SUCCESS &&
            r.workspaceSize <= kWorkspaceBytes) {
          g->cands.push_back({a});
          break;
        }
      }
    }
  }
  if (pref) cublasLtMatmulPreferenceDestroy(pref);
  if (la) cublasLtMatrixLayoutDestroy(la);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  if (st != CUBLAS_STATUS_SUCCESS || found == 0) {
    delete g;
    check(st, "heuristic");
    throw std::runtime_error("batch TunedGemm: no cuBLASLt algorithm");
  }
  const std::size_t n = g->cands.size();
  g->best.assign(n, 0.0);
  g->timed.assign(n, 0);
  g->issued.assign(n, 0);
  g->dropped.assign(n, false);
  groups_.emplace(gkey, g);
  return *g;
}

void TunedGemm::run(const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32,
                    int M, int N, int K, cudaStream_t stream, const Strided& batch) {
  if (M <= 0 || N <= 0 || K <= 0 || batch.count <= 0) throw std::invalid_argument("batch TunedGemm: empty shape");
  const Strided bs = batch.count > 1 ? batch : Strided{};
  Plan& p = plan(Key{M, N, K, lda, ldb, ldc, c_fp32, bs.count, bs.a, bs.b, bs.c});
  Group& g = *p.group;
  const int n = int(g.cands.size());
  auto ok = [&](int i) { return bool(p.valid[i]); };
  if (!g.used) g.used = true, used_.push_back(&g);
  // a candidate still to time (fewest runs first; once the group ran in kWarmSteps steps), else the fastest measured
  int cand = -1;
  if (g.steps >= kWarmSteps)
    for (int i = 0; i < n; ++i)
      if (ok(i) && !g.dropped[i] && g.issued[i] < kSamples && (cand < 0 || g.issued[i] < g.issued[cand])) cand = i;
  const bool timed = cand >= 0;
  if (cand >= 0) {
    ++g.issued[cand];
  } else {
    if (p.best_version != g.version) {
      p.best = -1;
      for (int i = 0; i < n; ++i)
        if (p.valid[i] && g.timed[i] > 0 && (p.best < 0 || g.best[i] < g.best[p.best])) p.best = i;
      p.best_version = g.version;
    }
    cand = p.best >= 0 ? p.best : p.first;  // no timings yet: the heuristic's first
  }
  Timing t{&g, cand, nullptr, nullptr};
  if (timed) {
    for (cudaEvent_t* e : {&t.start, &t.stop}) {
      if (events_.empty()) {
        cuda_check(cudaEventCreate(e), "event create");
      } else {
        *e = events_.back();
        events_.pop_back();
      }
    }
    pending_.push_back(t);
    cuda_check(cudaEventRecord(t.start, stream), "event record");
  }
  const float alpha = 1.f, beta = 0.f;
  check(cublasLtMatmul(handle_, p.op, &alpha, B, p.la, A, p.lb, &beta, C, p.lc, C, p.lc, &g.cands[cand].algo,
                       workspace_, kWorkspaceBytes, stream),
        "matmul");
  if (timed) cuda_check(cudaEventRecord(t.stop, stream), "event record");
}

void TunedGemm::collect() {
  for (Group* g : used_) ++g->steps, g->used = false;
  used_.clear();
  for (const Timing& t : pending_) {
    cuda_check(cudaEventSynchronize(t.stop), "event synchronize");
    float ms = 0.f;
    cuda_check(cudaEventElapsedTime(&ms, t.start, t.stop), "event elapsed");
    double& best = t.group->best[t.cand];
    best = t.group->timed[t.cand]++ == 0 ? ms : std::min(best, double(ms));
    ++t.group->version;
    events_.push_back(t.start);
    events_.push_back(t.stop);
  }
  // drop the candidates slower than kDrop times the fastest (twice as slow after one timing: a first run may
  // include a module load)
  for (const Timing& t : pending_) {
    Group& g = *t.group;
    double fastest = -1;
    for (std::size_t i = 0; i < g.cands.size(); ++i)
      if (g.timed[i] > 0 && (fastest < 0 || g.best[i] < fastest)) fastest = g.best[i];
    for (std::size_t i = 0; i < g.cands.size(); ++i)
      if (g.timed[i] > 0 && g.best[i] > (g.timed[i] > 1 ? kDrop : 2.f) * fastest) g.dropped[i] = true;
  }
  pending_.clear();
}

}  // namespace ingine::kernels::batch
