// Batched DSpark Markov head: per-row top-1024 candidates of the shortlist logits (a CTA per row), the 7 steps of
// the recurrent cell and candidate logits of every sequence (a cluster of two CTAs per sequence), and the draft
// outputs. The arithmetic of kernels/draft/markov.cu.
#include "kernels/batch/draft.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <cooperative_groups.h>

#include <stdexcept>

namespace ingine::kernels::bdraft {
namespace {

using namespace ::ingine::kernels::draft;

// ---- top-k ---------------------------------------------------------------------------------------------
// A CTA per row; a thread takes 8 consecutive elements (one 16-byte load) of each span of TOPK_SPAN. On the
// BF16 values' 16-bit order keys, an 8-bit radix select in two levels (warp-aggregated shared histograms) finds
// the 1024th key T, then an ordered compaction (a block scan of per-thread counts per span) writes the elements
// above T and the first ties, by ascending index.

constexpr int TOPK_THREADS = 512;
constexpr int TOPK_SPAN = 8 * TOPK_THREADS;

__device__ __forceinline__ uint32_t bf16_key(uint16_t b) { return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u); }
__device__ __forceinline__ float key_value(uint32_t k) {
  const uint32_t b = (k & 0x8000u) ? (k & 0x7fffu) : (~k & 0xffffu);
  return __uint_as_float(b << 16);
}

// Warp 0: the bin holding the k-th largest key of hist[256] (bins scanned from the top), and the count of keys
// in the bins above it plus above0.
__device__ __forceinline__ void select_bin(const uint32_t* hist, int k, int above0, int* sh_bin, int* sh_above) {
  if (threadIdx.x >= 32) return;
  const int lane = threadIdx.x;
  int c[8], s = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += (c[j] = int(hist[8 * lane + j]));
  int incl = s;  // keys in the bins of lanes >= lane
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_down_sync(0xffffffffu, incl, o);
    if (lane + o < 32) incl += u;
  }
  int above = above0 + incl - s;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    if (above < k && above + c[j] >= k) {
      *sh_bin = 8 * lane + j;
      *sh_above = above;
    }
    above += c[j];
  }
}

// the order keys of elements i .. i + 7 of row x (i a multiple of 8, the row's stride too); valid: how many are
// below S
__device__ __forceinline__ int load_keys(const uint16_t* x, int i, int S, uint32_t (&k)[8]) {
  const int valid = max(0, min(8, S - i));
  const uint4 v = valid > 0 ? *reinterpret_cast<const uint4*>(x + i) : make_uint4(0, 0, 0, 0);
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    k[2 * j] = bf16_key(uint16_t(w[j] & 0xffffu));
    k[2 * j + 1] = bf16_key(uint16_t(w[j] >> 16));
  }
  return valid;
}

// hist[bin] += 1 for every lane's bin < 256, one atomic per distinct bin of the warp
__device__ __forceinline__ void warp_hist(uint32_t* hist, uint32_t bin) {
  const unsigned same = __match_any_sync(0xffffffffu, bin);
  if (bin < 256u && (threadIdx.x & 31) == __ffs(same) - 1) atomicAdd(&hist[bin], uint32_t(__popc(same)));
}

__global__ void __launch_bounds__(TOPK_THREADS) topk_kernel(const uint16_t* base, int S, int ld, float* cand_val,
                                                            std::int32_t* cand_idx) {
  __shared__ uint32_t hist[256];
  __shared__ int sh_bin, sh_above;
  __shared__ uint32_t warp_tot[TOPK_THREADS / 32];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint16_t* x = base + size_t(row) * ld;
  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int i0 = 0; i0 < S; i0 += TOPK_SPAN) {
    uint32_t k[8];
    const int valid = load_keys(x, i0 + 8 * tid, S, k);
#pragma unroll
    for (int j = 0; j < 8; ++j) warp_hist(hist, j < valid ? k[j] >> 8 : 256u);
  }
  __syncthreads();
  select_bin(hist, CANDIDATES, 0, &sh_bin, &sh_above);
  __syncthreads();
  const uint32_t b1 = sh_bin;
  const int above1 = sh_above;
  __syncthreads();
  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int i0 = 0; i0 < S; i0 += TOPK_SPAN) {
    uint32_t k[8];
    const int valid = load_keys(x, i0 + 8 * tid, S, k);
#pragma unroll
    for (int j = 0; j < 8; ++j) warp_hist(hist, j < valid && (k[j] >> 8) == b1 ? k[j] & 255u : 256u);
  }
  __syncthreads();
  select_bin(hist, CANDIDATES, above1, &sh_bin, &sh_above);
  __syncthreads();
  const uint32_t T = (b1 << 8) | uint32_t(sh_bin);
  const int n_gt = sh_above;
  const int n_tie = CANDIDATES - n_gt;  // ties taken
  int c_gt = 0, c_tie = 0;              // seen in earlier spans (block-uniform)
  float* vout = cand_val + size_t(row) * CANDIDATES;
  std::int32_t* iout = cand_idx + size_t(row) * CANDIDATES;
  for (int i0 = 0; i0 < S && (c_gt < n_gt || c_tie < n_tie); i0 += TOPK_SPAN) {
    const int i = i0 + 8 * tid;
    uint32_t k[8];
    const int valid = load_keys(x, i, S, k);
    uint32_t mine = 0;  // above T (low 16 bits) and ties (high 16 bits) among this thread's elements
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (j < valid) mine += k[j] > T ? 1u : (k[j] == T ? 0x10000u : 0u);
    uint32_t incl = mine;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (lane == 31) warp_tot[warp] = incl;
    __syncthreads();
    uint32_t before = incl - mine, total = 0;
    for (int u = 0; u < TOPK_THREADS / 32; ++u) {
      if (u < warp) before += warp_tot[u];
      total += warp_tot[u];
    }
    int rg = c_gt + int(before & 0xffffu), rt = c_tie + int(before >> 16);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j >= valid) break;
      if (k[j] > T) {
        vout[rg] = key_value(k[j]);
        iout[rg++] = i + j;
      } else if (k[j] == T) {
        if (rt < n_tie) {
          vout[n_gt + rt] = key_value(k[j]);
          iout[n_gt + rt] = i + j;
        }
        ++rt;
      }
    }
    c_gt += int(total & 0xffffu);
    c_tie += int(total >> 16);
    __syncthreads();  // warp_tot is rewritten by the next span
  }
}

// ---- Markov head -------------------------------------------------------------------------------------------
// A cluster of two CTAs per sequence runs the 7 steps: CTA r computes projection rows [384 r, 384 r + 384) and
// candidates [512 r, 512 r + 512) of each step; the raw projection and the partial (max, sum, pick) of the
// candidates go to both CTAs through distributed shared memory, so both run the cell and take the same pick.

constexpr int LOOP_THREADS = 1024;                                // 32 warps per CTA
constexpr int HALF_ROWS = RNN_OUT / 2;                            // projection rows per CTA
constexpr int ROWS_PER_WARP = HALF_ROWS / (LOOP_THREADS / 32);    // 12
constexpr int HALF_CANDS = CANDIDATES / 2;                        // candidates per CTA
constexpr int CANDS_PER_WARP = HALF_CANDS / (LOOP_THREADS / 32);  // 16: lanes 0..15 own one each
static_assert(HALF_ROWS % (LOOP_THREADS / 32) == 0 && HALF_CANDS % (LOOP_THREADS / 32) == 0 && RANK == 256);

struct LoopArgs {
  const __nv_bfloat16 *rnn_w, *rnn_b, *w1, *fold_w, *fold_b;
  const float *part_h, *part_f, *cand_val;
  const std::int32_t *cand_idx, *map;
};

struct Partial {  // a CTA's candidates of a step: max logit, sum of exp(logit - max), packed pick
  float m, s;
  unsigned long long pick;
};

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(LOOP_THREADS, 1)
    markov_loop_kernel(int gamma, DraftSeqs S, LoopArgs A, float T, std::uint64_t seed, std::int32_t* tokens,
                       float* conf) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ __align__(16) __nv_bfloat16 z[2 * RANK];  // [state_i, W1[prev]]
  __shared__ __align__(16) __nv_bfloat16 lat[RANK];
  __shared__ float raw[RNN_OUT];
  __shared__ Partial warp_part[LOOP_THREADS / 32];
  __shared__ Partial part_of[2];  // by CTA rank
  __shared__ int sh_prev;
  const int rank = int(cluster.block_rank()), b = blockIdx.x >> 1;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float* raw_peer = cluster.map_shared_rank(raw, rank ^ 1);
  Partial* part_peer = cluster.map_shared_rank(part_of, rank ^ 1);
  if (tid < RANK) z[tid] = __float2bfloat16_rn(0.f);  // state_0 = 0
  if (tid == 0) sh_prev = S.anchor[b];
  const bool owner = lane < CANDS_PER_WARP;
  const int k = rank * HALF_CANDS + warp * CANDS_PER_WARP + lane;  // owners' candidate
  for (int i = 0; i < gamma; ++i) {
    // this step's inputs that do not depend on the previous pick, in flight across the barriers
    const size_t row0 = (size_t(b) * gamma + i) * CANDIDATES;
    const int my_c = owner ? A.cand_idx[row0 + k] : 0;
    const float my_val = owner ? A.cand_val[row0 + k] : 0.f;
    const float my_fb = bf(A.fold_b[my_c]);
    float part[3] = {0.f, 0.f, 0.f}, bias[3] = {0.f, 0.f, 0.f};
    if (tid < RANK) {
#pragma unroll
      for (int g = 0; g < 3; ++g) {
        const int row = g * RANK + tid;
        part[g] = A.part_h[(size_t(b) * gamma + i) * RNN_OUT + row] + A.part_f[size_t(b) * RNN_OUT + row];
        bias[g] = bf(A.rnn_b[row]);
      }
    }
    __syncthreads();  // z's state half, sh_prev
    if (tid < RANK) z[RANK + tid] = A.w1[size_t(sh_prev) * RANK + tid];
    __syncthreads();
    // raw[row] = rnn_w[row, :512] . z (state | W1[prev] halves as kernels::draft::markov_step sums them), this
    // CTA's rows, written to both CTAs
    {
      float z0[8], z1[8];
      unpack8(*reinterpret_cast<const uint4*>(z + 8 * lane), z0);
      unpack8(*reinterpret_cast<const uint4*>(z + RANK + 8 * lane), z1);
#pragma unroll 4
      for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
        const int row = rank * HALF_ROWS + warp + (LOOP_THREADS / 32) * rr;
        const __nv_bfloat16* wr = A.rnn_w + size_t(row) * RNN_IN;
        float w0[8], w1v[8];
        unpack8(*reinterpret_cast<const uint4*>(wr + 8 * lane), w0);
        unpack8(*reinterpret_cast<const uint4*>(wr + RANK + 8 * lane), w1v);
        float acc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc = fmaf(w0[e], z0[e], fmaf(w1v[e], z1[e], acc));
        acc = warp_sum(acc);
        if (lane == 0) raw[row] = raw_peer[row] = acc;
      }
    }
    cluster.sync();
    // the cell: state_{i+1}, latent (both CTAs, the same values)
    if (tid < RANK) {
      const int j = tid;
      float r[3];
#pragma unroll
      for (int g = 0; g < 3; ++g) r[g] = bf_round(__fadd_rn(bf_round(raw[g * RANK + j] + part[g]), bias[g]));
      const float gate = bf_round(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-r[0]))));
      const float cell = bf_round(tanhf(r[1]));
      const float s = bf(z[j]);
      const float gs = bf_round(__fmul_rn(gate, s));
      const float om = bf_round(__fsub_rn(1.f, gate));
      const float oc = bf_round(__fmul_rn(om, cell));
      z[j] = __float2bfloat16_rn(__fadd_rn(gs, oc));
      lat[j] = __float2bfloat16_rn(tanhf(r[2]));
    }
    __syncthreads();
    // candidate logits: warp w computes its 16 candidates (a warp per row of fold_w), lane u < 16 keeps one
    float logit = -INFINITY;
    {
      float l[8];
      unpack8(*reinterpret_cast<const uint4*>(lat + 8 * lane), l);
      float mine = 0.f;
#pragma unroll 8
      for (int u = 0; u < CANDS_PER_WARP; ++u) {
        const int c = __shfl_sync(0xffffffffu, my_c, u);
        float w[8];
        unpack8(*reinterpret_cast<const uint4*>(A.fold_w + size_t(c) * RANK + 8 * lane), w);
        float acc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc = fmaf(w[e], l[e], acc);
        acc = warp_sum(acc);
        if (lane == u) mine = acc;
      }
      if (owner) {
        const float bb = bf_round(__fadd_rn(my_fb, bf_round(mine)));
        const float t = bf_round(__fadd_rn(my_val, bb));
        logit = bf_round(__fmul_rn(30.f, tanhf(__fdiv_rn(t, 30.f))));
      }
    }
    // this CTA's (max, sum of exp(logit - max), pick): the pick = argmax of the logits or of logits / T + Gumbel
    // noise, lowest candidate on ties; per warp, then over the warps
    float score = logit;
    if (T > 0.f && owner)
      score = __fadd_rn(__fdiv_rn(logit, fmaxf(T, 1e-6f)), gumbel(seed, std::uint64_t(b), uint32_t(i), uint32_t(k)));
    unsigned long long p = owner ? pack_argmax(score, uint32_t(k)) : 0ull;
    const float mw = warp_max(logit);
    const float sw = warp_sum(owner ? expf(logit - mw) : 0.f);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) p = max(p, __shfl_xor_sync(0xffffffffu, p, o));
    if (lane == 0) warp_part[warp] = Partial{mw, sw, p};
    __syncthreads();
    if (tid == 0) {
      Partial q = warp_part[0];
      for (int w = 1; w < LOOP_THREADS / 32; ++w) {
        const Partial x = warp_part[w];
        const float mm = fmaxf(q.m, x.m);
        q = Partial{mm, q.s * expf(q.m - mm) + x.s * expf(x.m - mm), max(q.pick, x.pick)};
      }
      part_of[rank] = part_peer[rank] = q;
    }
    cluster.sync();
    if (tid == 0) {
      const Partial a = part_of[0], c = part_of[1];
      const float mm = fmaxf(a.m, c.m);
      const float ss = a.s * expf(a.m - mm) + c.s * expf(c.m - mm);
      const int token = A.map[A.cand_idx[row0 + unpack_argmax_index(max(a.pick, c.pick))]];
      if (rank == 0) {
        tokens[b * GAMMA + i] = token;
        conf[b * GAMMA + i] = 1.f / ss;
      }
      sh_prev = token;
    }
  }
  cluster.sync();  // no CTA exits while its peer may still write to its shared memory
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

}  // namespace

void topk_candidates(const __nv_bfloat16* base, int rows, int S, int ld, float* cand_val, std::int32_t* cand_idx,
                     cudaStream_t stream) {
  if (S < CANDIDATES || S > ld || ld % 8 != 0 || reinterpret_cast<std::uintptr_t>(base) % 16 != 0)
    throw std::invalid_argument("bdraft::topk_candidates: S/ld/alignment");
  if (rows <= 0) return;
  topk_kernel<<<rows, TOPK_THREADS, 0, stream>>>(reinterpret_cast<const uint16_t*>(base), S, ld, cand_val, cand_idx);
  check_launch();
}

void markov_draft(int n, int gamma, const DraftSeqs& seqs, const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b,
                  const __nv_bfloat16* w1, const float* part_h, const float* part_f, const float* cand_val,
                  const std::int32_t* cand_idx, const std::int32_t* map, const __nv_bfloat16* fold_w,
                  const __nv_bfloat16* fold_b, float temperature, std::uint64_t seed, std::int32_t* tokens, float* conf,
                  cudaStream_t stream) {
  if (n <= 0) return;
  if (gamma < 1 || gamma > GAMMA) throw std::invalid_argument("bdraft::markov_draft: gamma must be in 1..7");
  const LoopArgs A{rnn_w, rnn_b, w1, fold_w, fold_b, part_h, part_f, cand_val, cand_idx, map};
  markov_loop_kernel<<<2 * n, LOOP_THREADS, 0, stream>>>(gamma, seqs, A, temperature, seed, tokens, conf);
  check_launch();
}

}  // namespace ingine::kernels::bdraft
