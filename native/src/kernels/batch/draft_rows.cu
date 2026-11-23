// Batched DSpark drafter: row kernels (context commit, block input, q/k/v, residual stream). The arithmetic of
// kernels/draft/block.cu per sequence; rows of a sequence go to and come from its slot.
#include "kernels/batch/draft.h"
#include "kernels/draft/draft.h"
#include "kernels/batch/mx_device.cuh"
#include "kernels/draft/draft_device.cuh"

#include <stdexcept>

namespace ingine::kernels::bdraft {
namespace {

using namespace ::ingine::kernels::draft;

constexpr int ROW_THREADS = H / 8;  // 288: a thread owns 8 consecutive elements of a row
static_assert(H % 8 == 0 && ROW_THREADS % 32 == 0);

__device__ __forceinline__ uint4 pack8(const float* f) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    w[i] = *reinterpret_cast<const uint32_t*>(&b);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ float rms_scale(const float (&x)[8], float* scratch) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) ss += x[j] * x[j];
  return rsqrtf(block_sum(ss, scratch) / float(H) + EPS);
}

// bf16(rms(x) * w) of this thread's 8 values
__device__ __forceinline__ uint4 rms_norm(const float (&x)[8], const uint4& wv, float* scratch) {
  const float rs = rms_scale(x, scratch);
  float w[8], y[8];
  unpack8(wv, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(x[j], rs), w[j]);
  return pack8(y);
}

// mix_rmsnorm of this thread's 8 columns of a feats row [4, H]
__device__ __forceinline__ uint4 mix_row(const __nv_bfloat16* f, int c, const float (&a)[4], const uint4& wv,
                                         float* scratch) {
  float m[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float x[8];
    unpack8(*reinterpret_cast<const uint4*>(f + j * H + c), x);
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = j == 0 ? __fmul_rn(x[i], a[0]) : __fadd_rn(m[i], __fmul_rn(x[i], a[j]));
  }
  return rms_norm(m, wv, scratch);
}

// One head row of 128 values, one per thread: bf16(rms(x) * w), then neox RoPE at pos, rounded to BF16.
__device__ __forceinline__ float norm_rope_head(float x, const __nv_bfloat16* w, int pos, float* scratch,
                                                float* pair) {
  const int d = threadIdx.x;
  const float ss = block_sum(x * x, scratch);
  const float rs = rsqrtf(ss / float(HEAD_DIM) + EPS);
  const float y = bf_round(__fmul_rn(__fmul_rn(x, rs), bf(w[d])));
  pair[d] = y;
  __syncthreads();
  const int half = HEAD_DIM / 2, i = d & (half - 1);
  const float inv = powf(ROPE_THETA, (-float(i) * 2.f) / float(HEAD_DIM));
  const float ang = __fmul_rn(float(pos), inv);
  const float c = cosf(ang), s = sinf(ang);
  const float x1 = pair[i], x2 = pair[i + half];
  const float o = d < half ? __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)) : __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  __syncthreads();
  return bf_round(o);
}

// the sequence owning commit row r: the last s with row0[s] <= r
__device__ __forceinline__ int commit_seq(const CommitSeqs& S, int r) {
  int lo = 0, hi = S.n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (S.row0[mid] <= r) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

struct MixArgs {
  const __nv_bfloat16* alpha;
  const __nv_bfloat16* norm[4];  // ctx_norm[0..2], f_norm
};

// grid (rows, 4 parts): parts 0..2 the layers' context inputs, part 3 f_rnn of a sequence's last row
__global__ void __launch_bounds__(ROW_THREADS) commit_mix_kernel(const __nv_bfloat16* feats, int row_base,
                                                                 CommitSeqs S, MixArgs A, __nv_bfloat16* xin,
                                                                 __nv_bfloat16* f_rnn, int rows) {
  __shared__ float scratch[33];
  const int r = blockIdx.x, part = blockIdx.y, c = 8 * threadIdx.x;
  const int s = commit_seq(S, row_base + r);
  const bool last = row_base + r == S.row0[s] + S.count[s] - 1;
  if (part == 3 && !last) return;
  float a[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) a[j] = bf(A.alpha[4 * part + j]);
  const uint4 wv = *reinterpret_cast<const uint4*>(A.norm[part] + c);
  const uint4 y = mix_row(feats + size_t(r) * 4 * H, c, a, wv, scratch);
  __nv_bfloat16* out = part == 3 ? f_rnn + size_t(S.slot[s]) * H : xin + (size_t(part) * rows + r) * H;
  *reinterpret_cast<uint4*>(out + c) = y;
}

struct StoreArgs {
  const __nv_bfloat16* kv[3];
  const __nv_bfloat16* k_norm[3];
  Ring ring[3];
};

// grid (rows, KV_HEADS, 3 layers), block 128
__global__ void __launch_bounds__(HEAD_DIM) commit_store_kernel(int row_base, CommitSeqs S, StoreArgs A) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  const int r = blockIdx.x, g = blockIdx.y, l = blockIdx.z, d = threadIdx.x;
  const int s = commit_seq(S, row_base + r);
  const int pos = S.start[s] + (row_base + r - S.row0[s]);
  const Ring& R = A.ring[l];
  if (S.start[s] + S.count[s] - pos > R.rows) return;  // overwritten by a later row of the commit
  const int slot = S.slot[s];
  const size_t row = size_t(slot) * R.rows + pos % R.rows;  // of R.pos
  const __nv_bfloat16* src = A.kv[l] + size_t(r) * 2 * KV_DIM;
  const float k = norm_rope_head(bf(src[g * HEAD_DIM + d]), A.k_norm[l], pos, scratch, pair);
  const size_t at = ((size_t(slot) * KV_HEADS + g) * R.rows + pos % R.rows) * HEAD_DIM + d;
  store_cache(R.k, at, k, R.fp8);
  store_cache(R.v, at, bf(src[KV_DIM + g * HEAD_DIM + d]), R.fp8);
  if (R.pos != nullptr && g == 0 && d == 0) R.pos[row] = pos;
}

struct EmbedArgs {
  const __nv_bfloat16* embed;
  const float* embed_norm;
  const __nv_bfloat16* mask_embed;
  const __nv_bfloat16* in_norm;
  const __nv_bfloat16* f_rnn;
};

// a row of H normed values (this thread's 8 as packed BF16) to the bf16 matrix x (if not null) and its Mx copy
__device__ __forceinline__ void put_row(const uint4& y, __nv_bfloat16* x, const Mx& mx, size_t row) {
  const int c = 8 * threadIdx.x;
  if (x != nullptr) *reinterpret_cast<uint4*>(x + row * H + c) = y;
  if (mx.q != nullptr) {
    float f[8];
    unpack8(y, f);
    mx_store8(f, mx, H, int(row), c, true);
  }
}

// grid (R rows, n), block 288
__global__ void __launch_bounds__(ROW_THREADS) embed_block_kernel(DraftSeqs S, EmbedArgs A, __nv_bfloat16* res,
                                                                  __nv_bfloat16* xn, Mx xn_mx, __nv_bfloat16* f_in) {
  __shared__ float scratch[33];
  const int r = blockIdx.x, b = blockIdx.y, c = 8 * threadIdx.x;
  const size_t row = size_t(b) * gridDim.x + r, at = row * H + c;
  float x[8];
  if (r == 0) {
    float v[8];
    unpack8(*reinterpret_cast<const uint4*>(A.embed + size_t(S.anchor[b]) * H + c), v);
    const float rs = rms_scale(v, scratch);
    const float4 e0 = *reinterpret_cast<const float4*>(A.embed_norm + c);
    const float4 e1 = *reinterpret_cast<const float4*>(A.embed_norm + c + 4);
    const float en[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = bf_round(__fmul_rn(__fmul_rn(v[j], rs), en[j]));
    // the Markov head's input of this draft: f_rnn of the slot
    *reinterpret_cast<uint4*>(f_in + size_t(b) * H + c) =
        *reinterpret_cast<const uint4*>(A.f_rnn + size_t(S.slot[b]) * H + c);
  } else {
    unpack8(*reinterpret_cast<const uint4*>(A.mask_embed + c), x);
  }
  *reinterpret_cast<uint4*>(res + at) = pack8(x);
  put_row(rms_norm(x, *reinterpret_cast<const uint4*>(A.in_norm + c), scratch), xn, xn_mx, row);
}

// grid (R rows, 24 q + 8 k + 8 v heads, n), block 128
__global__ void __launch_bounds__(HEAD_DIM) qkv_rope_kernel(const __nv_bfloat16* qkv, DraftSeqs S,
                                                           const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                                           __nv_bfloat16* q_out, void* kb, void* vb, bool fp8) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  const int r = blockIdx.x, h = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
  const int pos = S.P[b] + r;
  const size_t row = size_t(b) * gridDim.x + r;
  const __nv_bfloat16* x = qkv + row * QKV_ROWS;
  if (h < HEADS) {
    const float q = norm_rope_head(bf(x[h * HEAD_DIM + d]), q_norm, pos, scratch, pair);
    q_out[(row * HEADS + h) * HEAD_DIM + d] = __float2bfloat16_rn(q);
  } else if (h < HEADS + KV_HEADS) {
    const int g = h - HEADS;
    const float k = norm_rope_head(bf(x[Q_DIM + g * HEAD_DIM + d]), k_norm, pos, scratch, pair);
    store_cache(kb, (row * KV_HEADS + g) * HEAD_DIM + d, k, fp8);
  } else {
    const int g = h - HEADS - KV_HEADS;
    store_cache(vb, (row * KV_HEADS + g) * HEAD_DIM + d, bf(x[Q_DIM + KV_DIM + g * HEAD_DIM + d]), fp8);
  }
}

__global__ void __launch_bounds__(ROW_THREADS) add_rmsnorm_kernel(__nv_bfloat16* res, const __nv_bfloat16* delta,
                                                                  const __nv_bfloat16* w, __nv_bfloat16* xn, Mx xn_mx,
                                                                  int R, __nv_bfloat16* h_out, Mx h_mx) {
  __shared__ float scratch[33];
  const int r = blockIdx.x;
  const size_t at = size_t(r) * H + 8 * threadIdx.x;
  float x[8], dl[8];
  unpack8(*reinterpret_cast<const uint4*>(res + at), x);
  unpack8(*reinterpret_cast<const uint4*>(delta + at), dl);
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = bf_round(__fadd_rn(x[j], dl[j]));
  *reinterpret_cast<uint4*>(res + at) = pack8(x);
  const uint4 y = rms_norm(x, *reinterpret_cast<const uint4*>(w + 8 * threadIdx.x), scratch);
  put_row(y, xn, xn_mx, r);
  const int i = r % R;
  if (i != 0) put_row(y, h_out, h_mx, size_t(r / R) * (R - 1) + i - 1);
}

// a thread per 8 outputs (inter % 32 == 0: the 4 lanes of an Mx block share a row)
__global__ void silu_mul_kernel(const __nv_bfloat16* gu, int rows, int inter, __nv_bfloat16* out, Mx out_mx) {
  const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const bool valid = i < size_t(rows) * inter;
  const size_t r = valid ? i / inter : 0, j = valid ? i % inter : 0;
  float g[8], u[8], y[8];
  unpack8(*reinterpret_cast<const uint4*>(gu + r * 2 * inter + j), g);
  unpack8(*reinterpret_cast<const uint4*>(gu + r * 2 * inter + inter + j), u);
#pragma unroll
  for (int e = 0; e < 8; ++e) y[e] = bf_round(__fmul_rn(__fdiv_rn(g[e], __fadd_rn(1.f, expf(-g[e]))), u[e]));
  if (valid && out != nullptr) *reinterpret_cast<uint4*>(out + i) = pack8(y);
  if (out_mx.q != nullptr) mx_store8(y, out_mx, inter, int(r), int(j), valid);
}

// a thread per 8 elements of each row's column chunks c = threadIdx.x + 256 k (K % 256 == 0: whole warps)
__global__ void mx_quantize_kernel(const __nv_bfloat16* x, int K, Mx out) {
  const int r = blockIdx.x;
  for (int c = threadIdx.x; c < K / 8; c += blockDim.x) {
    float y[8];
    unpack8(*reinterpret_cast<const uint4*>(x + size_t(r) * K + 8 * c), y);
    mx_store8(y, out, K, r, 8 * c, true);
  }
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

}  // namespace

void commit_mix(const __nv_bfloat16* feats, int rows, int row_base, const CommitSeqs& seqs, const __nv_bfloat16* alpha,
                const __nv_bfloat16* const ctx_norm[3], const __nv_bfloat16* f_norm, __nv_bfloat16* xin,
                __nv_bfloat16* f_rnn, cudaStream_t stream) {
  if (rows <= 0) return;
  if (reinterpret_cast<std::uintptr_t>(feats) % 16 != 0) throw std::invalid_argument("bdraft::commit_mix: feats alignment");
  const MixArgs A{alpha, {ctx_norm[0], ctx_norm[1], ctx_norm[2], f_norm}};
  commit_mix_kernel<<<dim3(rows, 4), ROW_THREADS, 0, stream>>>(feats, row_base, seqs, A, xin, f_rnn, rows);
  check_launch();
}

void commit_store(const __nv_bfloat16* const kv[3], int rows, int row_base, const CommitSeqs& seqs,
                  const __nv_bfloat16* const k_norm[3], const Ring rings[3], cudaStream_t stream) {
  if (rows <= 0) return;
  StoreArgs A{};
  for (int l = 0; l < 3; ++l) {
    if (rings[l].rows <= 0) throw std::invalid_argument("bdraft::commit_store: empty ring");
    A.kv[l] = kv[l];
    A.k_norm[l] = k_norm[l];
    A.ring[l] = rings[l];
  }
  commit_store_kernel<<<dim3(rows, KV_HEADS, 3), HEAD_DIM, 0, stream>>>(row_base, seqs, A);
  check_launch();
}

void embed_block(int n, int R, const DraftSeqs& seqs, const __nv_bfloat16* embed, const float* embed_norm,
                 const __nv_bfloat16* mask_embed, const __nv_bfloat16* in_norm, const __nv_bfloat16* f_rnn,
                 __nv_bfloat16* res, __nv_bfloat16* xn, Mx xn_mx, __nv_bfloat16* f_in, cudaStream_t stream) {
  if (n <= 0) return;
  if (R < 2 || R > BLOCK) throw std::invalid_argument("bdraft::embed_block: R must be in 2..8");
  const EmbedArgs A{embed, embed_norm, mask_embed, in_norm, f_rnn};
  embed_block_kernel<<<dim3(R, n), ROW_THREADS, 0, stream>>>(seqs, A, res, xn, xn_mx, f_in);
  check_launch();
}

void qkv_rope(const __nv_bfloat16* qkv, int n, int R, const DraftSeqs& seqs, const __nv_bfloat16* q_norm,
              const __nv_bfloat16* k_norm, __nv_bfloat16* q_out, void* kb, void* vb, bool fp8, cudaStream_t stream) {
  if (n <= 0) return;
  if (R < 2 || R > BLOCK) throw std::invalid_argument("bdraft::qkv_rope: R must be in 2..8");
  qkv_rope_kernel<<<dim3(R, HEADS + 2 * KV_HEADS, n), HEAD_DIM, 0, stream>>>(qkv, seqs, q_norm, k_norm, q_out, kb, vb,
                                                                             fp8);
  check_launch();
}

void add_rmsnorm(__nv_bfloat16* res, const __nv_bfloat16* delta, const __nv_bfloat16* w, __nv_bfloat16* xn, Mx xn_mx,
                 int rows, int R, __nv_bfloat16* h_out, Mx h_mx, cudaStream_t stream) {
  if (rows <= 0) return;
  if (R < 2 || R > BLOCK) throw std::invalid_argument("bdraft::add_rmsnorm: R must be in 2..8");
  add_rmsnorm_kernel<<<rows, ROW_THREADS, 0, stream>>>(res, delta, w, xn, xn_mx, R, h_out, h_mx);
  check_launch();
}

void silu_mul(const __nv_bfloat16* gu, int rows, int inter, __nv_bfloat16* out, Mx out_mx, cudaStream_t stream) {
  if (inter % 32 != 0) throw std::invalid_argument("bdraft::silu_mul: inter must be a multiple of 32");
  const size_t n8 = size_t(rows) * inter / 8;
  if (n8 == 0) return;
  silu_mul_kernel<<<unsigned((n8 + 255) / 256), 256, 0, stream>>>(gu, rows, inter, out, out_mx);
  check_launch();
}

void mx_quantize(const __nv_bfloat16* x, int rows, int K, Mx out, cudaStream_t stream) {
  if (K % 256 != 0) throw std::invalid_argument("bdraft::mx_quantize: K must be a multiple of 256");
  if (rows <= 0) return;
  mx_quantize_kernel<<<rows, 256, 0, stream>>>(x, K, out);
  check_launch();
}

}  // namespace ingine::kernels::bdraft
