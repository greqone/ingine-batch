// Batched DSpark block attention: kernels/draft/attention.cu's tensor-core kernel with a sequence dimension.
// Sequence b's keys are its slot's ring rows [0, nring) followed by its 8 block rows (virtual key index
// nring + i); chunks of chunk_keys keys are reduced by a combine pass per (sequence, kv head, query).
#include "kernels/batch/draft.h"
#include "kernels/draft/draft.h"
#include "kernels/batch/mx_device.cuh"
#include "kernels/draft/draft_device.cuh"

#include <algorithm>
#include <stdexcept>

namespace ingine::kernels::bdraft {
namespace {

using namespace ::ingine::kernels::draft;

constexpr int QH = HEADS / KV_HEADS;  // 3 query heads per kv head
constexpr int NQ = QH * BLOCK;        // 24 queries per kv head: qi = head * 8 + row
constexpr int ATT_TILE = 16;
constexpr int ATT_THREADS = 128;
constexpr int MAX_CHUNKS = 128;       // one combine thread each
constexpr int TARGET_CHUNKS = 128;    // sequences x chunks per call (about)
static_assert(QH == 3 && BLOCK == 8 && HEAD_DIM == 128 && MAX_CHUNKS <= HEAD_DIM);

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* row) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* row) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* row) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}
__device__ __forceinline__ uint4 ld_cache(const void* p) {
  uint4 r;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
}

// ring rows sequence b may read: local rings only hold positions < P in rows < P until they fill
__device__ __forceinline__ int ring_keys(const Ring& R, int P) { return min(P, R.rows); }

template <bool FP8>
struct TileRegs {
  static constexpr int NL = FP8 ? 1 : 2;
  uint4 k[NL], v[NL];
  int cp;
};

// grid (chunks, KV_HEADS, n), block 128: the sequence's first Rb of the 8 query rows (the others zero, their
// outputs unused). ws_o fp32 [n, chunks, 8, 24, 128], ws_ml fp32 [n, chunks, 8, 24, 2].
template <bool LOCAL, bool FP8>
__global__ void __launch_bounds__(ATT_THREADS) attn_partial_kernel(const __nv_bfloat16* q, int Rb, DraftSeqs S, Ring R,
                                                                  const uint8_t* kb, const uint8_t* vb, int chunk_keys,
                                                                  float* ws_o, float* ws_ml) {
  constexpr int ROW_BYTES = FP8 ? HEAD_DIM : 2 * HEAD_DIM;
  constexpr int NL = TileRegs<FP8>::NL;
  __shared__ __align__(16) __nv_bfloat16 qs[QH][BLOCK][HEAD_DIM];
  __shared__ __align__(16) __nv_bfloat16 kt[2][ATT_TILE][HEAD_DIM];
  __shared__ __align__(16) __nv_bfloat16 vt[2][ATT_TILE][HEAD_DIM];
  __shared__ __align__(16) __nv_bfloat16 pt[QH][BLOCK][ATT_TILE];
  __shared__ unsigned vmask[2];
  const int chunk = blockIdx.x, g = blockIdx.y, b = blockIdx.z, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int gl = lane >> 2, t = lane & 3, lm = lane >> 3, lr = lane & 7;
  const int P = S.P[b], slot = S.slot[b];
  const int nring = ring_keys(R, P), nkeys = nring + Rb;
  const int k0 = chunk * chunk_keys, kend = min(k0 + chunk_keys, nkeys);
  if (k0 >= nkeys) return;  // the combine reads only this sequence's chunks
  constexpr float scale = 0.08838834764831845f;
  // this sequence's slot, kv head g: rows [0, R.rows) of 128 values (head-major rings)
  const uint8_t* kr = static_cast<const uint8_t*>(R.k) + (size_t(slot) * KV_HEADS + g) * R.rows * ROW_BYTES;
  const uint8_t* vr = static_cast<const uint8_t*>(R.v) + (size_t(slot) * KV_HEADS + g) * R.rows * ROW_BYTES;
  const uint8_t* kbb = kb + size_t(b) * Rb * KV_HEADS * ROW_BYTES;
  const uint8_t* vbb = vb + size_t(b) * Rb * KV_HEADS * ROW_BYTES;
  const std::int32_t* rpos = LOCAL ? R.pos + size_t(slot) * R.rows : nullptr;

  auto load = [&](TileRegs<FP8>& x, int t0) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = t0 + idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      const bool live = key < kend;
      const bool ring = key < nring;
      const size_t at = (ring ? size_t(key) * ROW_BYTES : (size_t(key - nring) * KV_HEADS + g) * ROW_BYTES) + 16 * piece;
      x.k[j] = live ? ld_cache((ring ? kr : kbb) + at) : make_uint4(0, 0, 0, 0);
      x.v[j] = live ? ld_cache((ring ? vr : vbb) + at) : make_uint4(0, 0, 0, 0);
    }
    x.cp = 0;
    if constexpr (LOCAL) {
      if (tid < ATT_TILE && t0 + tid < nring) x.cp = rpos[t0 + tid];
    }
  };
  TileRegs<FP8> ra, rb;
  if (k0 < kend) load(ra, k0);
  if (k0 + ATT_TILE < kend) load(rb, k0 + ATT_TILE);

  {
    uint4 qv[QH];
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j, hh = idx >> 7, r = (idx >> 4) & 7, piece = idx & 15;
      qv[j] = r < Rb ? *reinterpret_cast<const uint4*>(q + ((size_t(b) * Rb + r) * HEADS + g * QH + hh) * HEAD_DIM +
                                                         8 * piece)
                     : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j;
      *reinterpret_cast<uint4*>(&qs[idx >> 7][(idx >> 4) & 7][8 * (idx & 15)]) = qv[j];
    }
  }
  __syncthreads();
  const bool head = w < QH;
  const int hw = head ? w : 0;
  uint32_t qb[HEAD_DIM / 16][2];
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) ldmatrix_x2(qb[s], &qs[hw][lr][16 * s + 8 * (lm & 1)]);

  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float oacc[HEAD_DIM / 16][4];
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d)
#pragma unroll
    for (int i = 0; i < 4; ++i) oacc[d][i] = 0.f;

  auto tile = [&](TileRegs<FP8>& x, int t0, int buf) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      if constexpr (FP8) {
        const uint32_t kw[4] = {x.k[j].x, x.k[j].y, x.k[j].z, x.k[j].w}, vw[4] = {x.v[j].x, x.v[j].y, x.v[j].z, x.v[j].w};
        uint32_t kk[8], vv[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          kk[2 * i] = e4m3x2_to_bf16x2(kw[i]);
          kk[2 * i + 1] = e4m3x2_to_bf16x2(kw[i] >> 16);
          vv[2 * i] = e4m3x2_to_bf16x2(vw[i]);
          vv[2 * i + 1] = e4m3x2_to_bf16x2(vw[i] >> 16);
        }
        uint4* kd = reinterpret_cast<uint4*>(&kt[buf][key][16 * piece]);
        uint4* vd = reinterpret_cast<uint4*>(&vt[buf][key][16 * piece]);
        kd[0] = make_uint4(kk[0], kk[1], kk[2], kk[3]);
        kd[1] = make_uint4(kk[4], kk[5], kk[6], kk[7]);
        vd[0] = make_uint4(vv[0], vv[1], vv[2], vv[3]);
        vd[1] = make_uint4(vv[4], vv[5], vv[6], vv[7]);
      } else {
        *reinterpret_cast<uint4*>(&kt[buf][key][8 * piece]) = x.k[j];
        *reinterpret_cast<uint4*>(&vt[buf][key][8 * piece]) = x.v[j];
      }
    }
    if (w == 0) {
      const int j = t0 + lane;
      bool v = lane < ATT_TILE && j < kend;
      if constexpr (LOCAL) {
        if (j < nring) v = v && x.cp >= 0 && x.cp < P && x.cp >= P - WINDOW;
      }
      const unsigned mask = __ballot_sync(0xffffffffu, v);
      if (lane == 0) vmask[buf] = mask;
    }
    __syncthreads();
    if (t0 + 2 * ATT_TILE < kend) load(x, t0 + 2 * ATT_TILE);
    if (!head) return;

    float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
      uint32_t a[4];
      ldmatrix_x4(a, &kt[buf][8 * (lm & 1) + lr][16 * s + 8 * (lm >> 1)]);
      mma_bf16(sc, a, qb[s]);
    }
    const unsigned mask = vmask[buf];
    const bool valid[2] = {((mask >> gl) & 1u) != 0, ((mask >> (gl + 8)) & 1u) != 0};
    float al[2];
    __nv_bfloat16 pb16[2][2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float xs[2], mt = -INFINITY;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        xs[kk] = valid[kk] ? sc[2 * kk + e] * scale : -INFINITY;
        mt = fmaxf(mt, xs[kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, sh));
      const float m_new = fmaxf(m_run[e], mt);
      al[e] = m_new == m_run[e] ? 1.f : (m_run[e] == -INFINITY ? 0.f : __expf(m_run[e] - m_new));
      float ps = 0.f;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        pb16[e][kk] = __float2bfloat16_rn(valid[kk] ? __expf(xs[kk] - m_new) : 0.f);
        ps += __bfloat162float(pb16[e][kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, sh);
      l_run[e] = l_run[e] * al[e] + ps;
      m_run[e] = m_new;
    }
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) pt[w][2 * t + e][gl + 8 * kk] = pb16[e][kk];
    __syncwarp();
    uint32_t pf[2];
    ldmatrix_x2(pf, &pt[w][lr][8 * (lm & 1)]);
    const bool rescale = __any_sync(0xffffffffu, al[0] != 1.f || al[1] != 1.f);
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      uint32_t a[4];
      ldmatrix_x4_trans(a, &vt[buf][8 * (lm >> 1) + lr][16 * d + 8 * (lm & 1)]);
      mma_bf16(oacc[d], a, pf);
    }
  };
  for (int t0 = k0; t0 < kend; t0 += 2 * ATT_TILE) {
    tile(ra, t0, 0);
    if (t0 + ATT_TILE < kend) tile(rb, t0 + ATT_TILE, 1);
  }
  if (!head) return;

  const size_t base = (size_t(b) * gridDim.x + chunk) * KV_HEADS + g;
  const int qi = BLOCK * w + 2 * t;
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d) {
    float* at = ws_o + (base * NQ + qi) * HEAD_DIM + 16 * d + gl;
    at[0] = oacc[d][0];
    at[HEAD_DIM] = oacc[d][1];
    at[8] = oacc[d][2];
    at[HEAD_DIM + 8] = oacc[d][3];
  }
  if (gl == 0) {
    float* at = ws_ml + (base * NQ + qi) * 2;
    *reinterpret_cast<float4*>(at) = make_float4(m_run[0], l_run[0], m_run[1], l_run[1]);
  }
}

// grid (KV_HEADS * QH * Rb, n), block 128: query qi = head * 8 + row (row < Rb); thread c weighs chunk c, then
// every thread sums its dim over the chunks
__global__ void __launch_bounds__(HEAD_DIM) attn_combine_kernel(const float* ws_o, const float* ws_ml, int chunks,
                                                               int chunk_keys, int Rb, DraftSeqs S, Ring R,
                                                               __nv_bfloat16* out, Mx out_mx) {
  __shared__ float scratch[33];
  __shared__ float f[HEAD_DIM];
  const int r = blockIdx.x % Rb, hh = (blockIdx.x / Rb) % QH, g = blockIdx.x / (Rb * QH);
  const int qi = hh * BLOCK + r, b = blockIdx.y, d = threadIdx.x;
  const int mine = (ring_keys(R, S.P[b]) + Rb + chunk_keys - 1) / chunk_keys;  // this sequence's chunks
  const size_t seq0 = size_t(b) * chunks;
  float mc = -INFINITY, lc = 0.f;
  if (d < mine) {
    const float2 ml = *reinterpret_cast<const float2*>(ws_ml + (((seq0 + d) * KV_HEADS + g) * NQ + qi) * 2);
    mc = ml.x;
    lc = ml.y;
  }
  const float m = block_max(mc, scratch);
  const float fc = mc == -INFINITY ? 0.f : __expf(mc - m);
  f[d] = fc;
  const float l = block_sum(lc * fc, scratch);
  float o = 0.f;
  for (int c = 0; c < mine; ++c) o += ws_o[(((seq0 + c) * KV_HEADS + g) * NQ + qi) * HEAD_DIM + d] * f[c];
  const int h = g * QH + hh;
  const float y = bf_round(l > 0.f ? o / l : 0.f);
  const size_t row = size_t(b) * Rb + r;
  if (out != nullptr) out[row * Q_DIM + h * HEAD_DIM + d] = __float2bfloat16_rn(y);
  if (out_mx.q != nullptr) mx_store1(y, out_mx, Q_DIM, int(row), h * HEAD_DIM + d);
}

// chunks per sequence for n sequences: about TARGET_CHUNKS CTAs per kv head in all
int chunk_target(int n) { return std::clamp((TARGET_CHUNKS + n - 1) / n, 1, MAX_CHUNKS); }

}  // namespace

std::size_t attention_workspace_floats(int max_seqs) {
  std::size_t most = 0;  // n x chunks <= n x chunk_target(n)
  for (int n = 1; n <= max_seqs; ++n) most = std::max(most, std::size_t(n) * chunk_target(n));
  return most * KV_HEADS * NQ * (HEAD_DIM + 2);
}

void block_attention(const __nv_bfloat16* q, int n, int Rb, const DraftSeqs& seqs, const Ring& ring, const void* kb,
                     const void* vb, int max_keys, float* ws, __nv_bfloat16* out, Mx out_mx, cudaStream_t stream) {
  if (n <= 0) return;
  if (Rb < 2 || Rb > BLOCK) throw std::invalid_argument("bdraft::block_attention: R must be in 2..8");
  if (max_keys <= Rb) throw std::invalid_argument("bdraft::block_attention: max_keys must exceed the block");
  const int target = chunk_target(n);
  const int chunk_keys = std::max(2 * ATT_TILE, (max_keys + target - 1) / target + 2 * ATT_TILE - 1) / (2 * ATT_TILE) *
                         (2 * ATT_TILE);
  const int chunks = (max_keys + chunk_keys - 1) / chunk_keys;
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(n) * chunks * KV_HEADS * NQ * HEAD_DIM;
  const dim3 grid(chunks, KV_HEADS, n), block(ATT_THREADS);
  const auto* k = static_cast<const uint8_t*>(kb);
  const auto* v = static_cast<const uint8_t*>(vb);
  if (ring.pos != nullptr) {
    if (ring.fp8) throw std::invalid_argument("bdraft::block_attention: local rings are BF16");
    attn_partial_kernel<true, false><<<grid, block, 0, stream>>>(q, Rb, seqs, ring, k, v, chunk_keys, ws_o, ws_ml);
  } else if (ring.fp8) {
    attn_partial_kernel<false, true><<<grid, block, 0, stream>>>(q, Rb, seqs, ring, k, v, chunk_keys, ws_o, ws_ml);
  } else {
    attn_partial_kernel<false, false><<<grid, block, 0, stream>>>(q, Rb, seqs, ring, k, v, chunk_keys, ws_o, ws_ml);
  }
  CUDA_CHECK(cudaGetLastError());
  attn_combine_kernel<<<dim3(KV_HEADS * QH * Rb, n), HEAD_DIM, 0, stream>>>(ws_o, ws_ml, chunks, chunk_keys, Rb, seqs,
                                                                            ring, out, out_mx);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::bdraft
