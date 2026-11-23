// Expert GEMMs of the batched step over NVFP4 / MXFP8 experts decoded on the fly, and their routing (moe.h).
//
// Design. Few tokens per expert (T = 64: ~4, T = 512: ~32; real routing is skewed, one expert gets most tokens), so
// the weights are the M side of the MMA (mma.m16n8k16, 16 weight rows x 8 tokens): a tile of n <= TT tokens costs
// ceil(n / 8) MMAs per 16 x 16 weights instead of the prefill kernel's ceil(n / 16) per 8 x 16. A work item is ROWS
// weight rows (warps of 16 rows) x one tile's tokens over the whole K, so each weight element is decoded once per
// tile, straight from the staged checkpoint codes into A fragments in registers:
//   NVFP4: bf16((e2m1 * e4m3 block scale) * factor)      (model.decode_experts; e2m1 * scale is exact in FP16)
//   MXFP8: bf16(e4m3 * 2^(e8m0 - 127))                    (model.decode_experts)
// K order. Inside every 64-element step, thread c (lane & 3) of a quad owns the contiguous elements 16c .. 16c + 15,
// and its four k16 MMAs take them four at a time: MMA j's logical k {2c, 2c + 1, 2c + 8, 2c + 9} are physical
// 16c + 4j + {0, 1, 2, 3}, for the weights and the activations alike (a permutation of the k sum). A thread thus
// reads 8 code bytes (NVFP4; 16 MXFP8) and one block scale per weight row and 32 contiguous activation bytes per
// token per 64 k.
// Row order. A rows g and g + 8 of a warp's m16 tile are weight rows 2g and 2g + 1 of its 16: w13's gate and up
// rows of the same i (interleaved 2i / 2i + 1), or adjacent output channels of w2, land in the same thread.
// Work. A CTA per work item: ROWS weight rows (warps of 16 rows) of one tile (item = tile * NE / ROWS + row block,
// tiles in expert order). Each step is one chunk of 128 code bytes per weight row (whole L2 lines: half-line pieces
// cost ~15% of DRAM bandwidth; WK = 256 k NVFP4, 128 k MXFP8) with the activation rows of the item's tokens for the
// same k, in two shared-memory slots: one in use, one in flight. 16-byte cp.async (weights evict-first: read once per
// step), one commit group and one barrier per step; two CTAs per SM, so light (decode-bound) and heavy (MMA-bound)
// items mix. Each thread loads the block scales of its own two rows for the chunk in flight into registers. Inside a
// step, the decoding of 64-k step h + 1 is spread between the MMAs of step h (in-order issue would otherwise keep the
// tensor pipe idle while a warp decodes); tiles of one or two n8 blocks spread their MMAs over 4 or 2 accumulators
// (independent chains, summed at the end). Token rows past n are neither loaded nor stored (an output column depends
// only on its own token column).
// Launch chain: topk_kernel, route_kernel, moe_gate_up, moe_down, each a programmatic dependent of the one before; a
// kernel lets the next one launch once it has passed its own wait.
#include "kernels/batch/moe.h"

#include "kernels/common/l2.cuh"
#include "kernels/prefill/common.cuh"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace ingine::kernels::batch {
namespace {

using prefill::kExperts;
using prefill::kHidden;
using prefill::kInter;
using prefill::kTopK;
using prefill::MoeWork;

enum class Mode { Up, Down };
constexpr int FMT_NVFP4 = 0, FMT_MXFP8 = 1;
constexpr int WB = 128;  // code bytes per weight row per chunk

// ROWS: weight rows per item (warps of 16 rows: 2 ROWS threads); TT: tokens per tile
template <int FMT, int K, int TT, int ROWS_>
struct Cfg {
  static constexpr int ROWS = ROWS_, NT = 2 * ROWS_;
  static constexpr int CODE_ROW = FMT == FMT_NVFP4 ? K / 2 : K;    // checkpoint bytes per weight row
  static constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : K / 32;
  static constexpr int WK = FMT == FMT_NVFP4 ? 2 * WB : WB;        // k per chunk (step)
  static constexpr int KC = (K + WK - 1) / WK;                     // chunks (NVFP4 K = 1152: the last one half)
  static constexpr int KTAIL = K % WK;                             // k of a partial last chunk (0: none)
  static constexpr int ACH = WK / 8;                               // 16-byte activation pieces per token row
  static constexpr int AI = TT * ACH / NT;                         // activation pieces per thread
  static constexpr int W_SLOT = ROWS * WB, A_SLOT = TT * WK * 2;
  static constexpr int OFF_A = 2 * W_SLOT;
  static constexpr int SMEM = OFF_A + 2 * A_SLOT;
  static_assert(KTAIL % 64 == 0 && TT * ACH % NT == 0 && SMEM <= 101376);
};

struct Args {
  const uint8_t* codes = nullptr;   // expert codes [128, NE, ..]
  const uint8_t* scales = nullptr;  // block scales [128, NE, ..]
  const float* factors = nullptr;   // NVFP4 per-expert factor
  const uint16_t* a = nullptr;      // activations (bf16): x [T, H] (Up) or act [T * 8, I] by sorted pair (Down)
  MoeWork w;
  uint16_t* out = nullptr;
};

// Byte offset of 16-byte piece `ch` of weight row `row` in a chunk slot (128-byte rows). The piece index is XORed
// so that the reads of one instruction spread over all banks: NVFP4 (8-byte reads, half-warp = rows 0, 2, 4, 6 or
// 1, 3, 5, 7 at the same piece pair) by row bits 1-2; MXFP8 (16-byte reads, quarter-warp = rows 0, 2 or 1, 3 at
// the same 64-byte half) by row bit 1.
template <int FMT>
__device__ __forceinline__ int w_off(int row, int ch) {
  const int x = FMT == FMT_NVFP4 ? ((row >> 1) & 3) << 1 : (row & 2) << 1;
  return row * WB + ((ch ^ x) << 4);
}

// Byte offset of 16-byte piece `ch` of activation row `r` in a step slot (WK * 2-byte rows): odd rows swap the
// pieces of each pair, so the 32-byte reads of tokens 2m and 2m + 1 (one quarter-warp) hit distinct banks.
template <int WK>
__device__ __forceinline__ int a_off(int r, int ch) { return r * (WK * 2) + ((ch ^ (r & 1)) << 4); }

// two E2M1 codes (low byte of w, element 0 in the low nibble) -> packed f16 pair (exact)
__device__ __forceinline__ uint32_t e2m1x2_to_half2(uint32_t w) {
  uint32_t d;
  asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(d) : "r"(w));
  return d;
}

// two E4M3 codes (low 16 bits of w) -> packed f16 pair (exact)
__device__ __forceinline__ uint32_t e4m3x2_to_half2(uint32_t w) {
  uint32_t d;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(d) : "r"(w));
  return d;
}

// RN pack of two floats into a bf16 pair (low = lo): one F2FP.PACK_AB (prefill::pack_bf16x2's two separate
// conversions run on the slow F2F path and were the NVFP4 decode's bottleneck)
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// NVFP4 byte -> bf16 pair: bf16((e2m1 * s) * factor); s2 = the block scale in both f16 halves (the f16 product is
// the exact FP32 one: at most 5 significant bits, |p| in [2^-10, 2688])
__device__ __forceinline__ uint32_t dec_nvfp4(uint32_t w, uint32_t s2, float factor) {
  uint32_t h = e2m1x2_to_half2(w);
  const __half2 p = __hmul2(*reinterpret_cast<const __half2*>(&h), *reinterpret_cast<const __half2*>(&s2));
  const float2 f = __half22float2(p);
  return pack_bf16x2(f.x * factor, f.y * factor);
}

// MXFP8 pair (low 16 bits of w) -> bf16 pair: bf16(e4m3 * scale)
__device__ __forceinline__ uint32_t dec_mxfp8(uint32_t w, float s) {
  uint32_t h = e4m3x2_to_half2(w);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  return pack_bf16x2(f.x * s, f.y * s);
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// Block scales of one weight row for one chunk, as loaded: NVFP4 16 bytes (64 k each; 8 in a half last chunk),
// MXFP8 4 bytes (32 k each).
template <int FMT>
struct ChunkScales;
template <>
struct ChunkScales<FMT_NVFP4> {
  uint2 lo, hi;
};
template <>
struct ChunkScales<FMT_MXFP8> {
  uint32_t w;
};

// One work item as this thread sees it: tile (expert, first pair, pairs), first weight row, and the activation row
// of each of the thread's cp.async pieces (-1: past the tile's pairs).
template <int AI>
struct Item {
  int e = 0, pb = 0, n = 0, n0 = 0;
  int arow[AI];
};

// Up: A = w13 (NE = 2I), activations x rows of the pairs' tokens (K = H), out act [pairs, I].
// Down: A = w2 (NE = H), activations act rows (K = I), out down [T * 8, H] at pair_dst.
template <int FMT, Mode MODE, int K, int TT, int ROWS_>
__global__ void __launch_bounds__(2 * ROWS_) expert_kernel(const Args g) {
  using C = Cfg<FMT, K, TT, ROWS_>;
  constexpr int KC = C::KC, WK = C::WK, ACH = C::ACH, AI = C::AI, ROWS = C::ROWS, NT = C::NT;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;
  constexpr int NBLK = NE / ROWS;
  constexpr int HS = WK / 64;  // 64-k steps per chunk
  static_assert(NE % ROWS == 0);
  extern __shared__ __align__(128) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, c = lane & 3;
  const int row0 = warp * 16 + 2 * gq;  // this thread's weight rows row0 + k2 = A row gq + 8 k2 of the warp's tile

  if constexpr (MODE == Mode::Up) {
    // route_kernel's tiles: wait for it, then let moe_down launch (moe_down reads them before its own wait)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  // item = tile * NBLK + row block; the grid covers the most tiles T can have, the ones past tiles_n exit
  if (int(blockIdx.x) >= *g.w.tiles_n * NBLK) return;
  Item<AI> it;
  {
    const int4 tile = g.w.tiles[blockIdx.x / NBLK];
    it.e = tile.x, it.pb = tile.y, it.n = tile.z, it.n0 = (blockIdx.x % NBLK) * ROWS;
  }

  const uint64_t pol = l2::policy<l2::Evict::first>();
  ChunkScales<FMT> sc_next[2];  // scales of the chunk in flight
  uint32_t sc[2];               // scales of the chunk in use: NVFP4 byte j = this thread's block of 64-k step j

  // chunk q into slot s: codes and this thread's scales (registers)
  auto load_w = [&](int q, int s) {
    const uint8_t* codes = g.codes + (int64_t(it.e) * NE + it.n0) * C::CODE_ROW + q * WB;
    unsigned char* ws = smem + s * C::W_SLOT;
#pragma unroll
    for (int i = 0; i < ROWS * (WB / 16) / NT; ++i) {
      const int id = tid + i * NT, row = id / (WB / 16), ch = id % (WB / 16);
      if (C::CODE_ROW % WB == 0 || q * WB + ch * 16 < C::CODE_ROW)
        l2::cp_async16<l2::Evict::first>(ws + w_off<FMT>(row, ch), codes + int64_t(row) * C::CODE_ROW + ch * 16, pol);
    }
    const uint8_t* srow = g.scales + (int64_t(it.e) * NE + it.n0 + row0) * C::SCALE_ROW;
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
      if constexpr (FMT == FMT_NVFP4) {  // rows of 72 scale bytes (K = 1152) are 8-byte aligned
        const uint2* p = reinterpret_cast<const uint2*>(srow + k2 * C::SCALE_ROW + q * 16);
        sc_next[k2].lo = __ldg(p);
        if (C::SCALE_ROW % 16 == 0 || q * 16 + 8 < C::SCALE_ROW) sc_next[k2].hi = __ldg(p + 1);
      } else {
        sc_next[k2].w = __ldg(reinterpret_cast<const uint32_t*>(srow + k2 * C::SCALE_ROW + q * 4));
      }
    }
  };
  // chunk q's activation rows into slot s
  auto load_a = [&](int q, int s) {
    unsigned char* as = smem + C::OFF_A + s * C::A_SLOT;
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      const int r = (tid + i * NT) / ACH, ch = (tid + i * NT) % ACH;
      if (it.arow[i] >= 0 && (C::KTAIL == 0 || q < KC - 1 || ch < C::KTAIL / 8))
        prefill::cp_async16(as + a_off<WK>(r, ch), g.a + size_t(it.arow[i]) * K + q * WK + ch * 8);
    }
  };
  // the chunk in flight becomes the chunk in use: NVFP4 keeps byte c of each 4-byte group (its 16-k blocks)
  auto take_scales = [&] {
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
      if constexpr (FMT == FMT_NVFP4) {
        const uint32_t sel = c | (4 + c) << 4;
        sc[k2] = __byte_perm(__byte_perm(sc_next[k2].lo.x, sc_next[k2].lo.y, sel),
                             __byte_perm(sc_next[k2].hi.x, sc_next[k2].hi.y, sel), 0x5410);
      } else {
        sc[k2] = sc_next[k2].w;
      }
    }
  };

  // chunk 0's codes first, ahead of the pair-list reads
  load_w(0, 0);
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = (tid + i * NT) / ACH;
    if (r >= it.n) it.arow[i] = -1;
    else if constexpr (MODE == Mode::Up) it.arow[i] = g.w.pair_token[it.pb + r];
    else it.arow[i] = it.pb + r;
  }
  if constexpr (MODE == Mode::Down) {
    // act is moe_gate_up's output (a loop ptxas cannot see through keeps the copies above ahead of the wait)
#pragma unroll 1
    for (int i = 0; i <= (it.n >> 30); ++i) asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  load_a(0, 0);
  prefill::cp_commit();
  const float factor = FMT == FMT_NVFP4 ? __ldg(g.factors + it.e) : 1.f;

  auto run = [&]<int NB>() {
    // J accumulators per n8 block (k16 MMA j into j % J): independent MMA chains for small tiles (the fp32 sums
    // are added up at the end, a different order of the k sum)
    constexpr int J = NB == 1 ? 4 : NB == 2 ? 2 : 1;
    float acc[NB][J][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int jj = 0; jj < J; ++jj) acc[nb][jj][0] = acc[nb][jj][1] = acc[nb][jj][2] = acc[nb][jj][3] = 0.f;

#pragma unroll 1
    for (int q = 0; q < KC; ++q) {
      const int slot = q & 1;
      prefill::cp_wait<0>();
      __syncthreads();
      take_scales();
      if (q + 1 < KC) load_w(q + 1, slot ^ 1), load_a(q + 1, slot ^ 1);
      prefill::cp_commit();
      const unsigned char* W = smem + slot * C::W_SLOT;
      const unsigned char* A = smem + C::OFF_A + slot * C::A_SLOT;
      const int hn = C::KTAIL && q == KC - 1 ? C::KTAIL / 64 : HS;  // 64-k steps in this chunk

      // codes (NVFP4 8 bytes, MXFP8 16) and block scale of 64-k step h for this thread's rows
      auto fetch = [&](int h, uint32_t (&rw)[2][4], uint32_t (&sv)[2]) {
#pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
          if constexpr (FMT == FMT_NVFP4) {
            const uint2 wd =
                *reinterpret_cast<const uint2*>(W + w_off<FMT>(row0 + k2, h * 2 + (c >> 1)) + (c & 1) * 8);
            rw[k2][0] = wd.x, rw[k2][1] = wd.y;
            sv[k2] = e4m3x2_to_half2(((sc[k2] >> (8 * h)) & 0xffu) * 0x0101u);
          } else {
            const uint4 w4 = *reinterpret_cast<const uint4*>(W + w_off<FMT>(row0 + k2, h * 4 + c));
            rw[k2][0] = w4.x, rw[k2][1] = w4.y, rw[k2][2] = w4.z, rw[k2][3] = w4.w;
            const uint32_t sb = (sc[k2] >> (8 * (h * 2 + (c >> 1)))) & 0xffu;
            sv[k2] = sb ? sb << 23 : 0x00400000u;  // 2^(sb - 127)
          }
        }
      };
      // decode unit u of 16 (NVFP4 a code byte, MXFP8 a code pair) of row row0 + (u >> 3): A register
      // a[b >> 1][2 (b & 1) + k2], b = u & 7
      auto decode = [&](const uint32_t (&rw)[2][4], const uint32_t (&sv)[2], int u, unsigned (&a)[4][4]) {
        const int k2 = u >> 3, b = u & 7;
        if constexpr (FMT == FMT_NVFP4)
          a[b >> 1][2 * (b & 1) + k2] = dec_nvfp4(rw[k2][b >> 2] >> (8 * (b & 3)), sv[k2], factor);
        else
          a[b >> 1][2 * (b & 1) + k2] = dec_mxfp8(rw[k2][b >> 1] >> (16 * (b & 1)), __uint_as_float(sv[k2]));
      };
      // the 4 NB MMAs of 64-k step h, each followed by its share of the 16 decode units of step h + 1 (in-order
      // issue would otherwise leave the tensor pipe idle while a warp decodes)
      unsigned a[2][4][4];
      uint32_t rw[2][2][4], sv[2][2];
      fetch(0, rw[0], sv[0]);
#pragma unroll
      for (int u = 0; u < 16; ++u) decode(rw[0], sv[0], u, a[0]);
#pragma unroll
      for (int h = 0; h < HS; ++h) {
        if (h >= hn) break;  // partial last chunk
        const int cur = h & 1;
        const bool more = h + 1 < hn;
        if (more) fetch(h + 1, rw[cur ^ 1], sv[cur ^ 1]);
        uint4 bq[NB][2];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
          const int r = nb * 8 + gq;
          bq[nb][0] = *reinterpret_cast<const uint4*>(A + a_off<WK>(r, h * 8 + 2 * c));
          bq[nb][1] = *reinterpret_cast<const uint4*>(A + a_off<WK>(r, h * 8 + 2 * c + 1));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nb = 0; nb < NB; ++nb) {
            const uint4& b = bq[nb][j >> 1];
            if (j & 1) prefill::mma_bf16(acc[nb][j % J], a[cur][j], b.z, b.w);
            else prefill::mma_bf16(acc[nb][j % J], a[cur][j], b.x, b.y);
            const int s = j * NB + nb;
#pragma unroll
            for (int u = s * 16 / (4 * NB); u < (s + 1) * 16 / (4 * NB); ++u)
              if (more) decode(rw[cur ^ 1], sv[cur ^ 1], u, a[cur ^ 1]);
          }
      }
    }

#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int tok = nb * 8 + 2 * c + p;
        if (tok >= it.n) continue;
        auto sum = [&](int r) {  // (a0 + a1) + (a2 + a3)
          if constexpr (J == 4) return (acc[nb][0][r] + acc[nb][1][r]) + (acc[nb][2][r] + acc[nb][3][r]);
          else if constexpr (J == 2) return acc[nb][0][r] + acc[nb][1][r];
          else return acc[nb][0][r];
        };
        const float c0 = sum(p), c1 = sum(2 + p);  // weight rows n0 + row0 and n0 + row0 + 1
        if constexpr (MODE == Mode::Up) {
          const float gt = prefill::round_bf16(c0), u = prefill::round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(it.pb + tok) * kInter + (it.n0 + row0) / 2] = prefill::f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[it.pb + tok];
          *reinterpret_cast<uint32_t*>(g.out + size_t(dst) * kHidden + it.n0 + row0) = pack_bf16x2(c0, c1);
        }
      }
  };

  // the n8 block count is a template constant: a runtime bound would leave predicated MMAs in the tensor pipe
  switch ((it.n + 7) / 8) {
    case 1: run.template operator()<1>(); break;
    case 2: run.template operator()<2>(); break;
    case 3: if constexpr (TT >= 32) run.template operator()<3>(); break;
    default: if constexpr (TT >= 32) run.template operator()<4>(); break;
  }
  prefill::cp_wait<0>();
}

// Router selection, warp per token (prefill::moe_topk's arithmetic): top-8 of s + bias (descending, lowest expert on
// ties), s = sigmoid; wts = 8 s / (sum s + 1e-20). No counts: route_kernel counts the pairs itself.
__global__ void __launch_bounds__(256) topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                   int T, int* __restrict__ idx, float* __restrict__ wts) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int t = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  float sc[4], key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    sc[i] = 1.f / (1.f + expf(-logits[size_t(t) * kExperts + e]));
    key[i] = sc[i] + bias[e];
  }
  int sel_e[kTopK];
  float sel_s[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -INFINITY;
    int be = kExperts;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (key[i] > best) best = key[i], be = lane + 32 * i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
    }
    sel_e[k] = be;
    float s_be = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i == be) s_be = sc[i], key[i] = -INFINITY;
    sel_s[k] = __shfl_sync(0xffffffffu, s_be, be & 31);
  }
  if (lane == 0) {
    float sum = sel_s[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) sum += sel_s[k];
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      idx[size_t(t) * kTopK + k] = sel_e[k];
      wts[size_t(t) * kTopK + k] = sel_s[k] / den * 8.f;
    }
  }
}

// One CTA, a programmatic dependent of topk_kernel: pair counts, expert offsets, tiles of TR pairs in expert order,
// scatter of the pairs into expert order. Lets moe_gate_up's CTAs launch at once (they wait for this kernel before
// reading anything).
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ idx, int T, int TR, MoeWork w) {
  __shared__ int cnt[kExperts], offs[kExperts], tile0[kExperts], cursor[kExperts];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < kExperts) cnt[tid] = 0, cursor[tid] = 0;
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");  // idx
  for (int p = tid; p < T * kTopK; p += blockDim.x) atomicAdd(&cnt[idx[p]], 1);
  __syncthreads();
  // warp 0: exclusive scans of the pair and tile counts over the experts (4 per lane)
  if (warp == 0) {
    int c4[4], pairs = 0, tiles = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c4[i] = cnt[4 * lane + i];
      pairs += c4[i], tiles += (c4[i] + TR - 1) / TR;
    }
    int p_incl = pairs, t_incl = tiles;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int p = __shfl_up_sync(0xffffffffu, p_incl, o), tt = __shfl_up_sync(0xffffffffu, t_incl, o);
      if (lane >= o) p_incl += p, t_incl += tt;
    }
    int p = p_incl - pairs, tl = t_incl - tiles;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int e = 4 * lane + i;
      offs[e] = p, tile0[e] = tl;
      w.counts[e] = c4[i];
      p += c4[i], tl += (c4[i] + TR - 1) / TR;
    }
    if (lane == 31) *w.tiles_n = t_incl;
  }
  __syncthreads();
  if (tid < kExperts) {
    const int c = cnt[tid];
    for (int i = 0; i * TR < c; ++i) w.tiles[tile0[tid] + i] = make_int4(tid, offs[tid] + i * TR, min(TR, c - i * TR), 0);
  }
  for (int p = tid; p < T * kTopK; p += blockDim.x) {
    const int e = idx[p];
    const int pos = offs[e] + atomicAdd(&cursor[e], 1);
    w.pair_token[pos] = p / kTopK;
    w.pair_dst[pos] = p;
  }
}

cudaLaunchAttribute pdl_attribute() {
  cudaLaunchAttribute pdl{};
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  return pdl;
}

// One CTA per possible work item (the ones past the routed tiles exit at once).
template <int FMT, Mode MODE, int K, int TT, int ROWS>
void launch(const Args& g, int T, cudaStream_t stream) {
  using C = Cfg<FMT, K, TT, ROWS>;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;
  auto* kernel = expert_kernel<FMT, MODE, K, TT, ROWS>;
  static const cudaError_t attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  if (attr != cudaSuccess) throw std::runtime_error("batch moe: shared memory attribute failed");
  const int tiles = (T * kTopK + TT - 1) / TT + kExperts;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(unsigned(tiles * (NE / ROWS)));
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute pdl = pdl_attribute();  // a dependent of route_kernel (gate/up) or moe_gate_up (down)
  cfg.attrs = &pdl, cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, kernel, g) != cudaSuccess) throw std::runtime_error("batch moe: launch failed");
}

// Per tile size, one chunk per step, two CTAs per SM: TT = 16 in 128-row items (48 KB NVFP4, 40 KB MXFP8), TT = 32
// in 64-row items (48 KB, 32 KB).
template <int FMT, Mode MODE, int K>
void dispatch(const Args& g, int T, cudaStream_t stream) {
  if (moe_tile_rows(T) == 16) launch<FMT, MODE, K, 16, 128>(g, T, stream);
  else launch<FMT, MODE, K, 32, 64>(g, T, stream);
}

Args expert_args(const void* codes, const void* scales, const float* factors, const __nv_bfloat16* a, const MoeWork& w,
                 __nv_bfloat16* out) {
  Args g;
  g.codes = static_cast<const uint8_t*>(codes);
  g.scales = static_cast<const uint8_t*>(scales);
  g.factors = factors;
  g.a = reinterpret_cast<const uint16_t*>(a);
  g.w = w;
  g.out = reinterpret_cast<uint16_t*>(out);
  return g;
}

}  // namespace

int moe_tile_rows(int T) {
  static const int forced = [] {
    const char* e = std::getenv("INGINE_BATCH_MOE_TILE");
    const int v = e ? std::atoi(e) : 0;
    return v == 16 || v == 32 ? v : 0;
  }();
  return forced ? forced : T <= 128 ? 16 : 32;
}

int moe_tiles_capacity(int max_T) {
  int cap = 0;
  for (int tr : {16, 32}) cap = std::max(cap, (max_T * kTopK + tr - 1) / tr + kExperts);
  return cap;
}

void moe_select(const float* logits, const float* bias, int T, int* idx, float* wts, const MoeWork& w,
                cudaStream_t stream) {
  topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(logits, bias, T, idx, wts);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(1024);
  cfg.stream = stream;
  cudaLaunchAttribute pdl = pdl_attribute();
  cfg.attrs = &pdl, cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, route_kernel, static_cast<const int*>(idx), T, moe_tile_rows(T), w) != cudaSuccess)
    throw std::runtime_error("batch moe: route launch failed");
}

void moe_gate_up(const prefill::MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream) {
  const Args g = expert_args(e.w13, e.w13_scale, e.w13_factor, x, w, act);
  if (e.fmt == 0) dispatch<FMT_NVFP4, Mode::Up, kHidden>(g, T, stream);
  else dispatch<FMT_MXFP8, Mode::Up, kHidden>(g, T, stream);
}

void moe_down(const prefill::MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream) {
  const Args g = expert_args(e.w2, e.w2_scale, e.w2_factor, act, w, down);
  if (e.fmt == 0) dispatch<FMT_NVFP4, Mode::Down, kInter>(g, T, stream);
  else dispatch<FMT_MXFP8, Mode::Down, kInter>(g, T, stream);
}

}  // namespace ingine::kernels::batch
