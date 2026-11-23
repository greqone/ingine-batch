// Expert GEMMs of the batched step over NVFP4 / MXFP8 experts decoded on the fly (moe.h).
//
// Design. Few tokens per expert (T = 64: ~4, T = 512: ~32), so the weights are the M side of the MMA
// (mma.m16n8k16, 16 weight rows x 8 tokens): a tile of n <= TT tokens costs ceil(n / 8) MMAs per 16 x 16 weights
// instead of the prefill kernel's ceil(n / 16) per 8 x 16. A CTA computes 128 weight rows (8 warps of 16 rows) x
// the tile's tokens over the whole K, so each weight element is decoded once per tile, straight from the staged
// checkpoint codes into A fragments in registers:
//   NVFP4: bf16((e2m1 * e4m3 block scale) * factor)      (model.decode_experts; e2m1 * scale is exact in FP16)
//   MXFP8: bf16(e4m3 * 2^(e8m0 - 127))                    (model.decode_experts)
// K order. Inside every 64-element step, thread c (lane & 3) of a quad owns the contiguous elements 16c .. 16c + 15,
// and its four k16 MMAs take them four at a time: MMA j's logical k {2c, 2c + 1, 2c + 8, 2c + 9} are physical
// 16c + 4j + {0, 1, 2, 3}, for the weights and the activations alike (a permutation of the k sum). A thread thus
// reads 8 code bytes (NVFP4; 16 MXFP8) and one block scale per weight row and 32 contiguous activation bytes per
// token per 64 k.
// Row order. A rows g and g + 8 of a warp's m16 tile are weight rows 2g and 2g + 1 of its 16: w13's gate and up
// rows of the same i (interleaved 2i / 2i + 1), or adjacent output channels of w2, land in the same thread.
// Pipeline. Weight codes stream in chunks of 128 bytes per row (whole L2 lines: half-line pieces cost ~15% of
// DRAM bandwidth), WK = 256 k (NVFP4) or 128 k (MXFP8), into two slots: one in use, one in flight. Activation rows
// of the tile's tokens stream in steps of SK k (one chunk, or half an NVFP4 chunk for 64-token tiles) into AS slots.
// 16-byte cp.async (weights evict-first: read once per step), one commit group and one barrier per step; a chunk is
// issued with the activation step AS - 1 = R ahead of its first use (R = WK / SK steps per chunk), so the FIFO
// group waits never hold a chunk back. Each thread loads the block scales of its own two rows for the chunk in
// flight into registers (no shared memory: two CTAs of 16-token tiles fit an SM). Token rows past n are neither
// loaded nor stored (an output column depends only on its own token column).
// moe_down is a programmatic dependent of moe_gate_up: it stages its first weight chunk before griddepcontrol.wait;
// moe_gate_up triggers it at the start of every CTA.
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
constexpr int WB = 128;                 // code bytes per weight row per chunk
constexpr int ROWS = 128, NT = 256;     // weight rows and threads per CTA: 8 warps of 16 rows

template <int FMT, int K, int TT, int SK>
struct Cfg {
  static constexpr int CODE_ROW = FMT == FMT_NVFP4 ? K / 2 : K;    // checkpoint bytes per weight row
  static constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : K / 32;
  static constexpr int WK = FMT == FMT_NVFP4 ? 2 * WB : WB;        // k per weight chunk
  static constexpr int R = WK / SK;                                // activation steps per chunk
  static constexpr int KC = (K + WK - 1) / WK;                     // chunks (NVFP4 K = 1152: the last one half)
  static constexpr int KS = (K + SK - 1) / SK;                     // activation steps
  static constexpr int KTAIL = K % SK;                             // k of a partial last step (0: none)
  static constexpr int AS = R + 1;                                 // activation slots
  static constexpr int W_SLOT = ROWS * WB, A_SLOT = TT * SK * 2;
  static constexpr int OFF_A = 2 * W_SLOT;
  static constexpr int SMEM = OFF_A + AS * A_SLOT;
  static_assert(WK % SK == 0 && KTAIL % 64 == 0 && SMEM <= 101376);
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

// Byte offset of 16-byte piece `ch` of activation row `r` in a step slot (SK * 2-byte rows): odd rows swap the
// pieces of each pair, so the 32-byte reads of tokens 2m and 2m + 1 (one quarter-warp) hit distinct banks.
template <int SK>
__device__ __forceinline__ int a_off(int r, int ch) { return r * (SK * 2) + ((ch ^ (r & 1)) << 4); }

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

// Up: A = w13 (NE = 2I), activations x rows of the pairs' tokens (K = H), out act [pairs, I].
// Down: A = w2 (NE = H), activations act rows (K = I), out down [T * 8, H] at pair_dst.
template <int FMT, Mode MODE, int K, int TT, int SK>
__global__ void __launch_bounds__(NT) expert_kernel(const Args g) {
  using C = Cfg<FMT, K, TT, SK>;
  constexpr int KS = C::KS, KC = C::KC, R = C::R, AS = C::AS;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;
  constexpr int NBLK = NE / ROWS;
  static_assert(NE % ROWS == 0);
  extern __shared__ __align__(128) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, c = lane & 3;

  if constexpr (MODE == Mode::Up) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // tiles and pair lists come from moe_route, which completed before moe_gate_up started (the tile list has room
  // for every CTA's tile: one past tiles_n is stale, never used)
  const int tile_idx = blockIdx.x / NBLK;
  const int4 tile = g.w.tiles[tile_idx];
  if (tile_idx >= *g.w.tiles_n) return;
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NBLK) * ROWS;
  const uint8_t* code_tile = g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW;
  const uint64_t pol = l2::policy<l2::Evict::first>();

  // this thread's weight rows: row0 + k2 = A row gq + 8 k2 of the warp's m16 tile
  const int row0 = warp * 16 + 2 * gq;
  const uint8_t* srow = g.scales + (int64_t(e) * NE + n0 + row0) * C::SCALE_ROW;
  ChunkScales<FMT> sc_next[2];  // scales of the chunk in flight
  uint32_t sc[2];               // scales of the chunk in use: NVFP4 byte j = this thread's block of 64-k step j

  auto load_w = [&](int chunk) {
    unsigned char* ws = smem + (chunk % 2) * C::W_SLOT;
#pragma unroll
    for (int i = 0; i < ROWS * (WB / 16) / NT; ++i) {
      const int id = tid + i * NT, row = id / (WB / 16), ch = id % (WB / 16);
      if (C::CODE_ROW % WB == 0 || chunk * WB + ch * 16 < C::CODE_ROW)
        l2::cp_async16<l2::Evict::first>(ws + w_off<FMT>(row, ch),
                                         code_tile + int64_t(row) * C::CODE_ROW + chunk * WB + ch * 16, pol);
    }
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
      if constexpr (FMT == FMT_NVFP4) {  // rows of 72 scale bytes (K = 1152) are 8-byte aligned
        const uint2* p = reinterpret_cast<const uint2*>(srow + k2 * C::SCALE_ROW + chunk * 16);
        sc_next[k2].lo = __ldg(p);
        if (C::SCALE_ROW % 16 == 0 || chunk * 16 + 8 < C::SCALE_ROW) sc_next[k2].hi = __ldg(p + 1);
      } else {
        sc_next[k2].w = __ldg(reinterpret_cast<const uint32_t*>(srow + k2 * C::SCALE_ROW + chunk * 4));
      }
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

  // chunk 0 (group 0) first, ahead of the pair-list reads; chunk q >= 1 is issued at step (q - 1) R, in group q R
  load_w(0);

  // activation rows of this thread's cp.async pieces: rows (tid + i NT) / ACH, piece (tid + i NT) % ACH
  constexpr int ACH = SK / 8, AI = TT * ACH / NT;
  static_assert(TT * ACH % NT == 0);
  const uint16_t* arow[AI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = (tid + i * NT) / ACH;
    if (r >= n) arow[i] = nullptr;
    else if constexpr (MODE == Mode::Up) arow[i] = g.a + size_t(g.w.pair_token[pb + r]) * K;
    else arow[i] = g.a + size_t(pb + r) * K;
  }
  auto load_a = [&](int ks) {
    unsigned char* as = smem + C::OFF_A + (ks % AS) * C::A_SLOT;
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      const int r = (tid + i * NT) / ACH, ch = (tid + i * NT) % ACH;
      if (arow[i] && (C::KTAIL == 0 || ks < KS - 1 || ch < C::KTAIL / 8))
        prefill::cp_async16(as + a_off<SK>(r, ch), arow[i] + ks * SK + ch * 8);
    }
  };

  if constexpr (MODE == Mode::Down) {
    // act is moe_gate_up's output (a loop ptxas cannot see through keeps the copies above ahead of the wait)
#pragma unroll 1
    for (int i = 0; i <= (n >> 30); ++i) asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#pragma unroll
  for (int s = 0; s < AS - 1; ++s) {
    if (s < KS) load_a(s);
    prefill::cp_commit();
  }

  const float factor = FMT == FMT_NVFP4 ? __ldg(g.factors + e) : 1.f;

  auto run = [&]<int NB>() {
    float acc[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;

#pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
      prefill::cp_wait<AS - 2>();
      __syncthreads();
      if (ks % R == 0) {
        take_scales();
        if (ks / R + 1 < KC) load_w(ks / R + 1);
      }
      if (ks + AS - 1 < KS) load_a(ks + AS - 1);
      prefill::cp_commit();
      const unsigned char* W = smem + ((ks / R) % 2) * C::W_SLOT;
      const unsigned char* A = smem + C::OFF_A + (ks % AS) * C::A_SLOT;
      [[maybe_unused]] const int sub = ks % R;  // NVFP4: activation step inside the chunk
#pragma unroll
      for (int h = 0; h < SK / 64; ++h) {
        if (C::KTAIL && ks == KS - 1 && h >= C::KTAIL / 64) break;  // partial last step
        // A fragments of the four k16 MMAs of this 64-element step: [mma][register]
        unsigned a[4][4];
#pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
          const int row = row0 + k2;
          if constexpr (FMT == FMT_NVFP4) {
            const int step = sub * (SK / 64) + h;  // 64-k step inside the chunk
            const uint32_t sb = (sc[k2] >> (8 * step)) & 0xffu;
            const uint2 wd = *reinterpret_cast<const uint2*>(W + w_off<FMT>(row, step * 2 + (c >> 1)) + (c & 1) * 8);
            const uint32_t s2 = e4m3x2_to_half2(sb * 0x0101u);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t word = j < 2 ? wd.x : wd.y;
              a[j][k2] = dec_nvfp4(word >> (16 * (j & 1)), s2, factor);
              a[j][2 + k2] = dec_nvfp4(word >> (16 * (j & 1) + 8), s2, factor);
            }
          } else {
            const uint32_t sb = (sc[k2] >> (8 * (h * 2 + (c >> 1)))) & 0xffu;
            const uint4 q = *reinterpret_cast<const uint4*>(W + w_off<FMT>(row, h * 4 + c));
            const float s = sb ? __uint_as_float(sb << 23) : __uint_as_float(0x00400000u);  // 2^(sb - 127)
            const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              a[j][k2] = dec_mxfp8(words[j], s);
              a[j][2 + k2] = dec_mxfp8(words[j] >> 16, s);
            }
          }
        }
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
          const int r = nb * 8 + gq;
          const uint4 q0 = *reinterpret_cast<const uint4*>(A + a_off<SK>(r, h * 8 + 2 * c));
          const uint4 q1 = *reinterpret_cast<const uint4*>(A + a_off<SK>(r, h * 8 + 2 * c + 1));
          prefill::mma_bf16(acc[nb], a[0], q0.x, q0.y);
          prefill::mma_bf16(acc[nb], a[1], q0.z, q0.w);
          prefill::mma_bf16(acc[nb], a[2], q1.x, q1.y);
          prefill::mma_bf16(acc[nb], a[3], q1.z, q1.w);
        }
      }
    }

#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int tok = nb * 8 + 2 * c + p;
        if (tok >= n) continue;
        const float c0 = acc[nb][p], c1 = acc[nb][2 + p];  // weight rows n0 + row0 and n0 + row0 + 1
        if constexpr (MODE == Mode::Up) {
          const float gt = prefill::round_bf16(c0), u = prefill::round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + tok) * kInter + (n0 + row0) / 2] = prefill::f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + tok];
          *reinterpret_cast<uint32_t*>(g.out + size_t(dst) * kHidden + n0 + row0) = pack_bf16x2(c0, c1);
        }
      }
  };

  // the n8 block count is a template constant: a runtime bound would leave predicated MMAs in the tensor pipe
  switch ((n + 7) / 8) {
    case 1: run.template operator()<1>(); break;
    case 2: run.template operator()<2>(); break;
    case 3: if constexpr (TT >= 32) run.template operator()<3>(); break;
    case 4: if constexpr (TT >= 32) run.template operator()<4>(); break;
    case 5: if constexpr (TT >= 64) run.template operator()<5>(); break;
    case 6: if constexpr (TT >= 64) run.template operator()<6>(); break;
    case 7: if constexpr (TT >= 64) run.template operator()<7>(); break;
    default: if constexpr (TT >= 64) run.template operator()<8>(); break;
  }
  prefill::cp_wait<0>();
}

// One CTA (prefill's route kernel with the tile size as a parameter): expert offsets, tiles, scatter of the pairs
// into expert order.
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ idx, int T, int TR, MoeWork w) {
  __shared__ int offs[kExperts], tile0[kExperts + 1], cursor[kExperts];
  const int tid = threadIdx.x;
  if (tid == 0) {
    int o = 0, tl = 0;
    for (int e = 0; e < kExperts; ++e) {
      const int cnt = w.counts[e];
      offs[e] = o;
      tile0[e] = tl;
      o += cnt;
      tl += (cnt + TR - 1) / TR;
    }
    tile0[kExperts] = tl;
    *w.tiles_n = tl;
  }
  if (tid < kExperts) cursor[tid] = 0;
  __syncthreads();
  if (tid < kExperts) {
    const int cnt = w.counts[tid];
    for (int i = 0; i * TR < cnt; ++i)
      w.tiles[tile0[tid] + i] = make_int4(tid, offs[tid] + i * TR, min(TR, cnt - i * TR), 0);
  }
  for (int p = tid; p < T * kTopK; p += blockDim.x) {
    const int e = idx[p];
    const int pos = offs[e] + atomicAdd(&cursor[e], 1);
    w.pair_token[pos] = p / kTopK;
    w.pair_dst[pos] = p;
  }
}

int max_tiles(int T) {
  const int tr = moe_tile_rows(T);
  return (T * kTopK + tr - 1) / tr + kExperts;
}

template <int FMT, Mode MODE, int K, int TT, int SK>
void launch(const Args& g, int T, cudaStream_t stream) {
  using C = Cfg<FMT, K, TT, SK>;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;
  auto* kernel = expert_kernel<FMT, MODE, K, TT, SK>;
  static const cudaError_t attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  if (attr != cudaSuccess) throw std::runtime_error("batch moe: shared memory attribute failed");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(unsigned(max_tiles(T) * (NE / ROWS)));
  cfg.blockDim = dim3(NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute pdl{};
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  if constexpr (MODE == Mode::Down) cfg.attrs = &pdl, cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, kernel, g) != cudaSuccess) throw std::runtime_error("batch moe: launch failed");
}

// Per tile size: TT = 16, one chunk per step: two CTAs per SM (48 KB NVFP4, 40 KB MXFP8). TT = 64: one CTA (its
// activation slots fill the rest; NVFP4 in half-chunk steps).
template <int FMT, Mode MODE, int K>
void dispatch(const Args& g, int T, cudaStream_t stream) {
  constexpr int WK = FMT == FMT_NVFP4 ? 2 * WB : WB;
  if (moe_tile_rows(T) == 16) launch<FMT, MODE, K, 16, WK>(g, T, stream);
  else launch<FMT, MODE, K, 64, 128>(g, T, stream);
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
    return v == 16 || v == 64 ? v : 0;
  }();
  return forced ? forced : T <= 128 ? 16 : 64;
}

int moe_tiles_capacity(int max_T) {
  int cap = 0;
  for (int tr : {16, 64}) cap = std::max(cap, (max_T * kTopK + tr - 1) / tr + kExperts);
  return cap;
}

void moe_route(const int* idx, int T, const MoeWork& w, cudaStream_t stream) {
  route_kernel<<<1, 1024, 0, stream>>>(idx, T, moe_tile_rows(T), w);
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
