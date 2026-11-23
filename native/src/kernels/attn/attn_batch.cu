// Decode attention for a batch of sequences (attn_decode with batch > 1). Contract: attn.h / native/KERNELS.md,
// area `attn`; the arithmetic and rounding points are the cluster kernel's (attn.cu) M > 1 path.
//
// Design
// ------
// * Work: one block of 4 warps per (key split s, KV head g, sequence z): grid (S, 8, batch). Shared memory is
//   sized for four resident blocks per SM (680 on the RTX 5090), so batch 64 (512 blocks per split) runs in
//   one wave, instead of the cluster kernel's 16 x 8 blocks per sequence with their fixed per-block latency
//   (~200 us per visit at batch 64). A block takes its 3 GQA heads x M tokens as rows r = 3 t + h.
// * Ragged rows (DecodeArgs::rows): the kernel is instantiated for the step's largest M (shared memory and
//   register arrays), and every block works on its own sequence's m_b tokens: loads, conv, bias rows, the
//   16-row score / P V tiles and the outputs past 3 m_b are skipped, so the cost follows m_b.
// * Splits: the host picks the largest S whose blocks are all resident at once (S = 1 at batch 64, more for
//   small batches), capped at 256 cached keys per split; each sequence divides its cached keys [0, L_old)
//   into contiguous chunks of `per` keys (from pos on the device, so the launch shape is fixed for a CUDA
//   graph) and blocks past the last chunk exit. Split 0 also takes the call's own M keys (as its last tile).
//   With one chunk the block writes the output itself; otherwise every block writes its partial
//   (unnormalised o, max, sum) to the workspace and the last one of the head (counter) merges them, several
//   partials' loads in flight at once.
// * Prologue: q / rel scales and conv taps are read before griddepcontrol.wait; after it, pos, then every
//   projection row and conv state the block needs in one burst of loads, then the first two tiles (they
//   stream during the prologue). q / rel rows are cleaned up by warps 2, 3 (by all four warps past split 0),
//   the rows' norm reductions interleaved; each warp then holds the q fragments of its own k-steps in
//   registers and the q tile's shared memory becomes the S exchange and the rel_proj window.
// * Keys are streamed in tiles of 16 with cp.async, double-buffered: K and V rows (16-byte chunks XOR-swizzled
//   by key) and ring kpos. Warp w copies, converts and reads only dims 32 w .. 32 w + 31 of every K and V row
//   (and the kpos of the keys whose bias it computes), so it waits for its own copies and re-issues its part of
//   a stage as soon as it is done with it: two tiles stay in flight per block. BF16 V chunks are converted to
//   fp16 in place; E4M3 chunks are widened (K to bf16, V to fp16, both exact) into one 16-bit tile.
// * Per tile and 16-row tile: each warp computes S over its 2 k-steps (mma m16n8k16 bf16, exact bf16 q and k,
//   fp32 sums) and the bias of a quarter of the (row, key) pairs (bf16(rel . rel_proj[:, d]) for d < 1024, 0
//   beyond, -inf for a masked key; rel_proj from a 28-distance window per tile staged through shared memory a
//   tile ahead, the wrapped ring's far keys from global), and writes both to shared memory; after one barrier
//   every warp adds the four partial S in warp order (all warps hold the same S and row statistics), applies
//   1/sqrt(128) and the bias, runs the online softmax in the log2 domain and O += P V for its own 32 dims (mma
//   fp16, P as fp16 hi + lo (P - hi), ~22 bits; V fp16). P stays in registers (the S accumulators are the A
//   fragments of P V). Two barriers per tile (one more per extra row tile).
// * Split 0's warps 0 and 1 run the key / value short convolutions of the head (+ key clip / value RMS norm,
//   lanes holding 4 channels as the cluster kernel, so the reductions and bits match), write the advanced
//   conv state (no other block reads these channels) and write the new keys / values (BF16, then E4M3 for FP8
//   caches) to the cache rows, from where split 0 streams them as its last tile. Cache rows: full layers write
//   rows p0.. at once (other blocks only read rows below p0); sliding rings too, unless M > 1 and the ring has
//   wrapped (p0 + M - 1 >= 1024): then the overwritten slots still hold keys earlier tokens of the call see, so
//   the new rows go to the workspace (the last tile streams them from there) and the head's last block writes
//   them, and the last head (sequence counter) writes kpos. M = 1 overwrites the slot at distance 1024 (invalid
//   either way; a kpos >= p0 is ignored too) and writes kpos at once.
// * Ring semantics, masking, FP8 rule and capacity as attn.cu: a cached ring entry is used iff
//   0 <= kpos < p0 and p - kpos < 1024; masked keys' values never reach O (non-finite V -> 0 in fp16).
// * Programmatic dependent launch: nothing an earlier kernel writes is read before griddepcontrol.wait.
#include "kernels/attn/attn_batch.h"

#include "kernels/attn/attn_impl.cuh"
#include "kernels/common/l2.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace ingine::kernels::attn {
namespace {

using namespace detail;

constexpr int NT = 128;              // threads per block (4 warps); warp w owns output dims 32 w .. 32 w + 31
constexpr int PSTR = D + 4;          // workspace partial row: o (D), then (max, sum)
constexpr int CNT = 16;              // workspace counters per sequence: [g] the head's splits, [HKV] its heads
constexpr int SPLIT_KEYS = 256;      // fewest cached keys per split
constexpr int MAX_SPLITS = 48;       // the merge's per-partial scratch fits the tile stages
constexpr int BLOCKS_PER_SM = 4;     // resident blocks per SM (shared memory) = what the splits fill

// shared memory layout (bytes), per M and cache type
template <int M, bool F8>
struct Lay {
    static constexpr int ROWS = GQA * M;
    static constexpr int MT = (ROWS + 15) / 16;  // 16-row mma tiles
    static constexpr int KT = 16;                // keys per tile
    static constexpr int NSTG = 2;               // tiles in the cp.async ring
    static constexpr int T16 = KT * 2 * D * 2;   // a 16-bit tile [KT][K | V][D]
    // BF16: the cp.async stages (V converted to fp16 in place); E4M3: the widened tile
    static constexpr int STG = 0;
    static constexpr int RAW = STG + (F8 ? 1 : NSTG) * T16;          // E4M3 stages [NSTG][K | V][KT][D]
    static constexpr int KPS = RAW + (F8 ? NSTG * 2 * KT * D : 0);   // ring kpos low words, int [NSTG][warp][8]
    // q bf16 [ROWS][D] (chunks swizzled by row) for the prologue; then the S exchange float4 [warp][n][32] and
    // the tile's rel_proj window fp32 [WS][DREL]
    static constexpr int WS = 28;
    static constexpr int QX = KPS + NSTG * 4 * 8 * 4, XS = 4096, RW = QX + XS;
    static constexpr int RELD = QX + (ROWS * D * 2 > XS + DREL * WS * 4 ? ROWS * D * 2 : XS + DREL * WS * 4);
    static constexpr int BIAS = RELD + ROWS * DREL * 2;  // rel bf16 [ROWS][DREL]
    static constexpr int FLAG = BIAS + 16 * KT * 2;      // bias bf16 [16][KT] of a row tile
    static constexpr int BYTES = FLAG + 16;              // int: this block merges the head
    // four blocks per SM: 100 KB of shared memory per SM, 1 KB of it reserved per block (sm_120)
    static_assert(BYTES + 1024 <= 102400 / BLOCKS_PER_SM, "shared memory for four blocks per SM");
    // the merge's (max, sum) and weights per row and partial, and the merged (max, sum) per row
    static_assert(ROWS * MAX_SPLITS * 12 + ROWS * 8 + 8 <= KPS, "merge scratch fits the tile stages");
};

struct Params {
    const float* proj;            // fp32 rows, or (null)
    const __nv_bfloat16* proj16;  // bf16 rows
    int R;
    long long cap;  // cache rows: full layers; sliding rings below 1024 rows (positions never wrap)
    int log_scaling;
    const float* q_scale;
    const float* r_scale;
    const float* rel_proj;
    const float* k_conv;
    const float* v_conv;
    float* conv_k;
    float* conv_v;
    void* k_cache;  // __nv_bfloat16, or E4M3 bytes (F8)
    void* v_cache;
    long long* kpos;
    const long long* pos;
    int* counter;               // [batch][CNT]
    __nv_bfloat16* nkv;         // deferred new ring rows [batch][HKV][K | V][M][D]
    float* part;                // split partials [batch][HKV][S][ROWS][PSTR]
    __nv_bfloat16* out;
    const int* slots;
    const int* rows;  // ragged rows [batch + 1] (null: M per sequence)
    long long cache_stride, kpos_stride, conv_stride;
    int zero;  // 0 at run time, opaque to the compiler (fenced griddepcontrol.wait)
    int conv_write;  // write the advanced conv states
};

// 16-bit offset of (key, K | V, 16-byte chunk) in a [KT][2][D] tile; chunks XOR-swizzled by key
__device__ __forceinline__ int toff(int key, int kv, int ch) { return (key * 2 + kv) * D + ((ch ^ (key & 7)) * 8); }

// fp32 pair -> fp16 hi and lo = (x - hi) pairs
__device__ __forceinline__ void split_half2(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 f = __half22float2(h);
    const __half2 l = __floats2half2_rn(a - f.x, b - f.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}

// RAG: ragged rows (p.rows); without, every per-sequence row bound folds to the compile-time M
template <int M, bool FULL, bool F8, bool RAG>
__global__ void __launch_bounds__(NT, 4) attn_batch_kernel(const Params p) {
    static_assert(FULL || !F8, "FP8 caches are full-layer only");
    using CT = std::conditional_t<F8, __nv_fp8_storage_t, __nv_bfloat16>;  // cache element
    using L = Lay<M, F8>;
    constexpr int ROWS = L::ROWS, MT = L::MT, KT = L::KT, NSTG = L::NSTG;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* const stg = reinterpret_cast<__nv_bfloat16*>(smem + L::STG);
    unsigned char* const raw = smem + L::RAW;
    int* const kps = reinterpret_cast<int*>(smem + L::KPS);
    __nv_bfloat16* const qb = reinterpret_cast<__nv_bfloat16*>(smem + L::QX);
    __nv_bfloat16* const reld = reinterpret_cast<__nv_bfloat16*>(smem + L::RELD);
    __nv_bfloat16* const bias = reinterpret_cast<__nv_bfloat16*>(smem + L::BIAS);
    int* const flag = reinterpret_cast<int*>(smem + L::FLAG);

    const int s = blockIdx.x, g = blockIdx.y, z = blockIdx.z, S = gridDim.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, half = lane >> 4, c = lane & 15;
    const int gid = lane >> 2, t4 = lane & 3;
    const int R = p.R;
    // this sequence's slot: the slot table is written before the step, never by the kernels before this one
    const long long slot = p.slots ? (long long)p.slots[z] : (long long)z;
    CT* const kc = static_cast<CT*>(p.k_cache) + slot * p.cache_stride;
    CT* const vc = static_cast<CT*>(p.v_cache) + slot * p.cache_stride;
    long long* const kpos_z = p.kpos ? p.kpos + slot * p.kpos_stride : nullptr;
    int* const counter_z = p.counter + size_t(z) * CNT;

    // weights (no kernel writes them) before the wait: q / rel scales (q warps: 2, 3 on split 0, else all), conv
    // taps (split 0's warps 0, 1)
    constexpr int NQ = (ROWS + 1) / 2, NR = (ROWS + 3) / 4;  // q rows per warp, rel rows per half-warp (at most)
    const bool qwarp = warp >= 2 || s != 0;
    const int nqw = s == 0 ? 2 : 4, qw = s == 0 ? warp - 2 : warp;  // q warps, this one's index among them
    const int kv = warp;                  // conv warp: 0 = k, 1 = v
    const int ch0 = g * D + 4 * lane;    // conv warp lane: channels ch0 .. ch0 + 3
    float4 qs[GQA], w4[4];               // w4 sliding: [channel][tap] (conv1d order); full: [lag][channel]
    float rs[GQA];
    if (qwarp) {
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            qs[i] = __ldg(reinterpret_cast<const float4*>(p.q_scale + (g * GQA + i) * D) + lane);
            rs[i] = __ldg(p.r_scale + (g * GQA + i) * DREL + c);
        }
    } else if (s == 0) {
        const float* const w = kv ? p.v_conv : p.k_conv;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            w4[j] = __ldg(reinterpret_cast<const float4*>(FULL ? w + j * CONV_C + ch0 : w + (ch0 + j) * 4));
    }
    // a loop ptxas cannot see through (a plain wait lets it sink earlier loads below it)
#pragma unroll 1
    for (int i = 0; i <= p.zero; ++i) asm volatile("griddepcontrol.wait;" ::: "memory");
    // this sequence's rows: [row0, row0 + mb) of proj and out; rb = its 3 mb query rows (of the ROWS the block
    // is sized for)
    int row0 = z * M, mb = M;
    if constexpr (RAG) row0 = p.rows[z], mb = p.rows[z + 1] - row0;
    const int p0 = int(ld_s64(p.pos + z)), rb = GQA * mb;
    // projection values (BF16-valued) at element offset `off` of the sequence's rows: fp32 proj or bf16 proj16
    const bool p16 = p.proj16 != nullptr;
    const float* const proj_z = p16 ? nullptr : p.proj + size_t(row0) * R;
    const unsigned short* const proj16_z = p16 ? reinterpret_cast<const unsigned short*>(p.proj16) + size_t(row0) * R : nullptr;
    auto bf = [](unsigned bits) { return __uint_as_float(bits << 16); };
    auto pj1 = [&](size_t off) { return p16 ? bf(__ldcg(proj16_z + off)) : __ldcg(proj_z + off); };
    auto pj2 = [&](size_t off) {  // off even
        if (!p16) return __ldcg(reinterpret_cast<const float2*>(proj_z + off));
        const unsigned u = __ldcg(reinterpret_cast<const unsigned*>(proj16_z + off));
        return make_float2(bf(u & 0xffffu), bf(u >> 16));
    };
    auto pj4 = [&](size_t off) {  // off a multiple of 4
        if (!p16) return __ldcg(reinterpret_cast<const float4*>(proj_z + off));
        const uint2 u = __ldcg(reinterpret_cast<const uint2*>(proj16_z + off));
        return make_float4(bf(u.x & 0xffffu), bf(u.x >> 16), bf(u.y & 0xffffu), bf(u.y >> 16));
    };
    __nv_bfloat16* const out_z = p.out + size_t(row0) * HQ * D;
    // the projection rows and conv state next, all in flight together and ahead of the K/V tiles
    float4 qa[NQ], x4[3 + M];  // x4: [conv state 0..2, x_0 .. x_{mb-1}, zeros]
    float ra[NR], dynv = 0.f;  // dynv: full layers' dynamic tap (lane / 4, lane % 4) = (token, lag)
    float* const cst = (kv ? p.conv_v : p.conv_k) + slot * p.conv_stride + ch0;
    if (qwarp) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {  // rows past the sequence's (ragged) are skipped, warp-uniformly
            const int row = min(qw + nqw * k, rb - 1);
            qa[k] = qw + nqw * k < rb ? pj4(size_t(row / GQA) * R + (g * GQA + row % GQA) * D + 4 * lane)
                                      : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            const int row = min(qw * 2 + half + 2 * nqw * k, rb - 1);
            ra[k] = qw * 2 + 2 * nqw * k < rb ? pj1(size_t(row / GQA) * R + OFF_REL + (g * GQA + row % GQA) * DREL + c)
                                               : 0.f;
        }
    } else if (s == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j) x4[j] = __ldcg(reinterpret_cast<const float4*>(cst + j * CONV_C));
#pragma unroll
        for (int m = 0; m < M; ++m)
            x4[3 + m] = m < mb ? pj4(size_t(m) * R + (kv ? OFF_V : OFF_K) + ch0) : make_float4(0.f, 0.f, 0.f, 0.f);
        if (FULL && lane < 4 * mb) dynv = pj1(size_t(lane / 4) * R + (kv ? OFF_VDYN : OFF_KDYN) + g * 4 + lane % 4);
    }
    // out of cache capacity (uniform: no block reaches a barrier); a ring below 1024 rows must not wrap
    if ((FULL || p.cap < WIN) && p0 + mb > p.cap) return;
    // cached keys: the linear cache up to p0; the ring's 1024 slots, or before it wraps slots 0..p0-1
    const int L_old = FULL ? p0 : min(p0, WIN);
    const int per = (max(SPLIT_KEYS, (L_old + S - 1) / S) + KT - 1) / KT * KT;
    const int nact = L_old > 0 ? (L_old + per - 1) / per : 1;
    if (s >= nact) return;
    const int begin = s * per, end = min(L_old, begin + per), ntile = (end - begin + KT - 1) / KT;
    // M > 1 on a wrapped ring: the new rows overwrite slots earlier tokens of the call still read
    const bool defer = !FULL && mb > 1 && p0 + mb - 1 >= WIN;

    // this block's tiles: its cached keys, then on split 0 the call's own keys (the rows the prologue writes to
    // the cache, or to the workspace when deferred), so the stream does not wait for the prologue
    const int nt = ntile + (s == 0);
    // ---- tile t into stage t % NSTG: this warp's part of it, the K / V dims 32 warp .. 32 warp + 31 it alone reads
    // (and the ring kpos of the keys it computes the bias of); always one group. The own keys' tile only with
    // `fresh_ok` (after the prologue's barrier); keys past mb are zero-filled.
    // This lane's 16-byte chunks c: K or V (ckv) of key ckey, column cch (BF16: 8 dims, E4M3: 16).
    constexpr int NC = F8 ? 2 : 4, CE = 16 / sizeof(CT);
    auto ckv = [](const int c) { return F8 ? c : c >> 1; };
    auto ckey = [&](const int c) { return F8 ? lane >> 1 : (lane >> 2) + 8 * (c & 1); };
    const int cch = F8 ? 2 * warp + (lane & 1) : 4 * warp + (lane & 3);
    auto cdst = [&](const int c, const int st) {
        return reinterpret_cast<float*>(F8 ? raw + ((st * 2 + ckv(c)) * KT + ckey(c)) * D + cch * 16
                                           : reinterpret_cast<unsigned char*>(stg + st * (L::T16 / 2) +
                                                                              toff(ckey(c), ckv(c), cch)));
    };
    const std::uint64_t kvpol = l2::policy<KV_L2>();
    auto issue = [&](const int t, const bool fresh_ok) {
        const int st = t % NSTG, kb = begin + t * KT;
        if (t < ntile) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int j = kb + ckey(c);
                const bool ok = j < end;
                const CT* const src = (ckv(c) ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + cch * CE;
                cp16z(cdst(c, st), reinterpret_cast<const float*>(src), ok, kvpol);
            }
            if constexpr (!FULL) {  // low word of the int64 position, keys 8 (warp / 2) ..
                const int j = kb + (warp >> 1) * 8 + lane;
                if (lane < 8)
                    cp4z(reinterpret_cast<float*>(kps + (st * 4 + warp) * 8 + lane),
                         reinterpret_cast<const float*>(kpos_z + (j < end ? j : 0)), j < end);
            }
        } else if (fresh_ok && t == ntile && s == 0) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int key = ckey(c), kv = ckv(c), m = key < mb ? key : 0;
                const CT* const src =
                    defer ? reinterpret_cast<const CT*>(p.nkv + ((size_t(z * HKV + g) * 2 + kv) * M + m) * D)
                          : (kv ? vc : kc) + (size_t(FULL ? p0 + m : (p0 + m) & (WIN - 1)) * HKV + g) * D;
                cp16z(cdst(c, st), reinterpret_cast<const float*>(src + cch * CE), key < mb, kvpol);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    for (int t = 0; t < NSTG; ++t) issue(t, false);
    // the own keys' tile is among those: it goes out after the prologue, in a group of its own
    const bool late = s == 0 && ntile < NSTG;

    // tile t's rel_proj window: columns wbase(t) .. + WS - 1 hold every distance its bias needs (cached keys
    // kb .. kb + 15 at ring distance e + 1 - key from the first token, own keys at 0 .. mb - 1); a wrapped ring's
    // tile also has keys near distance 1024 (read from global). Staged through registers, a tile ahead.
    float* const rw = reinterpret_cast<float*>(smem + L::RW);  // [WS][DREL]
    constexpr int WS = L::WS, NW = DREL * WS / 4;
    auto wbase = [&](const int t) {
        if (t >= ntile) return 0;
        const int kb = begin + t * KT, e = FULL ? p0 - 1 - kb : (p0 - 1 - kb) & (WIN - 1);
        return ((e - 14) >> 2) * 4;
    };
    auto win_load = [&](const int t) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < NW && t < nt) {
            const int d = wbase(t) + 4 * (tid % (WS / 4));
            if (d >= 0 && d < EXT) v = __ldg(reinterpret_cast<const float4*>(p.rel_proj + (tid / (WS / 4)) * EXT + d));
        }
        return v;
    };
    float4 rwv = win_load(0);

    // ---- q / rel post-processing (q warps) and, on split 0, the new keys / values (warps 0, 1)
    auto tau_of = [&](int t) {
        return p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)) : 1.f;
    };
    if (qwarp) {
        auto pick = [](const auto (&v)[GQA], int i) { return i == 0 ? v[0] : i == 1 ? v[1] : v[2]; };
        // q: one row per warp, lane = 4 dims; the rows' norm reductions interleaved
        float x[NQ][4], ss[NQ];
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const float4 a = qa[k], sc = pick(qs, min(qw + nqw * k, rb - 1) % GQA);
            x[k][0] = a.x * sc.x, x[k][1] = a.y * sc.y, x[k][2] = a.z * sc.z, x[k][3] = a.w * sc.w;
            ss[k] = x[k][0] * x[k][0] + x[k][1] * x[k][1] + x[k][2] * x[k][2] + x[k][3] * x[k][3];
        }
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
#pragma unroll
            for (int k = 0; k < NQ; ++k) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int rq = qw + nqw * k, row = min(rq, rb - 1), t = row / GQA;
            if (rq >= rb) break;
            const float f = clip_factor(ss[k], CLIP_QK);
            const float tau = tau_of(t);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                x[k][e] = bf16r(x[k][e] * f);
                if (p.log_scaling) x[k][e] = bf16r(x[k][e] * tau);
            }
            // exact bf16 q for the tensor cores (scale applied to the scores)
            reinterpret_cast<uint2*>(qb + row * D + (((lane >> 1) ^ (row & 7)) * 8) + (lane & 1) * 4)[0] =
                make_uint2(pack2(x[k][0], x[k][1]), pack2(x[k][2], x[k][3]));
        }
        // rel: one row per half-warp
        float r[NR], rr2[NR];
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            r[k] = ra[k] * pick(rs, min(qw * 2 + half + 2 * nqw * k, rb - 1) % GQA);
            rr2[k] = r[k] * r[k];
        }
#pragma unroll
        for (int o = 1; o < 16; o <<= 1)
#pragma unroll
            for (int k = 0; k < NR; ++k) rr2[k] += __shfl_xor_sync(0xffffffffu, rr2[k], o);
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            const int rr = qw * 2 + half + 2 * nqw * k, row = min(rr, rb - 1), t = row / GQA;
            if (qw * 2 + 2 * nqw * k >= rb) break;
            float y = bf16r(r[k] * clip_factor(rr2[k], CLIP_REL));
            if (p.log_scaling) y = bf16r(y * tau_of(t));
            if (rr < rb) reld[row * DREL + c] = __float2bfloat16_rn(y);  // exact: y is BF16-valued
        }
    } else if (s == 0) {
        // warp kv (0 = k, 1 = v), lane = channels ch0 .. ch0 + 3 of head g
        auto comp = [](const float4& v, int e) { return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w; };
        float y[M][4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float ext[3 + M];
#pragma unroll
            for (int j = 0; j < 3 + M; ++j) ext[j] = comp(x4[j], e);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                if (m >= mb) break;
                float o;
                if constexpr (FULL) {  // dynamic conv, weights by lag (0 = current), / sqrt(9)
                    o = 0.f;
#pragma unroll
                    for (int lag = 0; lag < 4; ++lag) {
                        const float wl = __fadd_rn(comp(w4[lag], e), __shfl_sync(0xffffffffu, dynv, m * 4 + lag));
                        o = __fadd_rn(o, __fmul_rn(wl, ext[m + 3 - lag]));
                    }
                    o = __fmul_rn(__fadd_rn(o, ext[m + 3]), INV_3);
                } else {  // static conv, conv1d order (tap 3 = current), / sqrt(5)
                    o = __fmul_rn(comp(w4[e], 0), ext[m]);
#pragma unroll
                    for (int j = 1; j < 4; ++j) o = __fadd_rn(o, __fmul_rn(comp(w4[e], j), ext[m + j]));
                    o = __fmul_rn(__fadd_rn(o, ext[m + 3]), INV_SQRT5);
                }
                y[m][e] = bf16r(o);
            }
        }
        // the advanced conv state, x4[mb .. mb + 2] (only this block reads these channels), unless the caller
        // writes it after acceptance; one constant-index branch per row count keeps x4 in registers
        if (p.conv_write) {
#pragma unroll
            for (int n = 1; n <= M; ++n) {
                if (n == mb) {
#pragma unroll
                    for (int j = 0; j < 3; ++j) __stcg(reinterpret_cast<float4*>(cst + j * CONV_C), x4[n + j]);
                }
            }
        }
        // key clip / value RMS norm per token (the reductions interleaved), the cache values and the cache rows
        // (or the workspace)
        float ss[M];
#pragma unroll
        for (int m = 0; m < M; ++m)
            ss[m] = m < mb ? y[m][0] * y[m][0] + y[m][1] * y[m][1] + y[m][2] * y[m][2] + y[m][3] * y[m][3] : 0.f;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
#pragma unroll
            for (int m = 0; m < M; ++m) ss[m] += __shfl_xor_sync(0xffffffffu, ss[m], o);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            if (m >= mb) break;
            const float f = kv == 0 ? clip_factor(ss[m], CLIP_QK) : rsqrtf(ss[m] / float(D) + 1e-6f);
            float v[4] = {y[m][0] * f, y[m][1] * f, y[m][2] * f, y[m][3] * f};
            if constexpr (F8) {  // the values the cache will hold: BF16, then E4M3 (exact in BF16)
#pragma unroll
                for (int e = 0; e < 4; ++e) v[e] = e4m3r(bf16r(v[e]));
            }
            const uint2 u = make_uint2(pack2(v[0], v[1]), pack2(v[2], v[3]));
            if (defer) {
                reinterpret_cast<uint2*>(p.nkv + ((size_t(z * HKV + g) * 2 + kv) * M + m) * D + 4 * lane)[0] = u;
            } else {
                const int row = FULL ? p0 + m : (p0 + m) & (WIN - 1);
                CT* const dst = (kv ? vc : kc) + (size_t(row) * HKV + g) * D + 4 * lane;
                if constexpr (F8) *reinterpret_cast<uint32_t*>(dst) = pack_e4m3x2(v[0], v[1]) | pack_e4m3x2(v[2], v[3]) << 16;
                else *reinterpret_cast<uint2*>(dst) = u;
            }
        }
        if (!FULL && !defer && g == 0 && kv == 0 && lane < mb) kpos_z[(p0 + lane) & (WIN - 1)] = p0 + lane;
    }
    __syncthreads();
    if (late) issue(ntile, true);
    // dependents may launch (they wait for this grid's completion before reading its output)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // this warp's q fragments: k-steps 2 warp, 2 warp + 1 (dims 32 warp ..) of every row tile; padding rows read
    // row rb - 1 (masked). The q tile's shared memory then holds the S exchange.
    uint32_t qf[MT][2][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int kq = 0; kq < 2; ++kq) {
            const int r = min(mt * 16 + (lane & 15), rb - 1), ch = 4 * warp + 2 * kq + (lane >> 4);
            if (mt * 16 < rb) ldsm_x4(smem_u32(qb + r * D + ((ch ^ (r & 7)) * 8)), qf[mt][kq]);
        }
    float4* const xs = reinterpret_cast<float4*>(smem + L::QX);  // [warp][n][lane]: partial S
    auto win_store = [&](const float4 v) {  // rel_proj rows k, distances 4 c .. 4 c + 3 of the window
        if (tid < NW) {
            float* const w = rw + 4 * (tid % (WS / 4)) * DREL + tid / (WS / 4);
            w[0] = v.x, w[DREL] = v.y, w[2 * DREL] = v.z, w[3 * DREL] = v.w;
        }
    };

    // bias of (row, keys key0, key0 + 1) of a tile (kb: its first cached key; kq: their ring kpos; wb: its window):
    // bf16(rel . rel_proj[:, d]) for d < 1024, 0 beyond, -inf masked
    auto bias2 = [&](const int row, const int key0, const int* kq, const bool fresh, const int kb, const int wb) {
        const uint4* rl4 = reinterpret_cast<const uint4*>(reld + row * DREL);
        float rl[DREL];
        unpack8(rl4[0], rl);
        unpack8(rl4[1], rl + 8);
        float b[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int key = key0 + i, j = kb + key;
            const int kp = fresh ? (key < mb ? p0 + key : -1) : (j < end ? (FULL ? j : kq[i]) : -1);
            const int d = p0 + row / GQA - kp, x = d - wb;
            // old entries need kp < p0 (and the window on sliding layers)
            const bool ok = kp >= 0 && d >= 0 && (fresh || (kp < p0 && (FULL || d < WIN)));
            float acc = 0.f;
            if (x >= 0 && x < WS) {
                const float4* const w4 = reinterpret_cast<const float4*>(rw + x * DREL);
#pragma unroll
                for (int k4 = 0; k4 < DREL / 4; ++k4) {
                    const float4 w = w4[k4];
                    acc = fmaf(rl[4 * k4], w.x, acc);
                    acc = fmaf(rl[4 * k4 + 1], w.y, acc);
                    acc = fmaf(rl[4 * k4 + 2], w.z, acc);
                    acc = fmaf(rl[4 * k4 + 3], w.w, acc);
                }
            } else if (ok && d < EXT) {
#pragma unroll
                for (int k = 0; k < DREL; ++k) acc = fmaf(rl[k], __ldg(p.rel_proj + k * EXT + d), acc);
            }
            b[i] = ok ? (d < EXT ? bf16r(acc) : 0.f) : -INFINITY;
        }
        return __floats2bfloat162_rn(b[0], b[1]);  // exact: BF16-valued, 0 or -inf
    };

    // ---- per row tile: S over every row; online softmax; O += P V for dims 32 warp .. 32 warp + 31
    float o[MT][4][4], mrow[MT][2], lrow[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int n = 0; n < 4; ++n) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
        mrow[mt][0] = mrow[mt][1] = NEG_INIT, lrow[mt][0] = lrow[mt][1] = 0.f;
    }
    // Each warp: its 2 k-steps of S (partial sums over its 32 dims) and the bias of (n-block warp / 2, row half
    // warp % 2) to shared memory; after the barrier every warp adds the four partials in warp order (so all hold
    // the same S and row statistics) and runs the softmax and P V on its own dims.
    auto consume = [&](const __nv_bfloat16* tile, const int mt, const int st, const bool fresh, const int kb,
                       const int wb) {
        float sp[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) sp[n][0] = sp[n][1] = sp[n][2] = sp[n][3] = 0.f;
#pragma unroll
        for (int kq = 0; kq < 2; ++kq) {
            const int q = lane >> 3, key = (q >> 1) * 8 + (lane & 7);
            uint32_t bf[4];
            ldsm_x4(smem_u32(tile + toff(key, 0, 4 * warp + 2 * kq + (q & 1))), bf);
            const uint32_t b0[2] = {bf[0], bf[1]}, b1[2] = {bf[2], bf[3]};
            mma_bf16(sp[0], qf[mt][kq], b0);
            mma_bf16(sp[1], qf[mt][kq], b1);
        }
        {
            const int lr = gid + 8 * (warp & 1), key0 = (warp >> 1) * 8 + 2 * t4;
            if (mt * 16 + lr < rb)
                *reinterpret_cast<__nv_bfloat162*>(bias + lr * KT + key0) =
                    bias2(mt * 16 + lr, key0, kps + (st * 4 + warp) * 8 + 2 * t4, fresh, kb, wb);
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) xs[(warp * 2 + n) * 32 + lane] = make_float4(sp[n][0], sp[n][1], sp[n][2], sp[n][3]);
        __syncthreads();
        float sa[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            float4 a = xs[n * 32 + lane];
#pragma unroll
            for (int w = 1; w < 4; ++w) {
                const float4 b = xs[(w * 2 + n) * 32 + lane];
                a.x += b.x, a.y += b.y, a.z += b.z, a.w += b.w;
            }
            sa[n][0] = a.x, sa[n][1] = a.y, sa[n][2] = a.z, sa[n][3] = a.w;
        }
        // scores (log2 domain) with bias and mask; c0, c1: row gid, keys 2 t4, +1; c2, c3: row gid + 8
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = gid + 8 * h, row = mt * 16 + lr;
            float mx = mrow[mt][h];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                float2 bt = make_float2(-INFINITY, -INFINITY);
                if (row < rb) bt = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + lr * KT + n * 8 + 2 * t4));
                float& s0 = sa[n][2 * h];
                float& s1 = sa[n][2 * h + 1];
                s0 = bt.x == -INFINITY ? -INFINITY : fmaf(bt.x, LOG2E, s0 * QSCALE);
                s1 = bt.y == -INFINITY ? -INFINITY : fmaf(bt.y, LOG2E, s1 * QSCALE);
                mx = fmaxf(mx, fmaxf(s0, s1));
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float sc = exp2f(mrow[mt][h] - mx);
            mrow[mt][h] = mx;
#pragma unroll
            for (int n = 0; n < 4; ++n) o[mt][n][2 * h] *= sc, o[mt][n][2 * h + 1] *= sc;
            float sum = 0.f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                float& s0 = sa[n][2 * h];
                float& s1 = sa[n][2 * h + 1];
                s0 = exp2f(s0 - mx), s1 = exp2f(s1 - mx);
                sum += s0 + s1;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            lrow[mt][h] = lrow[mt][h] * sc + sum;
        }
        // O += P V: P (16 rows x 16 keys) = the S accumulators (fp16 hi + lo)
        uint32_t ah[4], al[4];
        split_half2(sa[0][0], sa[0][1], ah[0], al[0]);
        split_half2(sa[0][2], sa[0][3], ah[1], al[1]);
        split_half2(sa[1][0], sa[1][1], ah[2], al[2]);
        split_half2(sa[1][2], sa[1][3], ah[3], al[3]);
#pragma unroll
        for (int dn = 0; dn < 2; ++dn) {
            const int q = lane >> 3, key = (q & 1) * 8 + (lane & 7);
            uint32_t vb[4];
            ldsm_x4_t(smem_u32(tile + toff(key, 1, warp * 4 + dn * 2 + (q >> 1))), vb);
            const uint32_t b0[2] = {vb[0], vb[1]}, b1[2] = {vb[2], vb[3]};
            mma_f16(o[mt][2 * dn], ah, b0);
            mma_f16(o[mt][2 * dn], al, b0);
            mma_f16(o[mt][2 * dn + 1], ah, b1);
            mma_f16(o[mt][2 * dn + 1], al, b1);
        }
    };

    // ---- the block's tiles. A warp only touches its own dims of a stage (and its own kpos), so it waits for and
    // re-issues its own copies without the block; the barriers guard the S exchange and the bias.
    __syncthreads();  // every warp holds its q fragments: the q tile's memory takes the exchange and the window
    win_store(rwv);
    // one split: the output's sinks, in flight through the loop
    auto sink_of = [&](const int row) { return pj1(size_t(row / GQA) * R + OFF_SINK + g * GQA + row % GQA); };
    float sk[MT][2];
    if (nact == 1)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) sk[mt][h] = sink_of(min(mt * 16 + gid + 8 * h, rb - 1));
    for (int t = 0; t < nt; ++t) {
        const int st = t % NSTG;
        rwv = win_load(t + 1);
        // this thread's part of tile t; with the own keys' late group: everything
        if (late) asm volatile("cp.async.wait_all;" ::: "memory");
        else asm volatile("cp.async.wait_group %0;" ::"n"(NSTG - 1) : "memory");
        if constexpr (F8) {  // this thread's E4M3 chunks -> K bf16, V fp16 (exact) in the 16-bit tile
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int kv = ckv(c), key = ckey(c);
                const uint4 r = *reinterpret_cast<const uint4*>(cdst(c, st));
                const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // dims 16 cch + 8 h .. + 7
                    uint32_t u[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = e4m3x2_f2((w[2 * h + e / 2] >> (16 * (e % 2))) & 0xffffu);
                        u[e] = kv ? half2_of(f.x, f.y) : pack2(f.x, f.y);
                    }
                    *reinterpret_cast<uint4*>(stg + toff(key, kv, 2 * cch + h)) = make_uint4(u[0], u[1], u[2], u[3]);
                }
            }
            issue(t + NSTG, true);  // its raw chunks are widened (full layers: no kpos)
        } else {  // this thread's V chunks to fp16
#pragma unroll
            for (int c = 2; c < NC; ++c) {
                uint4* const q = reinterpret_cast<uint4*>(cdst(c, st));
                const uint4 u = *q;
                *q = make_uint4(bf16x2_to_half2(u.x), bf16x2_to_half2(u.y), bf16x2_to_half2(u.z), bf16x2_to_half2(u.w));
            }
        }
        __syncthreads();  // the warp's copies are visible to it; every warp is done with the previous exchange
        const __nv_bfloat16* const tile = stg + (F8 ? 0 : st * (L::T16 / 2));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mt * 16 >= rb) break;
            if (mt) __syncthreads();  // the exchange of row tile mt - 1 is read
            consume(tile, mt, st, t == ntile, begin + t * KT, wbase(t));
        }
        win_store(rwv);  // every warp is past its bias of tile t
        if constexpr (!F8) {
            __syncwarp();  // the warp is done with its part of the stage
            issue(t + NSTG, true);
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");

    // ---- output: bf16(bf16(o / (sum + e^sink)) * sigmoid(gate)) for rows 3 t + h, head 3 g + h, from the merged
    // (max, sum, o) of the row (log2 domain); the gates are fetched first, all in flight together
    auto gate_of = [&](const int row, const int dim) {
        return pj2(size_t(row / GQA) * R + OFF_GATE + (g * GQA + row % GQA) * D + dim);
    };
    auto out_pair = [&](const int row, const int dim, const float2 ov, const float m_, const float l_, const float sinkv,
                        const float2 gv) {
        const float sink = sinkv * LOG2E, mx = fmaxf(sink, m_);
        const float f = exp2f(m_ - mx);
        const float l = exp2f(sink - mx) + l_ * f;
        const float g0 = 1.f / (1.f + expf(-gv.x)), g1 = 1.f / (1.f + expf(-gv.y));
        reinterpret_cast<__nv_bfloat162*>(out_z + size_t(row / GQA) * HQ * D + (g * GQA + row % GQA) * D + dim)[0] =
            __floats2bfloat162_rn(bf16r(ov.x * f / l) * g0, bf16r(ov.y * f / l) * g1);
    };
    if (nact == 1) {
        float2 gv[MT][2][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = min(mt * 16 + gid + 8 * h, rb - 1);
#pragma unroll
                for (int n = 0; n < 4; ++n) gv[mt][h][n] = gate_of(row, warp * 32 + n * 8 + 2 * t4);
            }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + gid + 8 * h;
                if (row < rb) {
#pragma unroll
                    for (int n = 0; n < 4; ++n)
                        out_pair(row, warp * 32 + n * 8 + 2 * t4, make_float2(o[mt][n][2 * h], o[mt][n][2 * h + 1]),
                                 mrow[mt][h], lrow[mt][h], sk[mt][h], gv[mt][h][n]);
                }
            }
    } else {
        // this split's partial to the workspace (L2); the head's last block merges
        float* const part_h = p.part + size_t(z * HKV + g) * S * ROWS * PSTR;
        float* const part = part_h + size_t(s) * ROWS * PSTR;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + gid + 8 * h;
                if (row < rb) {
#pragma unroll
                    for (int n = 0; n < 4; ++n)
                        __stcg(reinterpret_cast<float2*>(part + row * PSTR + warp * 32 + n * 8 + 2 * t4),
                               make_float2(o[mt][n][2 * h], o[mt][n][2 * h + 1]));
                    if (warp == 0 && t4 == 0)
                        __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mrow[mt][h], lrow[mt][h]));
                }
            }
        // thread items (row, dim pair) of the ni = rb D / 2 for the merge; its gates and sinks fetched before the
        // count, the first QU partials together with the (max, sum)s
        constexpr int IT = (ROWS * (D / 2) + NT - 1) / NT, QU = 16 / IT < 1 ? 1 : 16 / IT > 8 ? 8 : 16 / IT;
        const int ni = rb * (D / 2);
        auto item = [&](const int it, int& row, int& dim) {
            const int i = min(tid + it * NT, ni - 1);
            row = i / (D / 2), dim = 2 * (i % (D / 2));
        };
        float skm[IT];
        float2 gv[IT], a[IT];
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            int row, dim;
            item(it, row, dim);
            skm[it] = sink_of(row);
            gv[it] = gate_of(row, dim);
            a[it] = make_float2(0.f, 0.f);
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            const bool last = atomicAdd(counter_z + g, 1) == nact - 1;
            if (last) counter_z[g] = 0;
            *flag = last;
        }
        __syncthreads();
        if (!*flag) return;
        __threadfence();
        float2 ov[QU][IT];
        auto load_ov = [&](const int q0) {
#pragma unroll
            for (int u = 0; u < QU; ++u)
#pragma unroll
                for (int it = 0; it < IT; ++it) {
                    int row, dim;
                    item(it, row, dim);
                    ov[u][it] = __ldcg(reinterpret_cast<const float2*>(
                        part_h + (size_t(min(q0 + u, nact - 1)) * ROWS + row) * PSTR + dim));
                }
        };
        load_ov(0);
        // the partials' (max, sum) to shared memory (the stages are free), then per row the merged (max, sum)
        // and each partial's weight exp2(max_q - max)
        float2* const mlq = reinterpret_cast<float2*>(smem);  // [rb][nact]
        float* const wq = reinterpret_cast<float*>(mlq + rb * nact);  // [rb][nact]
        float2* const rml = reinterpret_cast<float2*>(wq + rb * nact + (rb * nact) % 2);  // [rb]
        for (int i = tid; i < rb * nact; i += NT)
            mlq[i] = __ldcg(reinterpret_cast<const float2*>(part_h + (size_t(i % nact) * ROWS + i / nact) * PSTR + D));
        __syncthreads();
        for (int row = tid; row < rb; row += NT) {
            float mx = NEG_INIT, l = 0.f;
            for (int q = 0; q < nact; ++q) mx = fmaxf(mx, mlq[row * nact + q].x);
            for (int q = 0; q < nact; ++q) {
                const float f = exp2f(mlq[row * nact + q].x - mx);
                wq[row * nact + q] = f;
                l += mlq[row * nact + q].y * f;
            }
            rml[row] = make_float2(mx, l);
        }
        __syncthreads();
        for (int q0 = 0; q0 < nact; q0 += QU) {
            if (q0) load_ov(q0);
#pragma unroll
            for (int u = 0; u < QU; ++u) {
                if (q0 + u >= nact) break;
#pragma unroll
                for (int it = 0; it < IT; ++it) {
                    int row, dim;
                    item(it, row, dim);
                    const float f = wq[row * nact + q0 + u];
                    a[it].x += ov[u][it].x * f, a[it].y += ov[u][it].y * f;
                }
            }
        }
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const int i = tid + it * NT, row = i / (D / 2), dim = 2 * (i % (D / 2));
            if (i < ni) out_pair(row, dim, a[it], rml[row].x, rml[row].y, skm[it], gv[it]);
        }
    }

    // ---- deferred ring rows (mb > 1, wrapped ring): every block of the head is done reading them
    if (defer) {
        for (int i = tid; i < 2 * mb * (D / 8); i += NT) {
            const int kv = i / (mb * (D / 8)), m = (i / (D / 8)) % mb, c8 = i % (D / 8);
            const uint4 u = __ldcg(reinterpret_cast<const uint4*>(p.nkv + ((size_t(z * HKV + g) * 2 + kv) * M + m) * D) + c8);
            reinterpret_cast<uint4*>((kv ? vc : kc) + (size_t((p0 + m) & (WIN - 1)) * HKV + g) * D)[c8] = u;
        }
        if (tid == 0) {  // kpos is read by every head: the last head writes it
            __threadfence();
            if (atomicAdd(counter_z + HKV, 1) == HKV - 1) {
                __threadfence();
                for (int m = 0; m < mb; ++m) kpos_z[(p0 + m) & (WIN - 1)] = p0 + m;
                counter_z[HKV] = 0;
            }
        }
    }
}

using KernelFn = void (*)(Params);

struct Kernel {
    KernelFn fn;
    int smem;
};

template <int M, bool FULL, bool F8, bool RAG>
Kernel kernel_of() {
    static const Kernel k = [] {
        const KernelFn f = attn_batch_kernel<M, FULL, F8, RAG>;
        CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<M, F8>::BYTES));
        return Kernel{f, Lay<M, F8>::BYTES};
    }();
    return k;
}

template <bool FULL, bool F8, bool RAG>
Kernel select_m(int M) {
    switch (M) {
        case 1: return kernel_of<1, FULL, F8, RAG>();
        case 2: return kernel_of<2, FULL, F8, RAG>();
        case 3: return kernel_of<3, FULL, F8, RAG>();
        case 4: return kernel_of<4, FULL, F8, RAG>();
        case 5: return kernel_of<5, FULL, F8, RAG>();
        case 6: return kernel_of<6, FULL, F8, RAG>();
        case 7: return kernel_of<7, FULL, F8, RAG>();
        default: return kernel_of<8, FULL, F8, RAG>();
    }
}

template <bool FULL, bool F8 = false>
Kernel select(int M, bool ragged) {
    return ragged ? select_m<FULL, F8, true>(M) : select_m<FULL, F8, false>(M);
}

int sm_count() {
    static const int n = [] {
        int dev = 0, v = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
        return v;
    }();
    return n;
}

// splits per (sequence, KV head) without the cache-size cap: as many as keep every block resident at once
int max_splits(int batch) {
    return std::clamp(BLOCKS_PER_SM * sm_count() / (HKV * batch), 1, MAX_SPLITS);
}

constexpr size_t round256(size_t n) { return (n + 255) / 256 * 256; }
size_t ws_counter_bytes(int batch) { return round256(size_t(batch) * CNT * 4); }
size_t ws_nkv_bytes(int M, int batch) { return round256(size_t(batch) * HKV * 2 * M * D * 2); }
size_t ws_part_bytes(int M, size_t units) { return units * HKV * GQA * M * PSTR * 4; }  // units = batch x splits

}  // namespace
int batched_splits(int batch, std::int64_t cache_rows) {
    const std::int64_t cap = std::max<std::int64_t>(1, (cache_rows + SPLIT_KEYS - 1) / SPLIT_KEYS);
    return int(std::min<std::int64_t>(max_splits(batch), cap));
}

std::size_t batched_workspace_bytes(int max_m, int max_batch) {
    require(max_m >= 1 && max_m <= MAXM, "max_m must be in 1..8");
    require(max_batch >= 1 && max_batch <= 65535, "max_batch must be in 1..65535");
    // b * max_splits(b) <= max(resident blocks per KV head, b) for every batch b
    const size_t units = std::max<size_t>(size_t(BLOCKS_PER_SM) * sm_count() / HKV, max_batch);
    return ws_counter_bytes(max_batch) + ws_nkv_bytes(max_m, max_batch) + ws_part_bytes(max_m, units);
}

void attn_decode_batched(const DecodeArgs& a, cudaStream_t stream) {
    check_args(a);
    const bool full = a.window == 0;
    const int M = a.M, S = batched_splits(a.batch, a.cache_rows);
    const size_t cb = ws_counter_bytes(a.batch), nb = ws_nkv_bytes(M, a.batch);
    require(a.workspace_size >= cb + nb + ws_part_bytes(M, size_t(a.batch) * S),
            "workspace too small: need workspace_bytes(M, batch)");
    require(S <= 65535, "too many key splits");

    Params p{};
    p.proj = a.proj;
    p.proj16 = a.proj16;
    p.conv_write = a.conv_write;
    p.R = full ? R_FULL : R_SLIDING;
    p.cap = a.cache_rows;
    p.log_scaling = a.log_scaling;
    p.q_scale = a.q_scale;
    p.r_scale = a.r_scale;
    p.rel_proj = a.rel_proj;
    p.k_conv = a.k_conv;
    p.v_conv = a.v_conv;
    p.conv_k = a.conv_k_state;
    p.conv_v = a.conv_v_state;
    p.k_cache = a.k_cache;
    p.v_cache = a.v_cache;
    p.kpos = full ? nullptr : reinterpret_cast<long long*>(a.kpos);
    p.pos = reinterpret_cast<const long long*>(a.pos);
    char* const ws = static_cast<char*>(a.workspace);
    p.counter = reinterpret_cast<int*>(ws);
    p.nkv = reinterpret_cast<__nv_bfloat16*>(ws + cb);
    p.part = reinterpret_cast<float*>(ws + cb + nb);
    p.out = a.out;
    p.slots = a.slots;
    p.rows = a.rows;
    p.cache_stride = a.cache_stride;
    p.kpos_stride = a.kpos_stride;
    p.conv_stride = a.conv_stride;
    const bool rag = a.rows != nullptr;
    const Kernel k = full ? (a.cache_fp8 ? select<true, true>(M, rag) : select<true>(M, rag)) : select<false>(M, rag);
    // programmatic dependent launch (the kernel waits with griddepcontrol.wait before touching anything an
    // earlier kernel may write; a no-op without PDL)
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(S, HKV, a.batch);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = k.smem;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, k.fn, p));
}

}  // namespace ingine::kernels::attn
