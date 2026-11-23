// Decode attention for a batch of sequences (attn_decode with batch > 1). Contract: attn.h / native/KERNELS.md,
// area `attn`; the arithmetic and rounding points are the cluster kernel's (attn.cu) M > 1 path.
//
// Design
// ------
// * Work: one block of 4 warps per (key split s, KV head g, sequence z): grid (S, 8, batch). Shared memory is
//   sized for four resident blocks per SM (680 on the RTX 5090), so batch 64 (512 blocks per split) runs in
//   one wave, instead of the cluster kernel's 16 x 8 blocks per sequence with their fixed per-block latency
//   (~200 us per visit at batch 64). A block takes its 3 GQA heads x M tokens as rows r = 3 t + h.
// * Splits: the host picks the largest S whose blocks are all resident at once (S = 1 at batch 64, more for
//   small batches), capped at 256 cached keys per split; each sequence divides its cached keys [0, L_old)
//   into contiguous chunks of `per` keys (from pos on the device, so the launch shape is fixed for a CUDA
//   graph) and blocks past the last chunk exit. Split 0 also takes the call's own M keys. With one chunk the
//   block writes the output itself; otherwise every block writes its partial (unnormalised o, max, sum) to
//   the workspace and the last one of the head (counter) merges them.
// * Prologue: q / rel scales and conv taps are read before griddepcontrol.wait; after it, pos, then every
//   projection row and conv state the block needs in one burst of loads (ahead of the K/V tiles, which
//   would delay them), then the first tile.
// * Keys are streamed in tiles of 16 with cp.async, double-buffered: K and V rows (16-byte chunks XOR-swizzled
//   by key) and ring kpos. BF16 V rows are converted to fp16 in place; E4M3 tiles are widened (K to bf16, V to
//   fp16, both exact) into one 16-bit tile. Per tile, the block first writes the bias of every (row, key) to
//   shared memory (bf16(rel . rel_proj[:, d]) for d < 1024, rel_proj through L1; 0 beyond; -inf for a masked
//   key), then each warp: S = Q K^T (mma m16n8k16 bf16, exact bf16 q and k, fp32 sums, 1/sqrt(128) applied to
//   the sums) for all rows and the tile's keys, online softmax in the log2 domain, and O += P V for its own 32
//   dims (mma fp16, P as fp16 hi + lo (P - hi), ~22 bits; V fp16). Every warp computes the same S and row
//   statistics, so no exchange is needed between them; P stays in registers (the S accumulators are the A
//   fragments of P V).
// * Split 0's warps 0 and 1 run the key / value short convolutions of the head (+ key clip / value RMS norm,
//   lanes holding 4 channels as the cluster kernel, so the reductions and bits match), write the advanced
//   conv state (no other block reads these channels) and put the new keys / values (BF16, then E4M3 for FP8
//   caches) into a 16-key tile it attends first. Cache rows: full layers write rows p0.. at once (other
//   blocks only read rows below p0); sliding rings too, unless M > 1 and the ring has wrapped
//   (p0 + M - 1 >= 1024): then the overwritten slots still hold keys earlier tokens of the call see, so the
//   new rows go to the workspace and the head's last block writes them, and the last head (sequence
//   counter) writes kpos. M = 1 overwrites the slot at distance 1024 (invalid either way; a kpos >= p0 is
//   ignored too) and writes kpos at once.
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
    static constexpr int KM = KT / 16;           // 16-key mma steps per tile
    static constexpr int NSTG = 2;               // tiles in the cp.async ring
    static constexpr int T16 = KT * 2 * D * 2;   // a 16-bit tile [KT][K | V][D]
    // BF16: the cp.async stages (V converted to fp16 in place); E4M3: the widened tile
    static constexpr int STG = 0;
    static constexpr int RAW = STG + (F8 ? 1 : NSTG) * T16;          // E4M3 stages [NSTG][K | V][KT][D]
    static constexpr int KPS = RAW + (F8 ? NSTG * 2 * KT * D : 0);   // ring kpos low words, int [NSTG][KT]
    static constexpr int QB = KPS + NSTG * KT * 4;                   // q bf16 [ROWS][D], chunks swizzled by row
    static constexpr int RELD = QB + ROWS * D * 2;                   // rel bf16 [ROWS][DREL]
    static constexpr int BIAS = RELD + ROWS * DREL * 2;              // bf16 [ROWS][KT]
    static constexpr int FLAG = BIAS + ROWS * KT * 2;                // int: this block merges the head
    static constexpr int BYTES = FLAG + 16;
    // four blocks per SM: 100 KB of shared memory per SM, 1 KB of it reserved per block (sm_120)
    static_assert(BYTES + 1024 <= 102400 / BLOCKS_PER_SM, "shared memory for four blocks per SM");
    // the merge's (max, sum) and weights per row and partial, and the merged (max, sum) per row
    static_assert(ROWS * MAX_SPLITS * 12 + ROWS * 8 + 8 <= KPS, "merge scratch fits the tile stages");
};

struct Params {
    const float* proj;
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
    long long cache_stride, kpos_stride, conv_stride;
    int zero;  // 0 at run time, opaque to the compiler (fenced griddepcontrol.wait)
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

template <int M, bool FULL, bool F8>
__global__ void __launch_bounds__(NT, 4) attn_batch_kernel(const Params p) {
    static_assert(FULL || !F8, "FP8 caches are full-layer only");
    using CT = std::conditional_t<F8, __nv_fp8_storage_t, __nv_bfloat16>;  // cache element
    using L = Lay<M, F8>;
    constexpr int ROWS = L::ROWS, MT = L::MT, KT = L::KT, KM = L::KM, NSTG = L::NSTG;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* const stg = reinterpret_cast<__nv_bfloat16*>(smem + L::STG);
    unsigned char* const raw = smem + L::RAW;
    int* const kps = reinterpret_cast<int*>(smem + L::KPS);
    __nv_bfloat16* const qb = reinterpret_cast<__nv_bfloat16*>(smem + L::QB);
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
    const float* const proj_z = p.proj + size_t(z) * M * R;
    __nv_bfloat16* const out_z = p.out + size_t(z) * M * HQ * D;
    int* const counter_z = p.counter + size_t(z) * CNT;

    // weights (no kernel writes them) before the wait: q / rel scales (warps 2, 3), conv taps (split 0's warps 0, 1)
    constexpr int NQ = (ROWS + 1) / 2, NR = (ROWS + 3) / 4;  // q rows per warp, rel rows per half-warp
    const int w2 = warp - 2, kv = warp;  // q / rel warp index; conv warp: 0 = k, 1 = v
    const int ch0 = g * D + 4 * lane;    // conv warp lane: channels ch0 .. ch0 + 3
    float4 qs[GQA], w4[4];               // w4 sliding: [channel][tap] (conv1d order); full: [lag][channel]
    float rs[GQA];
    if (warp >= 2) {
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
    const int p0 = int(ld_s64(p.pos + z));
    // the projection rows and conv state next, all in flight together and ahead of the K/V tiles
    float4 qa[NQ], x4[3 + M];  // x4: [conv state 0..2, x_0 .. x_{M-1}]
    float ra[NR], dynv = 0.f;  // dynv: full layers' dynamic tap (lane / 4, lane % 4) = (token, lag)
    float* const cst = (kv ? p.conv_v : p.conv_k) + slot * p.conv_stride + ch0;
    if (warp >= 2) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int row = min(w2 + 2 * k, ROWS - 1);
            qa[k] = __ldcg(reinterpret_cast<const float4*>(proj_z + size_t(row / GQA) * R + (g * GQA + row % GQA) * D) + lane);
        }
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            const int row = min(w2 * 2 + half + 4 * k, ROWS - 1);
            ra[k] = __ldcg(proj_z + size_t(row / GQA) * R + OFF_REL + (g * GQA + row % GQA) * DREL + c);
        }
    } else if (s == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j) x4[j] = __ldcg(reinterpret_cast<const float4*>(cst + j * CONV_C));
#pragma unroll
        for (int m = 0; m < M; ++m)
            x4[3 + m] = __ldcg(reinterpret_cast<const float4*>(proj_z + size_t(m) * R + (kv ? OFF_V : OFF_K) + ch0));
        if (FULL && lane < 4 * M) dynv = __ldcg(proj_z + size_t(lane / 4) * R + (kv ? OFF_VDYN : OFF_KDYN) + g * 4 + lane % 4);
    }
    // out of cache capacity (uniform: no block reaches a barrier); a ring below 1024 rows must not wrap
    if ((FULL || p.cap < WIN) && p0 + M > p.cap) return;
    // cached keys: the linear cache up to p0; the ring's 1024 slots, or before it wraps slots 0..p0-1
    const int L_old = FULL ? p0 : min(p0, WIN);
    const int per = (max(SPLIT_KEYS, (L_old + S - 1) / S) + KT - 1) / KT * KT;
    const int nact = L_old > 0 ? (L_old + per - 1) / per : 1;
    if (s >= nact) return;
    const int begin = s * per, end = min(L_old, begin + per), ntile = (end - begin + KT - 1) / KT;
    // M > 1 on a wrapped ring: the new rows overwrite slots earlier tokens of the call still read
    const bool defer = !FULL && M > 1 && p0 + M - 1 >= WIN;

    // ---- tile t of this split: cp.async of its K / V rows (and ring kpos); always one group
    const std::uint64_t kvpol = l2::policy<KV_L2>();
    auto issue = [&](const int t) {
        const int st = t % NSTG, kb = begin + t * KT;
        if (t < ntile) {
            if constexpr (F8) {
                for (int i = tid; i < KT * 2 * (D / 16); i += NT) {
                    const int key = i / (2 * (D / 16)), kv = (i / (D / 16)) % 2, ch = i % (D / 16), j = kb + key;
                    const bool ok = j < end;
                    cp16z(reinterpret_cast<float*>(raw + ((st * 2 + kv) * KT + key) * D + ch * 16),
                          reinterpret_cast<const float*>((kv ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + ch * 16),
                          ok, kvpol);
                }
            } else {
                for (int i = tid; i < KT * 2 * (D / 8); i += NT) {
                    const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8), j = kb + key;
                    const bool ok = j < end;
                    cp16z(reinterpret_cast<float*>(stg + st * (L::T16 / 2) + toff(key, kv, ch)),
                          reinterpret_cast<const float*>((kv ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + ch * 8),
                          ok, kvpol);
                }
            }
            if constexpr (!FULL) {
                for (int i = tid; i < KT; i += NT)  // low word of the int64 position
                    cp4z(reinterpret_cast<float*>(kps + st * KT + i),
                         reinterpret_cast<const float*>(kpos_z + (kb + i < end ? kb + i : 0)), kb + i < end);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
#pragma unroll
    for (int t = 0; t < NSTG - 1; ++t) issue(t);

    // ---- q / rel post-processing (warps 2, 3) and, on split 0, the new keys / values (warps 0, 1)
    auto tau_of = [&](int t) {
        return p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)) : 1.f;
    };
    // the call's own keys: a 16-key tile (BF16: the last stage, not issued yet; E4M3: the widened tile)
    __nv_bfloat16* const tnew = stg + (F8 ? 0 : (NSTG - 1) * (L::T16 / 2));
    if (warp >= 2) {
        auto pick = [](const auto (&v)[GQA], int i) { return i == 0 ? v[0] : i == 1 ? v[1] : v[2]; };
#pragma unroll
        for (int k = 0; k < NQ; ++k) {  // q: one row per warp, lane = 4 dims
            const int rq = w2 + 2 * k, row = min(rq, ROWS - 1), i = row % GQA, t = row / GQA;
            const float4 a = qa[k], sc = pick(qs, i);
            float x[4] = {a.x * sc.x, a.y * sc.y, a.z * sc.z, a.w * sc.w};
            const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
            const float tau = tau_of(t);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                x[e] = bf16r(x[e] * f);
                if (p.log_scaling) x[e] = bf16r(x[e] * tau);
            }
            if (rq < ROWS)  // exact bf16 q for the tensor cores (scale applied to the scores)
                reinterpret_cast<uint2*>(qb + row * D + (((lane >> 1) ^ (row & 7)) * 8) + (lane & 1) * 4)[0] =
                    make_uint2(pack2(x[0], x[1]), pack2(x[2], x[3]));
        }
#pragma unroll
        for (int k = 0; k < NR; ++k) {  // rel: one row per half-warp
            const int rr = w2 * 2 + half + 4 * k, row = min(rr, ROWS - 1), i = row % GQA, t = row / GQA;
            const float r = ra[k] * pick(rs, i);
            float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
            if (p.log_scaling) y = bf16r(y * tau_of(t));
            if (rr < ROWS) reld[row * DREL + c] = __float2bfloat16_rn(y);  // exact: y is BF16-valued
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
        // the advanced conv state (only this block reads these channels)
#pragma unroll
        for (int j = 0; j < 3; ++j) __stcg(reinterpret_cast<float4*>(cst + j * CONV_C), x4[M + j]);
        // key clip / value RMS norm per token, the cache values, the new-key tile and the cache rows
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float ss = wsum(y[m][0] * y[m][0] + y[m][1] * y[m][1] + y[m][2] * y[m][2] + y[m][3] * y[m][3]);
            const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
            float v[4] = {y[m][0] * f, y[m][1] * f, y[m][2] * f, y[m][3] * f};
            if constexpr (F8) {  // the values the cache will hold: BF16, then E4M3 (exact in BF16)
#pragma unroll
                for (int e = 0; e < 4; ++e) v[e] = e4m3r(bf16r(v[e]));
            }
            const uint2 u = make_uint2(pack2(v[0], v[1]), pack2(v[2], v[3]));
            reinterpret_cast<uint2*>(tnew + toff(m, kv, lane >> 1) + (lane & 1) * 4)[0] =
                kv ? make_uint2(bf16x2_to_half2(u.x), bf16x2_to_half2(u.y)) : u;
            if (defer) {
                reinterpret_cast<uint2*>(p.nkv + ((size_t(z * HKV + g) * 2 + kv) * M + m) * D + 4 * lane)[0] = u;
            } else {
                const int row = FULL ? p0 + m : (p0 + m) & (WIN - 1);
                CT* const dst = (kv ? vc : kc) + (size_t(row) * HKV + g) * D + 4 * lane;
                if constexpr (F8) *reinterpret_cast<uint32_t*>(dst) = pack_e4m3x2(v[0], v[1]) | pack_e4m3x2(v[2], v[3]) << 16;
                else *reinterpret_cast<uint2*>(dst) = u;
            }
        }
#pragma unroll
        for (int m = M; m < 16; ++m)
            reinterpret_cast<uint2*>(tnew + toff(m, kv, lane >> 1) + (lane & 1) * 4)[0] = make_uint2(0u, 0u);
        if (!FULL && !defer && g == 0 && kv == 0 && lane < M) kpos_z[(p0 + lane) & (WIN - 1)] = p0 + lane;
    }
    __syncthreads();
    // dependents may launch (they wait for this grid's completion before reading its output)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- bias of a tile's (row, key) pairs: bf16(rel . rel_proj[:, d]) for d < 1024, 0 beyond, -inf masked.
    // rel_proj columns through L1 (consecutive lanes read consecutive distances; the window of a tile overlaps
    // the next one's, and every block of the SM reads the same layer's rel_proj).
    auto bias_tile = [&](const int st, const int kb, const bool is_new) {
        for (int i = tid; i < ROWS * KT; i += NT) {
            const int row = i / KT, key = i % KT, t = row / GQA, j = kb + key;
            const int kp = is_new ? (key < M ? p0 + key : -1) : (j < end ? (FULL ? j : kps[st * KT + key]) : -1);
            const int d = p0 + t - kp;
            // old entries need kp < p0 (and the window on sliding layers)
            const bool ok = kp >= 0 && d >= 0 && (is_new || (kp < p0 && (FULL || d < WIN)));
            float bt = -INFINITY;
            if (ok) {
                bt = 0.f;
                if (d < EXT) {
                    const uint4* rl4 = reinterpret_cast<const uint4*>(reld + row * DREL);
                    const uint4 r0 = rl4[0], r1 = rl4[1];
                    float rl[DREL];
                    unpack8(r0, rl);
                    unpack8(r1, rl + 8);
                    float b = 0.f;
#pragma unroll
                    for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], __ldg(p.rel_proj + k * EXT + d), b);
                    bt = bf16r(b);
                }
            }
            bias[i] = __float2bfloat16_rn(bt);  // exact: BF16-valued, 0 or -inf
        }
    };

    // ---- per warp: S over every row, online softmax, O += P V for dims 32 warp .. 32 warp + 31
    float o[MT][4][4], mrow[MT][2], lrow[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int n = 0; n < 4; ++n) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
        mrow[mt][0] = mrow[mt][1] = NEG_INIT, lrow[mt][0] = lrow[mt][1] = 0.f;
    }
    auto consume = [&](const __nv_bfloat16* tile, const int nkm) {  // nkm: 16-key steps in use
        float sa[MT][2 * KM][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int n = 0; n < 2 * KM; ++n) sa[mt][n][0] = sa[mt][n][1] = sa[mt][n][2] = sa[mt][n][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {  // padding rows read row ROWS - 1 (masked)
                const int r = min(mt * 16 + (lane & 15), ROWS - 1), ch = 2 * ks + (lane >> 4);
                ldsm_x4(smem_u32(qb + r * D + ((ch ^ (r & 7)) * 8)), af[mt]);
            }
#pragma unroll
            for (int kk = 0; kk < KM; ++kk) {
                if (kk >= nkm) break;
                const int q = lane >> 3, key = kk * 16 + (q >> 1) * 8 + (lane & 7);
                uint32_t bf[4];
                ldsm_x4(smem_u32(tile + toff(key, 0, 2 * ks + (q & 1))), bf);
                const uint32_t b0[2] = {bf[0], bf[1]}, b1[2] = {bf[2], bf[3]};
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma_bf16(sa[mt][2 * kk], af[mt], b0);
                    mma_bf16(sa[mt][2 * kk + 1], af[mt], b1);
                }
            }
        }
        // scores (log2 domain) with bias and mask; c0, c1: row gid, keys 2 t4, +1; c2, c3: row gid + 8
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + gid + 8 * h;
                float mx = mrow[mt][h];
#pragma unroll
                for (int n = 0; n < 2 * KM; ++n) {
                    if (n >= 2 * nkm) break;
                    float2 bt = make_float2(-INFINITY, -INFINITY);
                    if (row < ROWS)
                        bt = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + row * KT + n * 8 + 2 * t4));
                    float& s0 = sa[mt][n][2 * h];
                    float& s1 = sa[mt][n][2 * h + 1];
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
                for (int n = 0; n < 2 * KM; ++n) {
                    if (n >= 2 * nkm) break;
                    float& s0 = sa[mt][n][2 * h];
                    float& s1 = sa[mt][n][2 * h + 1];
                    s0 = exp2f(s0 - mx), s1 = exp2f(s1 - mx);
                    sum += s0 + s1;
                }
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                lrow[mt][h] = lrow[mt][h] * sc + sum;
            }
        }
        // O += P V: P (rows x 16 keys) = the S accumulators of key tiles 2 kk, 2 kk + 1 (fp16 hi + lo)
#pragma unroll
        for (int kk = 0; kk < KM; ++kk) {
            if (kk >= nkm) break;
            uint32_t ah[MT][4], al[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                split_half2(sa[mt][2 * kk][0], sa[mt][2 * kk][1], ah[mt][0], al[mt][0]);
                split_half2(sa[mt][2 * kk][2], sa[mt][2 * kk][3], ah[mt][1], al[mt][1]);
                split_half2(sa[mt][2 * kk + 1][0], sa[mt][2 * kk + 1][1], ah[mt][2], al[mt][2]);
                split_half2(sa[mt][2 * kk + 1][2], sa[mt][2 * kk + 1][3], ah[mt][3], al[mt][3]);
            }
#pragma unroll
            for (int dn = 0; dn < 2; ++dn) {
                const int q = lane >> 3, key = kk * 16 + (q & 1) * 8 + (lane & 7);
                uint32_t vb[4];
                ldsm_x4_t(smem_u32(tile + toff(key, 1, warp * 4 + dn * 2 + (q >> 1))), vb);
                const uint32_t b0[2] = {vb[0], vb[1]}, b1[2] = {vb[2], vb[3]};
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma_f16(o[mt][2 * dn], ah[mt], b0);
                    mma_f16(o[mt][2 * dn], al[mt], b0);
                    mma_f16(o[mt][2 * dn + 1], ah[mt], b1);
                    mma_f16(o[mt][2 * dn + 1], al[mt], b1);
                }
            }
        }
    };

    // ---- the call's own keys first, while the first tiles are in flight
    if (s == 0) {
        bias_tile(0, 0, true);
        __syncthreads();
        consume(tnew, 1);
    }
    // ---- the cached keys of this split
    for (int t = 0; t < ntile; ++t) {
        const int st = t % NSTG;
        asm volatile("cp.async.wait_group %0;" ::"n"(NSTG - 2) : "memory");  // this thread's part of tile t
        if constexpr (!F8) {  // V rows to fp16: each thread converts the chunks it copied
            for (int i = tid; i < KT * 2 * (D / 8); i += NT) {
                const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8);
                if (kv) {
                    uint4* q = reinterpret_cast<uint4*>(stg + st * (L::T16 / 2) + toff(key, 1, ch));
                    const uint4 u = *q;
                    *q = make_uint4(bf16x2_to_half2(u.x), bf16x2_to_half2(u.y), bf16x2_to_half2(u.z), bf16x2_to_half2(u.w));
                }
            }
        }
        __syncthreads();  // tile t landed for everyone; the previous tile (stage, bias, widened tile) is consumed
        issue(t + NSTG - 1);
        if constexpr (F8) {  // E4M3 rows -> K bf16, V fp16 (exact)
            for (int i = tid; i < KT * 2 * (D / 16); i += NT) {
                const int key = i / (2 * (D / 16)), kv = (i / (D / 16)) % 2, ch = i % (D / 16);
                const uint4 r = *reinterpret_cast<const uint4*>(raw + ((st * 2 + kv) * KT + key) * D + ch * 16);
                const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // dims 16 ch + 8 h .. + 7
                    uint32_t u[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = e4m3x2_f2((w[2 * h + e / 2] >> (16 * (e % 2))) & 0xffffu);
                        u[e] = kv ? half2_of(f.x, f.y) : pack2(f.x, f.y);
                    }
                    *reinterpret_cast<uint4*>(stg + toff(key, kv, 2 * ch + h)) = make_uint4(u[0], u[1], u[2], u[3]);
                }
            }
        }
        bias_tile(st, begin + t * KT, false);
        __syncthreads();
        consume(stg + (F8 ? 0 : st * (L::T16 / 2)), KM);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");

    // ---- output: bf16(bf16(o / (sum + e^sink)) * sigmoid(gate)) for rows 3 t + h, head 3 g + h, from the merged
    // (max, sum, o) of the row (log2 domain); the gates and sinks are fetched first, all in flight together
    auto sink_at = [&](const int row) { return proj_z + size_t(row / GQA) * R + OFF_SINK + g * GQA + row % GQA; };
    auto gate_at = [&](const int row, const int dim) {
        return reinterpret_cast<const float2*>(proj_z + size_t(row / GQA) * R + OFF_GATE + (g * GQA + row % GQA) * D + dim);
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
        float sk[MT][2];
        float2 gv[MT][2][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = min(mt * 16 + gid + 8 * h, ROWS - 1);
                sk[mt][h] = __ldcg(sink_at(row));
#pragma unroll
                for (int n = 0; n < 4; ++n) gv[mt][h][n] = __ldcg(gate_at(row, warp * 32 + n * 8 + 2 * t4));
            }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + gid + 8 * h;
                if (row < ROWS) {
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
                if (row < ROWS) {
#pragma unroll
                    for (int n = 0; n < 4; ++n)
                        __stcg(reinterpret_cast<float2*>(part + row * PSTR + warp * 32 + n * 8 + 2 * t4),
                               make_float2(o[mt][n][2 * h], o[mt][n][2 * h + 1]));
                    if (warp == 0 && t4 == 0)
                        __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mrow[mt][h], lrow[mt][h]));
                }
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
        // the partials' (max, sum) to shared memory (the stages are free), then per row the merged (max, sum)
        // and each partial's weight exp2(max_q - max)
        float2* const mlq = reinterpret_cast<float2*>(smem);  // [ROWS][nact]
        float* const wq = reinterpret_cast<float*>(mlq + ROWS * nact);  // [ROWS][nact]
        float2* const rml = reinterpret_cast<float2*>(wq + ROWS * nact + (ROWS * nact) % 2);  // [ROWS]
        for (int i = tid; i < ROWS * nact; i += NT)
            mlq[i] = __ldcg(reinterpret_cast<const float2*>(part_h + (size_t(i % nact) * ROWS + i / nact) * PSTR + D));
        __syncthreads();
        for (int row = tid; row < ROWS; row += NT) {
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
        // thread items (row, dim pair); every load of a step in flight together
        constexpr int NI = ROWS * (D / 2), IT = (NI + NT - 1) / NT;
        float sk[IT];
        float2 gv[IT], a[IT];
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const int i = min(tid + it * NT, NI - 1), row = i / (D / 2), dim = 2 * (i % (D / 2));
            sk[it] = __ldcg(sink_at(row));
            gv[it] = __ldcg(gate_at(row, dim));
            a[it] = make_float2(0.f, 0.f);
        }
        for (int q = 0; q < nact; ++q) {
            float2 ov[IT];
#pragma unroll
            for (int it = 0; it < IT; ++it) {
                const int i = min(tid + it * NT, NI - 1), row = i / (D / 2), dim = 2 * (i % (D / 2));
                ov[it] = __ldcg(reinterpret_cast<const float2*>(part_h + (size_t(q) * ROWS + row) * PSTR + dim));
            }
#pragma unroll
            for (int it = 0; it < IT; ++it) {
                const int i = min(tid + it * NT, NI - 1), row = i / (D / 2);
                const float f = wq[row * nact + q];
                a[it].x += ov[it].x * f, a[it].y += ov[it].y * f;
            }
        }
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const int i = tid + it * NT, row = i / (D / 2), dim = 2 * (i % (D / 2));
            if (i < NI) out_pair(row, dim, a[it], rml[row].x, rml[row].y, sk[it], gv[it]);
        }
    }

    // ---- deferred ring rows (M > 1, wrapped ring): every block of the head is done reading them
    if (defer) {
        for (int i = tid; i < 2 * M * (D / 8); i += NT) {
            const int kv = i / (M * (D / 8)), m = (i / (D / 8)) % M, c8 = i % (D / 8);
            const uint4 u = __ldcg(reinterpret_cast<const uint4*>(p.nkv + ((size_t(z * HKV + g) * 2 + kv) * M + m) * D) + c8);
            reinterpret_cast<uint4*>((kv ? vc : kc) + (size_t((p0 + m) & (WIN - 1)) * HKV + g) * D)[c8] = u;
        }
        if (tid == 0) {  // kpos is read by every head: the last head writes it
            __threadfence();
            if (atomicAdd(counter_z + HKV, 1) == HKV - 1) {
                __threadfence();
                for (int m = 0; m < M; ++m) kpos_z[(p0 + m) & (WIN - 1)] = p0 + m;
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

template <int M, bool FULL, bool F8>
Kernel kernel_of() {
    static const Kernel k = [] {
        const KernelFn f = attn_batch_kernel<M, FULL, F8>;
        CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<M, F8>::BYTES));
        return Kernel{f, Lay<M, F8>::BYTES};
    }();
    return k;
}

template <bool FULL, bool F8 = false>
Kernel select(int M) {
    switch (M) {
        case 1: return kernel_of<1, FULL, F8>();
        case 2: return kernel_of<2, FULL, F8>();
        case 3: return kernel_of<3, FULL, F8>();
        case 4: return kernel_of<4, FULL, F8>();
        case 5: return kernel_of<5, FULL, F8>();
        case 6: return kernel_of<6, FULL, F8>();
        case 7: return kernel_of<7, FULL, F8>();
        default: return kernel_of<8, FULL, F8>();
    }
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
    p.cache_stride = a.cache_stride;
    p.kpos_stride = a.kpos_stride;
    p.conv_stride = a.conv_stride;
    const Kernel k = full ? (a.cache_fp8 ? select<true, true>(M) : select<true>(M)) : select<false>(M);
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
