// Device helpers and layer constants shared by the decode attention kernels (attn.cu: one sequence, 16-block
// clusters; attn_batch.cu: many sequences, one block per sequence, KV head and key split).
#pragma once

#include "kernels/attn/attn.h"
#include "kernels/common/l2.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ingine::kernels::attn::detail {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, GQA = 3, CONV_C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int R_SLIDING = 8600, R_FULL = 8664;
constexpr int MAXM = 8;
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;  // 1/sqrt(128) * log2(e)
constexpr float NEG_INIT = -1e30f;
constexpr float INV_SQRT5 = 1.f / 2.2360679774997896f;  // torch: x / scalar == x * (1 / float(scalar))
constexpr float INV_3 = 1.f / 3.f;

inline void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("attn: ") + what);
}

// The DecodeArgs contract both launchers share (the workspace size is checked by each).
inline void check_args(const DecodeArgs& a) {
    require(a.window == 0 || a.window == WIN, "window must be 0 (full layer) or 1024 (sliding layer)");
    const bool full = a.window == 0;
    require(a.M >= 1 && a.M <= MAXM, "M must be in 1..8");
    require(a.batch >= 1 && a.batch <= 65535, "batch must be in 1..65535");
    require((a.proj == nullptr) != (a.proj16 == nullptr), "exactly one of proj / proj16");
    require(a.q_scale && a.r_scale && a.rel_proj && a.k_conv && a.v_conv && a.conv_k_state && a.conv_v_state &&
                a.k_cache && a.v_cache && a.pos && a.workspace && a.out,
            "null argument");
    require(full || !a.cache_fp8, "float8_e4m3fn caches are supported on full layers (window = 0) only");
    // device positions and loop bounds are int32; 2^30 leaves headroom for the tile strides
    require(a.cache_rows >= 1 && a.cache_rows <= (int64_t(1) << 30), "caches hold 1..2^30 positions");
    if (!full) {
        require(a.cache_rows <= WIN, "sliding caches hold at most 1024 rows");
        require(a.kpos != nullptr, "sliding layers need kpos");
    }
    require(a.batch == 1 || (a.cache_stride > 0 && a.conv_stride > 0 && (full || a.kpos_stride > 0)),
            "a batch needs the slot strides");
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// D += A (16x16, row) * B (16x8, col), fp32 accumulators
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ void unpack8(const uint4 u, float f[8]) {
    f[0] = __uint_as_float(u.x << 16), f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16), f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16), f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16), f[7] = __uint_as_float(u.w & 0xffff0000u);
}

// E4M3 pair (low byte = lower dim) to floats, exact (cvt.f16x2.e4m3x2, then FP16 -> FP32)
__device__ __forceinline__ float2 e4m3x2_f2(uint32_t pair) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3)));
}

// 8 E4M3 values (dims 8c..8c+7, byte 0 = lowest dim)
__device__ __forceinline__ void unpack8(const uint2 u, float f[8]) {
    float2 a = e4m3x2_f2(u.x & 0xffffu), b = e4m3x2_f2(u.x >> 16);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
    a = e4m3x2_f2(u.y & 0xffffu), b = e4m3x2_f2(u.y >> 16);
    f[4] = a.x, f[5] = a.y, f[6] = b.x, f[7] = b.y;
}

// x rounded to E4M3: nearest even, saturated to +-448 (torch: x.clamp(-448, 448).to(float8_e4m3fn))
__device__ __forceinline__ float e4m3r(float x) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3), __NV_E4M3)));
}

// two floats holding E4M3 values to their bytes (a in the low byte)
__device__ __forceinline__ uint32_t pack_e4m3x2(float a, float b) {
    return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

// Loads of anything an earlier kernel writes (caches, kpos, pos) are coherent `asm volatile` with a memory
// clobber, so the compiler cannot hoist them above griddepcontrol.wait (programmatic dependent launch).
// Cached K/V rows are read once per step (the next read of a ring row is a full step, ~1.5 GB of streams,
// later): their L2 lines are filled evict-first (kernels/common/l2.cuh).
constexpr l2::Evict KV_L2 = l2::Evict::first;

__device__ __forceinline__ uint4 ld_kv(const void* ptr, std::uint64_t pol) {
    uint4 v;
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(ptr), "l"(pol)
                     : "memory");
    else
        asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(ptr)
                     : "memory");
    return v;
}

__device__ __forceinline__ uint2 ld_kv8(const void* ptr, std::uint64_t pol) {
    uint2 v;
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                     : "=r"(v.x), "=r"(v.y)
                     : "l"(ptr), "l"(pol)
                     : "memory");
    else
        asm volatile("ld.global.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ long long ld_s64(const long long* ptr) {
    long long v;
    asm volatile("ld.global.s64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void cp16(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src)
                 : "memory");
}

__device__ __forceinline__ void cp4(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}

// zero-filling variants: nothing is read when !ok (cp16z: cached K/V rows, KV_L2 lines)
__device__ __forceinline__ void cp16z(float* dst, const float* src, bool ok, std::uint64_t pol) {
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;" ::"r"(
                         unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "r"(ok ? 16 : 0), "l"(pol)
                     : "memory");
    else
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "r"(ok ? 16 : 0)
                     : "memory");
}

__device__ __forceinline__ void cp4z(float* dst, const float* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// clip factor min(1, t / (sqrt(ss) + 1e-6)) without the IEEE sqrt / div slow-path branches (which
// serialise the six reductions of the q / rel prologue); differs from torch by a couple of FP32 ulp
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
    return fminf(1.f, __fdividef(t, norm + 1e-6f));
}

// bf16 / e4m3 values -> fp16 pair; anything not finite and in range (only possible in masked, unused cache
// slots) -> 0
__device__ __forceinline__ uint32_t half2_of(float a, float b) {
    a = fabsf(a) <= 65504.f ? a : 0.f;
    b = fabsf(b) <= 65504.f ? b : 0.f;
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// a bf16 pair (low half = lower dim) -> fp16 pair
__device__ __forceinline__ uint32_t bf16x2_to_half2(uint32_t w) {
    return half2_of(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

}  // namespace ingine::kernels::attn::detail
