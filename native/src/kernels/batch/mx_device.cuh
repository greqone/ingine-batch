// Device side of the Mx (MXFP8) copies of kernels/batch/draft.h: the warp lanes holding a 32-element block share
// its amax, and each writes its elements as E4M3 of x / 2^e (e = ceil(log2(amax / 448)), the block's E8M0 scale).
#pragma once

#include "kernels/batch/draft.h"

#include <cuda_fp8.h>

namespace ingine::kernels::bdraft {

// the smallest e with 2^e >= amax / 448 (-127 for a zero block)
__device__ __forceinline__ int mx_exponent(float amax) {
  if (!(amax > 0.f)) return -127;
  int ex;
  const float m = frexpf(__fdiv_rn(amax, 448.f), &ex);  // amax / 448 = m 2^ex, m in [0.5, 1)
  return max(m == 0.5f ? ex - 1 : ex, -127);
}

__device__ __forceinline__ float mx_inverse(int e) { return __int_as_float((127 - e) << 23); }  // 2^-e, e >= -127

// 8 consecutive values y of row `row` at column col (col % 8 == 0) of [rows, K]; the 4 lanes 4j .. 4j + 3 of the
// warp hold one 32-element block. Every lane of the warp calls it (valid = false: no element, no store).
__device__ __forceinline__ void mx_store8(const float (&y)[8], const Mx& out, int K, int row, int col, bool valid) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) a = fmaxf(a, fabsf(y[j]));
  a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
  a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
  if (!valid) return;
  const int e = mx_exponent(a);
  const float inv = mx_inverse(e);
  uint32_t w[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(y[4 * h] * inv, y[4 * h + 1] * inv),
                                                              __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(y[4 * h + 2] * inv, y[4 * h + 3] * inv),
                                                              __NV_SATFINITE, __NV_E4M3);
    w[h] = uint32_t(lo) | (uint32_t(hi) << 16);
  }
  *reinterpret_cast<uint2*>(out.q + size_t(row) * K + col) = make_uint2(w[0], w[1]);
  if ((threadIdx.x & 3) == 0) out.sf[mx_sf_index(row, col / 32, K)] = uint8_t(e + 127);
}

// One value y of row `row` at column col; the 32 lanes of the warp hold one 32-element block (col % 32 = lane).
__device__ __forceinline__ void mx_store1(float y, const Mx& out, int K, int row, int col) {
  float a = fabsf(y);
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const int e = mx_exponent(a);
  out.q[size_t(row) * K + col] = __nv_cvt_float_to_fp8(y * mx_inverse(e), __NV_SATFINITE, __NV_E4M3);
  if ((threadIdx.x & 31) == 0) out.sf[mx_sf_index(row, col / 32, K)] = uint8_t(e + 127);
}

}  // namespace ingine::kernels::bdraft
