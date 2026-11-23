#include "kernels/batch/batch.h"

#include "core/device.h"

#include <cmath>
#include <stdexcept>

namespace ingine::kernels::batch {
namespace {

__device__ __forceinline__ float bf2f(std::uint16_t v) { return __uint_as_float(std::uint32_t(v) << 16); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// one thread per (channel, sequence); the sequence's rows in order, as prefill::static_conv + conv_state_update
__global__ void conv_kernel(const std::uint16_t* __restrict__ x, int ld, int M, int C, const float* __restrict__ w,
                            float* __restrict__ state, long long slot_stride, const std::int32_t* __restrict__ slots,
                            float* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= C) return;
  float* const st = state + (long long)slots[b] * slot_stride;
  const float4 w4 = *reinterpret_cast<const float4*>(w + std::size_t(c) * 4);
  float e0 = st[c], e1 = st[C + c], e2 = st[2 * C + c];
  const float inv = 1.f / 2.2360679774997896f;
  for (int m = 0; m < M; ++m) {
    const std::size_t r = std::size_t(b) * M + m;
    const float e3 = bf2f(x[r * ld + c]);
    float acc = __fmul_rn(w4.x, e0);
    acc = __fadd_rn(acc, __fmul_rn(w4.y, e1));
    acc = __fadd_rn(acc, __fmul_rn(w4.z, e2));
    acc = __fadd_rn(acc, __fmul_rn(w4.w, e3));
    out[r * C + c] = __fmul_rn(__fadd_rn(acc, e3), inv);
    e0 = e1, e1 = e2, e2 = e3;
  }
  st[c] = e0, st[C + c] = e1, st[2 * C + c] = e2;
}

__global__ void widen_kernel(const std::uint16_t* __restrict__ src, int ld_src, int rows, int cols, float* __restrict__ dst,
                             int ld_dst) {
  const std::size_t n = std::size_t(rows) * cols;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t r = i / cols, c = i % cols;
    dst[r * ld_dst + c] = bf2f(src[r * ld_src + c]);
  }
}

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ bool better(float a, int av, float b, int bv) { return a > b || (a == b && av < bv); }

constexpr int SAMPLE_T = 1024;

__global__ void __launch_bounds__(SAMPLE_T) sample_kernel(const float* __restrict__ logits, int ld, int V, int raw,
                                                          float inv_t, int greedy, std::uint64_t seed,
                                                          std::int32_t* __restrict__ out) {
  const int row = blockIdx.x;
  const float* const l = logits + std::size_t(row) * ld;
  const std::uint64_t row_key = mix64(seed ^ (std::uint64_t(row) * 0xD1B54A32D192ED03ull));
  float best = -INFINITY;
  int best_v = 0x7fffffff;
  for (int v = threadIdx.x; v < V; v += SAMPLE_T) {
    float key = raw ? 30.f * tanhf(bf16r(l[v]) / 30.f) : l[v];  // model.logits
    if (!greedy) {
      // Gumbel-max: argmax_v s_v / T + G_v is a sample of softmax(s / T); u in (0, 1) from 24 random bits
      const float u = (float(mix64(row_key ^ std::uint64_t(v)) >> 40) + 0.5f) * (1.f / 16777216.f);
      key = key * inv_t - __logf(-__logf(u));
    }
    if (better(key, v, best, best_v)) best = key, best_v = v;
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int ov = __shfl_xor_sync(0xffffffffu, best_v, o);
    if (better(ob, ov, best, best_v)) best = ob, best_v = ov;
  }
  __shared__ float s_best[SAMPLE_T / 32];
  __shared__ int s_v[SAMPLE_T / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) s_best[warp] = best, s_v[warp] = best_v;
  __syncthreads();
  if (warp == 0) {
    best = s_best[lane], best_v = s_v[lane];  // SAMPLE_T / 32 == 32 warps
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int ov = __shfl_xor_sync(0xffffffffu, best_v, o);
      if (better(ob, ov, best, best_v)) best = ob, best_v = ov;
    }
    if (lane == 0) out[row] = best_v;
  }
}

}  // namespace

void conv(const __nv_bfloat16* x, int ld, int n, int M, int C, const float* w, float* state, std::int64_t slot_stride,
          const std::int32_t* slots, float* out, cudaStream_t stream) {
  if (n <= 0 || M <= 0) return;
  const dim3 grid((C + 127) / 128, n);
  conv_kernel<<<grid, 128, 0, stream>>>(reinterpret_cast<const std::uint16_t*>(x), ld, M, C, w, state, slot_stride, slots,
                                        out);
  CUDA_CHECK(cudaGetLastError());
}

void widen(const __nv_bfloat16* src, int ld_src, int rows, int cols, float* dst, int ld_dst, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  const std::size_t n = std::size_t(rows) * cols;
  const int blocks = int(std::min<std::size_t>((n + 255) / 256, 4096));
  widen_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const std::uint16_t*>(src), ld_src, rows, cols, dst, ld_dst);
  CUDA_CHECK(cudaGetLastError());
}

void sample(const float* logits, int ld, int n, int V, bool raw, float temperature, std::uint64_t seed,
            std::int32_t* out, cudaStream_t stream) {
  if (n <= 0) return;
  if (V <= 0 || V > ld) throw std::invalid_argument("batch::sample: V outside 1..ld");
  const bool greedy = !(temperature > 0.f);
  sample_kernel<<<n, SAMPLE_T, 0, stream>>>(logits, ld, V, raw ? 1 : 0, greedy ? 1.f : 1.f / temperature,
                                            greedy ? 1 : 0, seed, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::batch
