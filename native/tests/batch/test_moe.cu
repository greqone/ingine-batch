// Batched expert GEMMs (kernels/batch/moe.h) against the prefill ones on identical inputs: synthetic NVFP4 and
// MXFP8 experts (all 128, checkpoint encodings), random router logits, T from 9 to 700 (every tile size, partial
// and multi-tile experts). Outputs agree up to BF16 rounding flips of a different FP32 summation order.
// ~2 GB of GPU memory.
#include "batch/moe_test.h"

using namespace ingine::test;
using namespace ingine::test::batch_moe;

namespace {

__device__ uint32_t mix(uint64_t i, uint32_t seed) {
  uint32_t h = uint32_t(i) * 2654435761u ^ uint32_t(i >> 32) * 40503u ^ seed;
  h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
  return h;
}

// kind 0: random bytes (NVFP4 codes); 1: E4M3 block scales 2^-2 .. 2^2; 2: E4M3 values |v| in [2^-6, 4);
// 3: E8M0 scales 2^-6 .. 2^-4; 4: bf16 activations in (-2, 2)
__global__ void fill(uint8_t* p, size_t n, uint32_t seed, int kind) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint32_t h = mix(i, seed);
    uint8_t v;
    switch (kind) {
      case 0: v = uint8_t(h); break;
      case 1: v = uint8_t(((5 + (h >> 8) % 4) << 3) | (h & 7)); break;
      case 2: v = uint8_t((h & 0x80) | ((1 + (h >> 8) % 9) << 3) | ((h >> 4) & 7)); break;
      default: v = uint8_t(121 + (h >> 8) % 3); break;
    }
    p[i] = v;
  }
}
__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint32_t h = mix(i, seed);
    p[i] = __bfloat16_as_ushort(__float2bfloat16(((h & 0xffffff) / 16777216.f - 0.5f) * 4.f));
  }
}
__global__ void fill_logits(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = ((mix(i, seed) & 0xffffff) / 16777216.f - 0.5f) * 6.f;
}

struct Experts {
  pk::MoeExperts ex;
  explicit Experts(int fmt) {
    const size_t c13 = fmt == 0 ? size_t(E) * 2 * I * H / 2 : size_t(E) * 2 * I * H;
    const size_t s13 = fmt == 0 ? size_t(E) * 2 * I * H / 16 : size_t(E) * 2 * I * H / 32;
    auto* w13 = dalloc<uint8_t>(c13);
    auto* sc13 = dalloc<uint8_t>(s13);
    auto* w2 = dalloc<uint8_t>(c13 / 2);
    auto* sc2 = dalloc<uint8_t>(s13 / 2);
    fill<<<2048, 256>>>(w13, c13, 1 + fmt, fmt == 0 ? 0 : 2);
    fill<<<2048, 256>>>(w2, c13 / 2, 3 + fmt, fmt == 0 ? 0 : 2);
    fill<<<2048, 256>>>(sc13, s13, 5 + fmt, fmt == 0 ? 1 : 3);
    fill<<<2048, 256>>>(sc2, s13 / 2, 7 + fmt, fmt == 0 ? 1 : 3);
    float* f13 = nullptr;
    float* f2 = nullptr;
    if (fmt == 0) {
      std::vector<float> f(E);
      Rng rng(11);
      for (auto& v : f) v = float(rng.uniform(0.01, 0.03));
      f13 = dalloc<float>(E), f2 = dalloc<float>(E);
      INGINE_TEST_CUDA(cudaMemcpy(f13, f.data(), E * 4, cudaMemcpyHostToDevice));
      for (auto& v : f) v = float(rng.uniform(0.01, 0.03));
      INGINE_TEST_CUDA(cudaMemcpy(f2, f.data(), E * 4, cudaMemcpyHostToDevice));
    }
    ex = {fmt, w13, sc13, f13, w2, sc2, f2};
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
  }
};

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  constexpr int maxT = 700;
  cudaStream_t s;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Bufs b(maxT);
  auto* x = dalloc<uint16_t>(size_t(maxT) * H);
  auto* logits = dalloc<float>(size_t(maxT) * E);
  auto* bias = dalloc<float>(E);
  fill_bf16<<<1024, 256>>>(x, size_t(maxT) * H, 21);
  fill_logits<<<256, 256>>>(logits, size_t(maxT) * E, 22);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  bool ok = true;
  for (int fmt : {0, 1}) {
    const Experts ex(fmt);
    for (int T : {9, 64, 128, 256, 512, 700}) {
      std::printf("-- %s T=%d (tile %d)\n", fmt == 0 ? "NVFP4" : "MXFP8", T, kb::moe_tile_rows(T));
      route(b, logits, bias, T, s);
      ok &= compare(b, ex.ex, x, T, s);
    }
  }
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
