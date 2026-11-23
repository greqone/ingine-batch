// Batched expert GEMMs (kernels/batch/moe.h) vs the prefill ones on real layer weights (NVFP4 layers 1, 2, 3, 5;
// MXFP8 layers 20, 21), rotating over the layers so the L2 holds none of the experts of the timed call. Routing:
// the layer's router and bias on random rows, or on real MLP inputs (INGINE_BENCH_X, below).
// Per T: gate_up + down time, effective bandwidth (bytes of the touched experts' codes and scales / time) and
// TFLOP/s, against max(memory, MMA) bound; then old vs new outputs on identical inputs. Needs the model (MODEL
// label) and ~6 GB of GPU memory. Runs only with INGINE_BENCH=1.
#include "batch/moe_test.h"
#include "model/runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using namespace ingine::test;
using namespace ingine::test::batch_moe;

namespace {

__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    p[i] = __bfloat16_as_ushort(__float2bfloat16(((h & 0xffffff) / 16777216.f - 0.5f) * scale));
  }
}

cudaStream_t g_stream;

// median of `reps` timings of f(rep)
float time_ms(const std::function<void(int)>& f, int reps = 9) {
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  f(0);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    cudaEventRecord(a, g_stream);
    f(i + 1);
    cudaEventRecord(b, g_stream);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    t.push_back(ms);
  }
  INGINE_TEST_CUDA(cudaGetLastError());
  std::sort(t.begin(), t.end());
  cudaEventDestroy(a), cudaEventDestroy(b);
  return t[t.size() / 2];
}

pk::MoeExperts experts(const Layer& L) {
  const Experts& e = L.experts;
  return pk::MoeExperts{e.fmt == ExpertFormat::NVFP4 ? 0 : 1, e.w13.data, e.w13_scale.data,
                        static_cast<const float*>(e.w13_factor.data), e.w2.data, e.w2_scale.data,
                        static_cast<const float*>(e.w2_factor.data)};
}

}  // namespace

int main() {
  const char* on = std::getenv("INGINE_BENCH");
  if (!on || std::string(on) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the batched MoE benchmark\n");
    return kSkip;
  }
  if (skip_without_gpu()) return kSkip;
  const std::string model_dir = INGINE_MODEL_DIR;
  if (!std::ifstream(model_dir + "/config.json")) {
    std::printf("skip: no model at %s\n", model_dir.c_str());
    return kSkip;
  }
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  const cudaStream_t s = g_stream;
  std::vector<int> Ts = {64, 128, 256, 512};
  if (const char* ts = std::getenv("INGINE_BENCH_T")) Ts = {std::atoi(ts)};
  const int maxT = *std::max_element(Ts.begin(), Ts.end());

  std::vector<DeviceBuffer> storage;
  std::vector<Layer> nv, mx;
  for (int l : {1, 2, 3, 5}) nv.push_back(load_layer(model_dir, l, true, s, storage));
  for (int l : {20, 21}) mx.push_back(load_layer(model_dir, l, true, s, storage));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  pk::Gemm gemm;

  // MLP input rows per layer: from INGINE_BENCH_X (a directory of xm_v<visit>.bin files, bf16 [N, 2304] rows of
  // the layer's first visit, e.g. dumped from a batch run), repeated up to maxT; else random rows
  const char* xdir = std::getenv("INGINE_BENCH_X");
  std::printf("routing on %s\n", xdir ? xdir : "random rows");
  std::map<int, uint16_t*> xs;
  for (const auto* set : {&nv, &mx})
    for (const Layer& L : *set) {
      auto* x = dalloc<uint16_t>(size_t(maxT) * H);
      if (xdir) {
        int v = 0;
        while (VISITS[v].layer != L.idx) ++v;
        const std::string path = std::string(xdir) + "/xm_v" + std::to_string(v) + ".bin";
        std::ifstream f(path, std::ios::binary);
        std::vector<uint16_t> rows(size_t(maxT) * H);
        f.read(reinterpret_cast<char*>(rows.data()), std::streamsize(rows.size() * 2));
        const size_t got = size_t(f.gcount()) / (size_t(H) * 2);
        if (got == 0) {
          std::printf("FAIL: no rows in %s\n", path.c_str());
          return 1;
        }
        for (size_t r = got; r < size_t(maxT); ++r) std::copy_n(rows.begin() + (r % got) * H, H, rows.begin() + r * H);
        INGINE_TEST_CUDA(cudaMemcpy(x, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
      } else {
        fill_bf16<<<1024, 256, 0, s>>>(x, size_t(maxT) * H, 1 + L.idx, 3.4f);
      }
      xs[L.idx] = x;
    }
  auto* logits = dalloc<float>(size_t(maxT) * E);
  Bufs b(maxT);
  auto do_route = [&](const Layer& L, int T) {
    gemm.run(bfp(xs[L.idx]), H, bfp(static_cast<const uint16_t*>(L.router_bf16.data)), H, logits, E, true, T, E, H, s);
    route(b, logits, static_cast<const float*>(L.router_bias.data), T, s);
  };

  constexpr double kPeakBw = 1.792e12, kPeakMma = 2.3e14;  // RTX 5090: GDDR7 and measured mma.sync BF16 / FP32
  bool ok = true;
  for (auto* set : {&nv, &mx}) {
    const bool nvfp4 = set == &nv;
    // checkpoint bytes per expert: codes + block scales
    const double w13_bytes = nvfp4 ? 2304.0 * 1152 * (1 + 1.0 / 8) : 2304.0 * 2304 * (1 + 1.0 / 32);
    const double w2_bytes = w13_bytes / 2;
    for (int T : Ts) {
      const size_t nl = set->size();
      // touched experts, averaged over the rotation's layers
      double hit = 0;
      int most = 0;
      for (const Layer& L : *set) {
        do_route(L, T);
        INGINE_TEST_CUDA(cudaStreamSynchronize(s));
        std::vector<int> counts(E);
        INGINE_TEST_CUDA(cudaMemcpy(counts.data(), b.wo.counts, E * 4, cudaMemcpyDeviceToHost));
        hit += double(std::count_if(counts.begin(), counts.end(), [](int c) { return c > 0; })) / double(nl);
        most = std::max(most, *std::max_element(counts.begin(), counts.end()));
      }
      const float t_route = time_ms([&](int r) { do_route((*set)[r % nl], T); });
      auto timed = [&](auto&& body) { return time_ms([&](int r) { do_route((*set)[r % nl], T), body((*set)[r % nl]); }) - t_route; };
      const float o_up = timed([&](const Layer& L) { pk::moe_gate_up(experts(L), bfp(xs[L.idx]), T, b.wo, bfp(b.act_o), s); });
      const float o_all = timed([&](const Layer& L) {
        pk::moe_gate_up(experts(L), bfp(xs[L.idx]), T, b.wo, bfp(b.act_o), s);
        pk::moe_down(experts(L), bfp(b.act_o), T, b.wo, bfp(b.down_o), s);
      });
      const float n_up = timed([&](const Layer& L) { kb::moe_gate_up(experts(L), bfp(xs[L.idx]), T, b.wn, bfp(b.act_n), s); });
      const float n_all = timed([&](const Layer& L) {
        kb::moe_gate_up(experts(L), bfp(xs[L.idx]), T, b.wn, bfp(b.act_n), s);
        kb::moe_down(experts(L), bfp(b.act_n), T, b.wn, bfp(b.down_n), s);
      });
      const double bytes = hit * (w13_bytes + w2_bytes);
      const double flop = 2.0 * T * 8 * (2304.0 * 2304 + 2304.0 * 1152);
      const double bound_ms = std::max(bytes / kPeakBw, flop / kPeakMma) * 1e3;
      auto line = [&](const char* who, float up, float all) {
        std::printf("  %s: gate_up %.3f ms + down %.3f ms = %.3f ms | %6.0f GB/s %6.1f TFLOP/s | %3.0f%% of bound\n", who,
                    up, all - up, all, bytes / all / 1e6, flop / all / 1e9, 100.0 * bound_ms / all);
      };
      std::printf("moe %s T=%3d: %.1f experts (max %d pairs), %.0f MB, bound %.3f ms (%s)\n", nvfp4 ? "nvfp4" : "mxfp8",
                  T, hit, most, bytes / 1e6, bound_ms, bytes / kPeakBw > flop / kPeakMma ? "memory" : "MMA");
      line("old", o_up, o_all);
      line("new", n_up, n_all);
    }
    // identical inputs (largest T): new vs old outputs
    std::printf("-- %s outputs, T=%d\n", nvfp4 ? "nvfp4" : "mxfp8", maxT);
    const Layer& L = (*set)[0];
    do_route(L, maxT);
    ok &= compare(b, experts(L), xs[L.idx], maxT, s);
  }
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
