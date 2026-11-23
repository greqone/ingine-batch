// Shared harness of the batched MoE tests: the batched expert GEMMs (kernels/batch/moe.h) against the prefill
// ones (kernels/prefill: moe_route / moe_gate_up / moe_down) on identical inputs, compared in BF16 ulps.
#pragma once

#include "common/test_util.h"
#include "kernels/batch/moe.h"
#include "kernels/prefill/prefill.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace ingine::test::batch_moe {

namespace pk = ingine::kernels::prefill;
namespace kb = ingine::kernels::batch;

constexpr int H = 2304, I = 1152, E = 128, TOPK = 8;

template <class T>
T* dalloc(size_t n) {
  T* p = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&p, n * sizeof(T) + 256));
  INGINE_TEST_CUDA(cudaMemset(p, 0, n * sizeof(T) + 256));
  return p;
}

inline float bf(uint16_t u) {
  const uint32_t w = uint32_t(u) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

// Differences of `got` from `ref` (bf16 bits). Allowed per value: `ulps` BF16 ulps of the larger magnitude (rounding
// flips) plus 2^floor_log2 rms(ref) (summation-order noise of a cancelling sum, flips of upstream values; a wrong
// product or a lost term is O(rms / 8) or more). Reports the share of differing values, those beyond one ulp, the
// largest |diff| and the worst |diff| / allowance.
struct Diff {
  size_t n = 0, differ = 0, over1 = 0;
  double max_abs = 0, rms = 0, worst = 0;
  Diff(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref, double ulps, int floor_log2)
      : n(ref.size()) {
    for (uint16_t r : ref) rms += double(bf(r)) * bf(r);
    rms = std::sqrt(rms / double(n ? n : 1));
    for (size_t i = 0; i < n; ++i) {
      if (got[i] == ref[i]) continue;
      ++differ;
      const double g = bf(got[i]), r = bf(ref[i]);
      const double d = std::fabs(g - r), m = std::max(std::fabs(g), std::fabs(r));
      const double ulp = m > 0 ? std::ldexp(1.0, std::ilogb(m) - 7) : 0;
      over1 += d > ulp;
      max_abs = std::max(max_abs, d);
      worst = std::max(worst, d / (ulps * ulp + std::ldexp(rms, floor_log2)));
      if (!(d == d)) worst = INFINITY;
    }
  }
  bool print(const char* what, bool check = true) const {
    const bool ok = !check || worst <= 1;
    std::printf("%s %s: %zu values, %.4f%% differ, %.5f%% by > 1 ulp, max |diff| %.3e (rms %.3e), worst diff / "
                "allowance %.3f%s\n",
                ok ? "PASS" : "FAIL", what, n, 100.0 * double(differ) / double(n ? n : 1),
                100.0 * double(over1) / double(n ? n : 1), max_abs, rms, worst, check ? "" : " (not checked)");
    return ok;
  }
};

// Work buffers for chunks of up to max_T tokens (prefill selection and tiles; batch selection and tiles).
struct Bufs {
  int max_T;
  float *wts_o, *wts_n;
  int *idx_o, *idx_n;
  pk::MoeWork wo, wn;  // prefill routing, batch routing
  uint16_t *act_o, *act_n, *act_p, *down_o, *down_n, *out_o, *out_n;
  explicit Bufs(int maxT) : max_T(maxT) {
    const size_t P = size_t(maxT) * TOPK;
    wts_o = dalloc<float>(P), wts_n = dalloc<float>(P);
    idx_o = dalloc<int>(P), idx_n = dalloc<int>(P);
    wo = {dalloc<int>(E), dalloc<int>(1), dalloc<int4>(pk::moe_tiles_capacity(maxT)), dalloc<int>(P), dalloc<int>(P)};
    wn = {dalloc<int>(E), dalloc<int>(1), dalloc<int4>(kb::moe_tiles_capacity(maxT)), dalloc<int>(P), dalloc<int>(P)};
    act_o = dalloc<uint16_t>(P * I), act_n = dalloc<uint16_t>(P * I), act_p = dalloc<uint16_t>(P * I);
    down_o = dalloc<uint16_t>(P * H), down_n = dalloc<uint16_t>(P * H);
    out_o = dalloc<uint16_t>(size_t(maxT) * H), out_n = dalloc<uint16_t>(size_t(maxT) * H);
  }
};

inline const __nv_bfloat16* bfp(const uint16_t* p) { return reinterpret_cast<const __nv_bfloat16*>(p); }
inline __nv_bfloat16* bfp(uint16_t* p) { return reinterpret_cast<__nv_bfloat16*>(p); }

// both selections and routings from FP32 logits [T, 128]: prefill::moe_topk + moe_route, batch moe_select
inline void route(Bufs& b, const float* logits, const float* bias, int T, cudaStream_t s) {
  INGINE_TEST_CUDA(cudaMemsetAsync(b.wo.counts, 0, E * sizeof(int), s));
  pk::moe_topk(logits, bias, T, b.idx_o, b.wts_o, b.wo.counts, s);
  pk::moe_route(b.idx_o, T, b.wo, s);
  kb::moe_select(logits, bias, T, b.idx_n, b.wts_n, b.wn, s);
}

// moe_select against prefill::moe_topk / moe_route: identical idx, wts and counts; tiles of at most
// moe_tile_rows(T) pairs of one expert covering every pair once.
inline bool check_select(Bufs& b, int T) {
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const size_t P = size_t(T) * TOPK;
  auto get = [&](const auto* p, size_t n) {
    std::vector<std::remove_const_t<std::remove_pointer_t<decltype(p)>>> h(n);
    INGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * sizeof(h[0]), cudaMemcpyDeviceToHost));
    return h;
  };
  size_t bad = 0;
  const auto io = get(b.idx_o, P), in = get(b.idx_n, P);
  const auto wo = get(b.wts_o, P), wn = get(b.wts_n, P);
  const auto co = get(b.wo.counts, E), cn = get(b.wn.counts, E);
  for (size_t i = 0; i < P; ++i) bad += io[i] != in[i] || std::memcmp(&wo[i], &wn[i], 4) != 0;
  for (int e = 0; e < E; ++e) bad += co[e] != cn[e];
  const int nt = get(b.wn.tiles_n, 1)[0], TR = kb::moe_tile_rows(T);
  const auto tl = get(b.wn.tiles, size_t(nt));
  const auto ptok = get(b.wn.pair_token, P), pdst = get(b.wn.pair_dst, P);
  std::vector<int> seen(P, 0);
  for (int i = 0; i < nt; ++i) {
    const int4 t = tl[i];
    bad += t.z <= 0 || t.z > TR;
    for (int r = 0; r < t.z; ++r) {
      const int d = pdst[t.y + r];
      bad += d / TOPK != ptok[t.y + r] || in[d] != t.x;
      ++seen[d];
    }
  }
  for (int v : seen) bad += v != 1;
  std::printf("%s   select: %d tiles, %zu mismatches (idx, wts, counts vs prefill::moe_topk; tile cover)\n",
              bad ? "FAIL" : "PASS", nt, bad);
  return bad == 0;
}

// Old and new chains on the routing of route(): act and down of the new kernels, the new moe_down alone on the
// old act rows (reordered to the new pair order), and the combined outputs.
inline bool compare(Bufs& b, const pk::MoeExperts& ex, const uint16_t* x, int T, cudaStream_t s) {
  const size_t P = size_t(T) * TOPK;
  pk::moe_gate_up(ex, bfp(x), T, b.wo, bfp(b.act_o), s);
  pk::moe_down(ex, bfp(b.act_o), T, b.wo, bfp(b.down_o), s);
  pk::moe_combine(bfp(b.down_o), b.wts_o, T, bfp(b.out_o), s);
  kb::moe_gate_up(ex, bfp(x), T, b.wn, bfp(b.act_n), s);
  kb::moe_down(ex, bfp(b.act_n), T, b.wn, bfp(b.down_n), s);
  pk::moe_combine(bfp(b.down_n), b.wts_n, T, bfp(b.out_n), s);
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  INGINE_TEST_CUDA(cudaGetLastError());

  auto get = [&](const auto* p, size_t n) {
    std::vector<std::remove_const_t<std::remove_pointer_t<decltype(p)>>> h(n);
    INGINE_TEST_CUDA(cudaMemcpy(h.data(), p, n * sizeof(h[0]), cudaMemcpyDeviceToHost));
    return h;
  };
  const auto dst_o = get(b.wo.pair_dst, P), dst_n = get(b.wn.pair_dst, P);
  const auto act_o = get(b.act_o, P * I), act_n = get(b.act_n, P * I);
  std::vector<size_t> pos_o(P), pos_n(P);
  for (size_t p = 0; p < P; ++p) pos_o[dst_o[p]] = p, pos_n[dst_n[p]] = p;
  // act rows by (token, slot); the old rows also in the new pair order (input of the isolated new moe_down)
  std::vector<uint16_t> a_o(P * I), a_n(P * I), act_p(P * I);
  for (size_t d = 0; d < P; ++d)
    for (int i = 0; i < I; ++i) {
      a_o[d * I + i] = act_o[pos_o[d] * I + i], a_n[d * I + i] = act_n[pos_n[d] * I + i];
      act_p[pos_n[d] * I + i] = act_o[pos_o[d] * I + i];
    }
  INGINE_TEST_CUDA(cudaMemcpy(b.act_p, act_p.data(), P * I * 2, cudaMemcpyHostToDevice));
  const auto down_chain = get(b.down_n, P * H);
  kb::moe_down(ex, bfp(b.act_p), T, b.wn, bfp(b.down_n), s);
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  INGINE_TEST_CUDA(cudaGetLastError());
  const auto down_o = get(b.down_o, P * H), down_n = get(b.down_n, P * H);
  const auto out_o = get(b.out_o, size_t(T) * H), out_n = get(b.out_n, size_t(T) * H);
  bool ok = true;
  // same inputs: one FP32 sum per value, a BF16 flip at most (act: g and u flips, and cancelled g / u many ulps
  // off, through the activation)
  ok &= Diff(a_n, a_o, 4, -10).print("  gate_up act     ");
  ok &= Diff(down_n, down_o, 1, -12).print("  down (same act) ");
  // through the act differences (~0.04% of act values)
  ok &= Diff(down_chain, down_o, 2, -6).print("  down (chained)  ");
  ok &= Diff(out_n, out_o, 2, -6).print("  combined        ");
  return ok;
}

}  // namespace ingine::test::batch_moe
