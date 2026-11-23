// BatchDrafter (model/batch_drafter.h) against the single-stream Drafter with the real mtp.safetensors and a small
// synthetic target (embedding/unembedding of 4096 tokens): four sequences of different lengths (one longer than
// the 1024-position local window) in non-adjacent slots, committed in one call, then 1..8 more rows each in one
// call; every draft of every sequence compared with the Drafter's on the same context. At DRAFTER precision the
// drafts agree but for near ties (different accumulation orders); at MXFP8 most do. Also: drafts are
// deterministic, sampled drafts reproduce per seed, and a reset slot drafts as an unused one.
#include "common/test_util.h"
#include "model/batch_drafter.h"
#include "model/drafter.h"

#include "kernels/dense/dense.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

using namespace ingine::test;
using ingine::DeviceBuffer;
using ingine::DType;
namespace model = ingine::model;

namespace {

constexpr int H = model::HIDDEN, V = 4096, CTX = 1536, G = model::BatchDrafter::GAMMA;
constexpr std::size_t ROW = std::size_t(4) * H;
int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

struct Out {
  std::vector<int32_t> tokens;
  std::vector<float> conf;
  bool operator==(const Out& o) const { return tokens == o.tokens && conf == o.conf; }
};

template <class D>
Out fetch(const D& d, int n, cudaStream_t s) {
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  Out o{std::vector<int32_t>(std::size_t(n) * G), std::vector<float>(std::size_t(n) * G)};
  INGINE_TEST_CUDA(cudaMemcpy(o.tokens.data(), d.tokens, o.tokens.size() * 4, cudaMemcpyDeviceToHost));
  INGINE_TEST_CUDA(cudaMemcpy(o.conf.data(), d.conf, o.conf.size() * 4, cudaMemcpyDeviceToHost));
  return o;
}

void check_precision(model::BatchDrafter::Precision precision, const char* name, const model::Weights& w,
                     const std::filesystem::path& dir, model::Drafter& single, cudaStream_t stream) {
  const std::vector<int32_t> slots = {0, 3, 5, 6}, anchors = {17, 401, 2900, 1234}, more = {1, 8, 3, 5};
  const std::vector<int> lengths = {300, 517, 64, 1200};
  const int B = int(slots.size());
  // each sequence's taps: random normals, lengths[b] + 8 rows
  Rng r(99);
  std::vector<DeviceBuffer> feats;
  std::vector<__nv_bfloat16> all;  // the first commit's rows, sequence after sequence
  for (int b = 0; b < B; ++b) {
    std::vector<__nv_bfloat16> f(std::size_t(lengths[b] + 8) * ROW);
    for (auto& x : f) x = float_to_bf16(float(r.normal()));
    all.insert(all.end(), f.begin(), f.begin() + std::ptrdiff_t(std::size_t(lengths[b]) * ROW));
    feats.emplace_back(f.size() * 2);
    feats.back().copy_from_host(f.data(), f.size() * 2);
  }
  DeviceBuffer d_all(all.size() * 2);
  d_all.copy_from_host(all.data(), all.size() * 2);

  model::BatchDrafter batch(dir, w, 8, CTX, B, stream, precision);
  long equal = 0, total = 0, first_equal = 0;
  float max_conf = 0.f;
  auto compare = [&](const Out& got, const std::vector<int>& P) {
    for (int b = 0; b < B; ++b) {
      single.reset();
      single.commit(static_cast<const __nv_bfloat16*>(feats[b].p), P[b], 0);
      const Out ref = fetch(single.draft(anchors[b], P[b], 0.f), 1, stream);
      bool prefix = true;
      for (int j = 0; j < G; ++j) {
        const bool eq = ref.tokens[j] == got.tokens[std::size_t(b) * G + j];
        equal += eq;
        if (prefix) max_conf = std::max(max_conf, std::fabs(ref.conf[j] - got.conf[std::size_t(b) * G + j]));
        prefix = prefix && eq;
      }
      first_equal += ref.tokens[0] == got.tokens[std::size_t(b) * G];
      total += G;
    }
  };
  std::vector<std::int64_t> start(B, 0), P(B);
  std::vector<int32_t> count(B);
  for (int b = 0; b < B; ++b) count[b] = lengths[b], P[b] = lengths[b];
  batch.commit(B, slots.data(), start.data(), count.data(), static_cast<const __nv_bfloat16*>(d_all.p));
  const Out first = fetch(batch.draft(B, slots.data(), anchors.data(), P.data(), 0.f, 0), B, stream);
  bool in_range = true;
  for (int i = 0; i < B * G; ++i)
    in_range = in_range && first.tokens[i] >= 0 && first.tokens[i] < V && first.conf[i] > 0.f && first.conf[i] <= 1.f;
  char what[160];
  std::snprintf(what, sizeof what, "%s: tokens in the vocabulary, conf in (0, 1]", name);
  expect(in_range, what);
  std::snprintf(what, sizeof what, "%s: the same draft call twice gives the same drafts", name);
  expect(fetch(batch.draft(B, slots.data(), anchors.data(), P.data(), 0.f, 0), B, stream) == first, what);
  compare(first, std::vector<int>(lengths.begin(), lengths.end()));

  // 1..8 more rows per sequence in one call (one sequence's rows packed after the other's)
  std::vector<int> P2(B);
  {
    DeviceBuffer d_more(std::size_t(8) * B * ROW * 2);
    std::size_t off = 0;
    for (int b = 0; b < B; ++b) {
      INGINE_TEST_CUDA(cudaMemcpy(static_cast<__nv_bfloat16*>(d_more.p) + off,
                                  static_cast<const __nv_bfloat16*>(feats[b].p) + std::size_t(lengths[b]) * ROW,
                                  std::size_t(more[b]) * ROW * 2, cudaMemcpyDeviceToDevice));
      off += std::size_t(more[b]) * ROW;
      start[b] = lengths[b];
      count[b] = more[b];
      P[b] = P2[b] = lengths[b] + more[b];
    }
    batch.commit(B, slots.data(), start.data(), count.data(), static_cast<const __nv_bfloat16*>(d_more.p));
    compare(fetch(batch.draft(B, slots.data(), anchors.data(), P.data(), 0.f, 0), B, stream), P2);
  }
  const double agree = double(equal) / double(total), agree_first = double(first_equal) / double(2 * B);
  std::printf("%s: %ld / %ld drafts equal to the Drafter's (%.1f%%), first drafts %.1f%%, max |conf diff| %.3g\n", name,
              equal, total, 100.0 * agree, 100.0 * agree_first, double(max_conf));
  const bool exact = precision == model::BatchDrafter::Precision::DRAFTER;
  std::snprintf(what, sizeof what, "%s: drafts agree with the Drafter's (>= %s)", name, exact ? "90%" : "50%");
  expect(agree >= (exact ? 0.9 : 0.5) && agree_first >= (exact ? 0.99 : 0.6), what);

  // sampled drafts reproduce per seed
  const Out s1 = fetch(batch.draft(B, slots.data(), anchors.data(), P.data(), 0.8f, 5), B, stream);
  const Out s2 = fetch(batch.draft(B, slots.data(), anchors.data(), P.data(), 0.8f, 5), B, stream);
  const Out s3 = fetch(batch.draft(B, slots.data(), anchors.data(), P.data(), 0.8f, 6), B, stream);
  std::snprintf(what, sizeof what, "%s: sampled drafts reproduce per seed and change with it", name);
  expect(s1 == s2 && s1.tokens != s3.tokens, what);

  // reset_slot: a used slot reset and given sequence 0's rows drafts as an unused slot given the same rows
  const std::int32_t slot0 = slots[0], fresh = 7;
  const std::int64_t at0 = 0, P0 = P2[0];
  batch.reset_slot(slot0);
  batch.commit(1, &slot0, &at0, &P2[0], static_cast<const __nv_bfloat16*>(feats[0].p));
  const Out reused = fetch(batch.draft(1, &slot0, &anchors[0], &P0, 0.f, 0), 1, stream);
  batch.commit(1, &fresh, &at0, &P2[0], static_cast<const __nv_bfloat16*>(feats[0].p));
  std::snprintf(what, sizeof what, "%s: a reset slot drafts as an unused one", name);
  expect(fetch(batch.draft(1, &fresh, &anchors[0], &P0, 0.f, 0), 1, stream) == reused, what);
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: %s/mtp.safetensors not found\n", dir.c_str());
    return kSkip;
  }
  size_t free_b = 0, total_b = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free_b, &total_b));
  if (free_b < (size_t(4) << 30)) {
    std::printf("skip: %.2f GB of GPU memory free, the test needs 4 GB\n", double(free_b) / (1 << 30));
    return kSkip;
  }
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // synthetic target: embedding, its norm, BF16 unembedding of V tokens
  Rng r(5);
  model::Weights w;
  {
    std::vector<__nv_bfloat16> embed(size_t(V) * H), unembed(size_t(V) * H);
    for (auto& x : embed) x = float_to_bf16(float(r.normal()));
    for (auto& x : unembed) x = float_to_bf16(float(r.normal() * 0.05));
    std::vector<float> en(H);
    for (auto& x : en) x = float(r.uniform(0.5, 1.5));
    w.storage.reserve(3);
    auto& e = w.storage.emplace_back(embed.size() * 2);
    e.copy_from_host(embed.data(), embed.size() * 2);
    auto& n = w.storage.emplace_back(en.size() * 4);
    n.copy_from_host(en.data(), en.size() * 4);
    auto& u = w.storage.emplace_back(unembed.size() * 2);
    u.copy_from_host(unembed.data(), unembed.size() * 2);
    w.embed = model::rm(w.storage[0].p, DType::BF16, {V, H});
    w.embed_norm = model::rm(w.storage[1].p, DType::FP32, {H});
    w.unembed.w = model::rm(w.storage[2].p, DType::BF16, {V, H});
  }
  {
    model::Drafter single(dir, w, CTX, stream);
    check_precision(model::BatchDrafter::Precision::DRAFTER, "DRAFTER", w, dir, single, stream);
    check_precision(model::BatchDrafter::Precision::MXFP8, "MXFP8", w, dir, single, stream);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
