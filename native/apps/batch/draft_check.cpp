// ingine-batch-draft: the batched DSpark drafter (model/batch_drafter.h) against the single-stream Drafter on real
// target taps, and its timing.
//
//     ingine-batch-draft --ref ref.json [--ctx 768]
//         ref.json: an ingine-batch-bench --out (prompts and greedy outputs). One Prefiller pass with taps over each
//         prompt + output, then (1) agreement: both drafters get the same committed rows and draft at the same
//         anchors (all sequences in one BatchDrafter call, each at its own position): equal greedy drafts per
//         draft index and the largest |conf| difference; (2) acceptance: a greedy speculative loop along the
//         reference outputs per draft count k = 1..gamma of a block of 1 + gamma rows (draft, accept the matching
//         prefix, commit the accepted rows + 1): mean accepted drafts per round. The BatchDrafter runs at both
//         precisions (DRAFTER: the Drafter's weights and arithmetic; MXFP8: block-scaled FP8 projections; gamma 7,
//         MXFP8 also gamma 6..2), with the calibration of conf at gamma 7 and 4.
//     ingine-batch-draft --bench [--seqs 64 --ctx 640]
//         draft() for gamma 7..1 and commit() of 1..8 rows per sequence at n = --seqs (L2 flushed before each call,
//         as a verify call in between does), and the drafter's device bytes.
#include "core/device.h"
#include "model/batch_drafter.h"
#include "model/drafter.h"
#include "model/batch.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace m = ingine::model;
constexpr int H = m::HIDDEN, G = m::BatchDrafter::GAMMA;
constexpr std::size_t ROW = std::size_t(4) * H;  // tap elements per row

std::string default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR")) return env;
  return std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini";
}

struct Seq {
  std::vector<std::int32_t> ids;  // prompt + reference output
  int prompt = 0;
  const __nv_bfloat16* taps = nullptr;  // device bf16 [ids.size(), 4, H]
};

struct Fetched {
  std::vector<std::int32_t> tokens;
  std::vector<float> conf;
};

template <class D>
Fetched fetch(const D& d, int n, cudaStream_t s) {
  CUDA_CHECK(cudaStreamSynchronize(s));
  Fetched f{std::vector<std::int32_t>(std::size_t(n) * G), std::vector<float>(std::size_t(n) * G)};
  CUDA_CHECK(cudaMemcpy(f.tokens.data(), d.tokens, f.tokens.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(f.conf.data(), d.conf, f.conf.size() * 4, cudaMemcpyDeviceToHost));
  return f;
}

int accepted(const std::int32_t* drafts, const Seq& s, int P, int k) {
  int a = 0;
  while (a < k && P + 1 + a < int(s.ids.size()) && drafts[a] == s.ids[std::size_t(P + 1 + a)]) ++a;
  return a;
}

// (1) agreement at 8 draft points per sequence
void agreement(m::Drafter& single, m::BatchDrafter& batch, const std::vector<Seq>& seqs, cudaStream_t stream) {
  const int B = int(seqs.size());
  constexpr int POINTS = 8;
  std::vector<int> P(B), done(B, 0);
  long equal[G] = {}, cond_equal[G] = {}, cond_total[G] = {}, total = 0;
  float max_conf = 0.f;
  for (int b = 0; b < B; ++b) batch.reset_slot(b);
  for (int k = 0; k < POINTS; ++k) {
    std::vector<std::int32_t> slots, anchors;
    std::vector<std::int64_t> pos;
    // incremental commits up to each sequence's draft point, then one draft call for all
    std::vector<int> who;
    for (int b = 0; b < B; ++b) {
      const Seq& s = seqs[b];
      const int out = int(s.ids.size()) - s.prompt - G - 1;
      if (out < POINTS) continue;
      P[b] = s.prompt + k * out / (POINTS - 1);
      who.push_back(b);
    }
    for (int b : who) {
      const int c = P[b] - done[b];
      const std::int32_t slot = b;
      const std::int64_t start = done[b];
      batch.commit(1, &slot, &start, &c, seqs[b].taps + std::size_t(done[b]) * ROW);
      done[b] = P[b];
      slots.push_back(b);
      anchors.push_back(seqs[b].ids[std::size_t(P[b])]);
      pos.push_back(P[b]);
    }
    const int n = int(who.size());
    const Fetched fb = fetch(batch.draft(n, slots.data(), anchors.data(), pos.data(), 0.f, 0), n, stream);
    for (int i = 0; i < n; ++i) {
      const int b = who[std::size_t(i)];
      single.reset();
      single.commit(seqs[b].taps, P[b], 0);
      const Fetched fs = fetch(single.draft(anchors[std::size_t(i)], P[b], 0.f), 1, stream);
      bool prefix = true;
      for (int j = 0; j < G; ++j) {
        const bool eq = fs.tokens[j] == fb.tokens[std::size_t(i) * G + j];
        equal[j] += eq;
        if (prefix) {
          ++cond_total[j];
          cond_equal[j] += eq;
          max_conf = std::max(max_conf, std::fabs(fs.conf[j] - fb.conf[std::size_t(i) * G + j]));
        }
        prefix = prefix && eq;
      }
      ++total;
    }
  }
  std::printf("  agreement over %ld drafts (%d sequences x %d points):\n   draft index   ", total, B, POINTS);
  for (int j = 0; j < G; ++j) std::printf(" %7d", j + 1);
  std::printf("\n   equal %%       ");
  for (int j = 0; j < G; ++j) std::printf(" %7.2f", 100.0 * double(equal[j]) / double(total));
  std::printf("\n   equal %% | prev");
  for (int j = 0; j < G; ++j) std::printf(" %7.2f", 100.0 * double(cond_equal[j]) / double(std::max(1L, cond_total[j])));
  std::printf("\n   max |conf diff| (rows after equal drafts) %.3g\n", double(max_conf));
}

// (2) greedy speculative loop along the reference outputs; returns mean accepted drafts per round for k = 1..7
std::vector<double> loop_single(m::Drafter& d, const std::vector<Seq>& seqs, cudaStream_t stream) {
  std::vector<double> mean;
  for (int k = 1; k <= G; ++k) {
    long rounds = 0, acc = 0;
    for (const Seq& s : seqs) {
      d.reset();
      d.commit(s.taps, s.prompt, 0);
      for (int P = s.prompt; P + 1 < int(s.ids.size());) {
        const Fetched f = fetch(d.draft(s.ids[std::size_t(P)], P, 0.f), 1, stream);
        const int a = accepted(f.tokens.data(), s, P, k);
        ++rounds, acc += a;
        d.commit(s.taps + std::size_t(P) * ROW, a + 1, P);
        P += a + 1;
      }
    }
    mean.push_back(double(acc) / double(rounds));
  }
  return mean;
}

// Calibration of conf: draft rows whose earlier drafts all matched, by conf bin and by draft index (rows past the
// reference's end left out)
struct Calibration {
  static constexpr int BINS = 10;
  long count[BINS] = {}, hits[BINS] = {}, index_count[G] = {}, index_hits[G] = {};
  double conf_sum[BINS] = {}, index_conf[G] = {};
  void add(int j, float c, bool hit) {
    const int bin = std::min(BINS - 1, int(c * BINS));
    ++count[bin], hits[bin] += hit, conf_sum[bin] += c;
    ++index_count[j], index_hits[j] += hit, index_conf[j] += c;
  }
  void print(int gamma) const {
    std::printf("   conf bin    rows   mean conf  hit rate\n");
    for (int b = 0; b < BINS; ++b)
      if (count[b] > 0)
        std::printf("   %.1f-%.1f %7ld   %8.3f  %8.3f\n", double(b) / BINS, double(b + 1) / BINS, count[b],
                    conf_sum[b] / double(count[b]), double(hits[b]) / double(count[b]));
    std::printf("   draft index  rows   mean conf  hit rate\n");
    for (int j = 0; j < gamma; ++j)
      if (index_count[j] > 0)
        std::printf("   %5d %9ld   %8.3f  %8.3f\n", j + 1, index_count[j], index_conf[j] / double(index_count[j]),
                    double(index_hits[j]) / double(index_count[j]));
  }
};

// mean accepted drafts per round for k = 1..gamma drafts per round (drafts of gamma); calibration of the k = gamma
// loop
std::vector<double> loop_batch(m::BatchDrafter& d, const std::vector<Seq>& seqs, cudaStream_t stream, int gamma,
                               Calibration* calib) {
  const int B = int(seqs.size());
  std::vector<double> mean;
  for (int k = 1; k <= gamma; ++k) {
    long rounds = 0, acc = 0;
    std::vector<int> P(B);
    {
      std::vector<std::int32_t> slots(B), counts(B);
      std::vector<std::int64_t> starts(B, 0);
      for (int b = 0; b < B; ++b) {
        d.reset_slot(b);
        slots[b] = b;
        counts[b] = P[b] = seqs[b].prompt;
      }
      // the prompts, a commit per sequence (their taps are not contiguous)
      for (int b = 0; b < B; ++b) d.commit(1, &slots[b], &starts[b], &counts[b], seqs[b].taps);
    }
    for (;;) {
      std::vector<std::int32_t> slots, anchors;
      std::vector<std::int64_t> pos;
      for (int b = 0; b < B; ++b)
        if (P[b] + 1 < int(seqs[b].ids.size())) {
          slots.push_back(b);
          anchors.push_back(seqs[b].ids[std::size_t(P[b])]);
          pos.push_back(P[b]);
        }
      const int n = int(slots.size());
      if (n == 0) break;
      const Fetched f = fetch(d.draft(n, slots.data(), anchors.data(), pos.data(), 0.f, 0, gamma), n, stream);
      for (int i = 0; i < n; ++i) {
        const int b = slots[std::size_t(i)];
        const int a = accepted(f.tokens.data() + std::size_t(i) * G, seqs[b], P[b], k);
        ++rounds, acc += a;
        if (calib != nullptr && k == gamma)
          for (int j = 0; j <= a && j < k && P[b] + 1 + j < int(seqs[b].ids.size()); ++j)
            calib->add(j, f.conf[std::size_t(i) * G + j], j < a);
        const std::int32_t c = a + 1;
        const std::int64_t start = P[b];
        d.commit(1, &slots[std::size_t(i)], &start, &c, seqs[b].taps + std::size_t(P[b]) * ROW);
        P[b] += a + 1;
      }
    }
    mean.push_back(double(acc) / double(rounds));
  }
  return mean;
}

void print_row(const char* name, const std::vector<double>& v) {
  std::printf("   %-26s", name);
  for (double x : v) std::printf(" %6.3f", x);
  std::printf("\n");
}

int check(const std::string& model_dir, const std::string& ref_file, int ctx) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  std::ifstream rf(ref_file);
  if (!rf) throw std::runtime_error("cannot open " + ref_file);
  const auto ref = nlohmann::json::parse(rf);
  std::vector<Seq> seqs;
  for (const auto& r : ref.at("requests")) {
    Seq s;
    const nlohmann::ordered_json messages =
        nlohmann::ordered_json::array({{{"role", "user"}, {"content", r.at("prompt").get<std::string>()}}});
    ingine::text::ChatOptions chat;
    chat.reasoning_effort = "none";
    s.ids = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
    if (s.ids.size() != r.at("prompt_tokens").get<std::size_t>()) throw std::runtime_error("prompt tokens differ from ref");
    s.prompt = int(s.ids.size());
    for (std::int32_t t : r.at("tokens").get<std::vector<std::int32_t>>()) s.ids.push_back(t);
    if (int(s.ids.size()) > ctx) throw std::runtime_error("a sequence exceeds --ctx");
    seqs.push_back(std::move(s));
  }
  const int B = int(seqs.size());

  m::Weights w = m::load_weights(model_dir, /*dense_fp8=*/false, stream);
  // taps of every prompt + output row, one Prefiller pass per sequence
  std::size_t rows = 0;
  for (const Seq& s : seqs) rows += s.ids.size();
  ingine::DeviceBuffer taps(rows * ROW * 2);
  {
    m::Ablation abl = m::load_ablation("");
    const int chunk = 512;
    m::BatchState st = m::make_batch_state(1, ctx, false);
    m::Prefiller pf(w, abl, chunk, stream);
    ingine::DeviceBuffer ids(std::size_t(chunk) * 4), logits(std::size_t(m::UNPADDED_VOCAB) * 4);
    std::size_t off = 0;
    for (Seq& s : seqs) {
      m::reset_slot(st, 0, stream);
      m::State v = st.view(0);
      s.taps = static_cast<const __nv_bfloat16*>(taps.p) + off * ROW;
      for (int p = 0; p < int(s.ids.size()); p += chunk) {
        const int T = std::min(chunk, int(s.ids.size()) - p);
        CUDA_CHECK(cudaMemcpyAsync(ids.p, s.ids.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, stream));
        ingine::Tensor t = m::rm(const_cast<__nv_bfloat16*>(s.taps) + std::size_t(p) * ROW, ingine::DType::BF16,
                                 {T, 4, H});
        pf.run(v, static_cast<const std::int32_t*>(ids.p), T, p, static_cast<float*>(logits.p), &t);
      }
      off += s.ids.size();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  std::printf("%d sequences, %zu tap rows\n", B, rows);

  using Precision = m::BatchDrafter::Precision;
  const std::pair<Precision, const char*> precisions[] = {{Precision::DRAFTER, "BatchDrafter (DRAFTER)"},
                                                          {Precision::MXFP8, "BatchDrafter (MXFP8)"}};
  std::vector<double> single_acc;
  {
    m::Drafter single(model_dir, w, ctx, stream);
    for (const auto& [p, name] : precisions) {
      m::BatchDrafter batch(model_dir, w, B, ctx, B, stream, p);
      std::printf("%s:\n", name);
      agreement(single, batch, seqs, stream);
    }
    single.capture();
    single_acc = loop_single(single, seqs, stream);
  }
  std::printf("acceptance: mean accepted drafts per round, k = 1..gamma drafts per round (blocks of 1 + gamma rows)\n");
  print_row("Drafter (gamma 7)", single_acc);
  Calibration calib[G + 1];
  for (const auto& [p, name] : precisions) {
    m::BatchDrafter batch(model_dir, w, B, ctx, B, stream, p);
    for (int gamma = G; gamma >= (p == Precision::MXFP8 ? 2 : G); --gamma) {
      char label[64];
      std::snprintf(label, sizeof label, "%s gamma %d", name + 13, gamma);  // name without "BatchDrafter "
      print_row(label, loop_batch(batch, seqs, stream, gamma, p == Precision::MXFP8 ? &calib[gamma] : nullptr));
    }
  }
  for (const int gamma : {G, 4}) {
    std::printf("conf calibration, MXFP8 gamma %d (k = %d: rows whose earlier drafts matched)\n", gamma, gamma);
    calib[gamma].print(gamma);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int bench(const std::string& model_dir, int seqs, int ctx, m::BatchDrafter::Precision precision) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  m::Weights w = m::load_weights(model_dir, /*dense_fp8=*/false, stream);
  std::size_t free0 = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  m::BatchDrafter d(model_dir, w, seqs, ctx, seqs, stream, precision);
  std::size_t free1 = 0;
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  std::printf("BatchDrafter (%s) slots %d ctx %d max_seqs %d: device_bytes %.3f GB (cudaMemGetInfo delta %.3f GB)\n",
              precision == m::BatchDrafter::Precision::MXFP8 ? "MXFP8" : "DRAFTER", seqs, ctx, seqs,
              double(d.device_bytes()) / 1e9, double(free0 - free1) / 1e9);

  // context: random taps (normal), prompts of ctx / 2 rows per slot
  const int prompt = ctx / 2;
  std::vector<__nv_bfloat16> host(std::size_t(std::max(prompt, seqs * m::BatchDrafter::BLOCK)) * ROW);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  for (auto& x : host) x = __float2bfloat16(nd(rng));
  ingine::DeviceBuffer feats(host.size() * 2);
  feats.copy_from_host(host.data(), host.size() * 2);
  const auto* fp = static_cast<const __nv_bfloat16*>(feats.p);
  for (std::int32_t s = 0; s < seqs; ++s) {
    const std::int64_t start = 0;
    d.reset_slot(s);
    d.commit(1, &s, &start, &prompt, fp);
  }
  const std::size_t flush_bytes = std::size_t(256) << 20;
  ingine::DeviceBuffer junk(flush_bytes);
  constexpr int ITERS = 40;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  auto time = [&](auto&& call) {
    std::vector<float> ms(ITERS);
    for (int i = 0; i < ITERS + 3; ++i) {
      CUDA_CHECK(cudaMemsetAsync(junk.p, i, flush_bytes, stream));
      CUDA_CHECK(cudaEventRecord(e0, stream));
      call(i);
      CUDA_CHECK(cudaEventRecord(e1, stream));
      CUDA_CHECK(cudaEventSynchronize(e1));
      if (i >= 3) CUDA_CHECK(cudaEventElapsedTime(&ms[i - 3], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    return ms[ITERS / 2];
  };
  std::vector<std::int32_t> slots(seqs), anchors(seqs);
  std::vector<std::int64_t> P(seqs);
  for (int s = 0; s < seqs; ++s) slots[s] = s, anchors[s] = 1000 + 37 * s;
  // drafts at positions prompt .. prompt + 63 (the rows past the prompt are as good as empty context)
  for (int gamma = m::BatchDrafter::GAMMA; gamma >= 1; --gamma) {
    const float draft_ms = time([&](int i) {
      for (int s = 0; s < seqs; ++s) P[s] = prompt + (s + i) % 64;
      d.draft(seqs, slots.data(), anchors.data(), P.data(), 0.f, 0, gamma);
    });
    std::printf("draft  n %d gamma %d: %.3f ms (median of %d, GPU time between events)\n", seqs, gamma,
                double(draft_ms), ITERS);
  }
  for (int r = 1; r <= m::BatchDrafter::BLOCK; ++r) {
    std::vector<std::int32_t> count(seqs, r);
    std::vector<std::int64_t> start(seqs);
    for (int s = 0; s < seqs; ++s) start[s] = prompt + 8 * (s % 8);
    const float ms = time([&](int) { d.commit(seqs, slots.data(), start.data(), count.data(), fp); });
    std::printf("commit n %d x %d rows: %.3f ms\n", seqs, r, double(ms));
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = default_model_dir(), ref_file, precision = "mxfp8";
  int seqs = 64, ctx = 0;
  bool bench_mode = false;
  ingine::serve::ArgParser ap("ingine-batch-draft", "The batched DSpark drafter: agreement, acceptance, timing.");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--ref", "", "ingine-batch-bench --out file (prompts and greedy outputs)", &ref_file);
  ap.option("--ctx", "", "positions per slot (default 768 with --ref, 640 with --bench)", &ctx);
  ap.option("--seqs", "", "--bench: sequences (and slots)", &seqs);
  ap.option("--precision", "", "--bench: mxfp8 or drafter", &precision);
  ap.flag("--bench", "time draft() and commit()", &bench_mode);
  ap.parse(argc, argv);
  try {
    if (bench_mode) {
      if (precision != "mxfp8" && precision != "drafter") throw std::invalid_argument("--precision: mxfp8 or drafter");
      return bench(model_dir, seqs, ctx ? ctx : 640,
                   precision == "mxfp8" ? m::BatchDrafter::Precision::MXFP8 : m::BatchDrafter::Precision::DRAFTER);
    }
    if (ref_file.empty()) throw std::invalid_argument("--ref or --bench required");
    return check(model_dir, ref_file, ctx ? ctx : 768);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-batch-draft: %s\n", e.what());
    return 1;
  }
}
