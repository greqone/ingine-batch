// ingine-batch-bench: many requests at once on one GPU. By default the round runs through model::BatchEngine (what
// `ingine-serve --batch` serves): every prompt is admitted into its own slot, then all sequences decode together
// until each reaches the end of its turn or --tokens; --spec K verifies K DSpark drafts per sequence and step.
// Reports the round's aggregate rate (every output token / wall time from the first prefill to the last token: the
// "concurrent aggregate" of Strata's benchmark protocol), the decode-only rate and the step time.
//
//     ingine-batch-bench --seqs 64 --tokens 512 [--spec 3]
//     ingine-batch-bench --seqs 8 --tokens 128 --reference --out ref.json   (the same prompts one at a time through
//         the single-request Generator: compare its greedy outputs with a batch run's --out)
// Test modes run their own loop over the BatchDecoder: --forced FILE (teacher-forced agreement), --spec K with
// --oracle FILE (another run's tokens as drafts) or --accept-all (timing of K + 1 rows per sequence).
#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/batch/batch.h"
#include "model/batch.h"
#include "model/batch_engine.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::string default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR")) return env;
  return std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = default_model_dir(), ablation, effort = "none", out_file, forced_file, oracle_file, spec_trace;
  // short prompts of the kinds people send (code, explanations, writing, math); repeated when --seqs exceeds them
  std::string prompts_file = std::string(INGINE_REPO_DIR) + "/bench/prompts-short.json";
  int seqs = 64, slots = 0, ctx = 768, tokens = 512, prefill_chunk = 512, spec = 0;
  double spec_threshold = 0.0;
  int spec_min = 0;
  double temperature = 0.0;
  int seed = 1;
  bool reference = false, ignore_eos = false, kv_fp8 = false, accept_all = false;

  ingine::serve::ArgParser ap("ingine-batch-bench", "Many requests at once: aggregate decode throughput on one GPU.");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--ablation", "", "ablation file (empty: the original model)", &ablation);
  ap.option("--seqs", "", "requests in the round", &seqs);
  ap.option("--slots", "", "request slots (default: --seqs)", &slots);
  ap.option("--ctx", "", "tokens per slot, prompt and output", &ctx);
  ap.option("--tokens", "", "output tokens per request at most", &tokens);
  ap.option("--prefill-chunk", "", "", &prefill_chunk);
  ap.option("--effort", "", "reasoning effort of the chat template", &effort);
  ap.option("--temperature", "", "0 = greedy", &temperature);
  ap.option("--seed", "", "", &seed);
  ap.option("--prompts", "", "JSON array of prompt strings (default: bench/prompts-short.json)", &prompts_file);
  ap.option("--out", "", "write every request's output tokens and text to FILE (JSON)", &out_file);
  ap.flag("--ignore-eos", "always generate --tokens tokens", &ignore_eos);
  ap.flag("--kv-fp8", "FP8 full-attention caches", &kv_fp8);
  ap.flag("--reference", "the same prompts one at a time through the single-request Generator", &reference);
  ap.option("--forced", "",
            "feed the output tokens of FILE (another run's --out) and report where this run's argmax differs from them",
            &forced_file);
  ap.option("--spec", "", "speculative decoding: verify K DSpark drafts per sequence and step (0 = off)", &spec);
  ap.option("--spec-threshold", "",
            "with --spec: per sequence, the drafts whose estimated acceptance exceeds this x running/slots (at most K; "
            "0 = always K)",
            &spec_threshold);
  ap.option("--spec-min", "", "with --spec-threshold: at least this many drafts per sequence", &spec_min);
  ap.option("--oracle", "", "with --spec: the drafts are the output tokens of FILE (another run's --out)", &oracle_file);
  ap.option("--spec-trace", "", "with --spec: write the drafts verified, accepted and their confidences per step to FILE",
            &spec_trace);
  ap.flag("--accept-all", "with --spec, timing only: keep every verified row (the drafts are copies of the anchor)",
          &accept_all);
  ap.parse(argc, argv);
  if (slots == 0) slots = seqs;
  if (spec < 0 || spec > 7 || (spec > 0 && (!forced_file.empty() || reference)) ||
      (!oracle_file.empty() && accept_all) || ((accept_all || !oracle_file.empty()) && spec == 0)) {
    std::fprintf(stderr, "--spec K: K in 0..7, not with --forced or --reference; --oracle or --accept-all need it\n");
    return 2;
  }

  try {
    namespace m = ingine::model;
    std::vector<std::string> prompts;
    {
      std::ifstream f(prompts_file);
      if (!f) throw std::runtime_error("cannot open " + prompts_file);
      prompts = nlohmann::json::parse(f).get<std::vector<std::string>>();
      if (prompts.empty()) throw std::runtime_error(prompts_file + ": no prompts");
    }
    const auto fmt = ingine::text::ChatFormat::load(model_dir);
    const auto& tok = fmt.tokenizer();
    const std::int32_t end_id = *tok.token_id("<|content_model_end_sampling|>");
    std::vector<std::vector<std::int32_t>> ids(seqs);
    for (int b = 0; b < seqs; ++b) {
      ingine::text::ChatOptions chat;
      chat.reasoning_effort = effort;
      const nlohmann::ordered_json messages =
          nlohmann::ordered_json::array({{{"role", "user"}, {"content", prompts[std::size_t(b) % prompts.size()]}}});
      ids[b] = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
      if (int(ids[b].size()) + tokens + spec > ctx) {
        std::fprintf(stderr, "prompt %d: %zu tokens + %d output tokens + %d drafts exceed --ctx %d\n", b, ids[b].size(),
                     tokens, spec, ctx);
        return 2;
      }
    }
    std::vector<std::vector<std::int32_t>> out(seqs);
    nlohmann::json report;
    std::vector<std::vector<std::int32_t>> forced;
    if (!forced_file.empty()) {
      std::ifstream f(forced_file);
      const auto j = nlohmann::json::parse(f);
      for (int b = 0; b < seqs; ++b) {
        const auto& r = j.at("requests").at(b);
        if (r.at("prompt").get<std::string>() != prompts[std::size_t(b) % prompts.size()])
          throw std::runtime_error("--forced: request " + std::to_string(b) + " has another prompt");
        forced.push_back(r.at("tokens").get<std::vector<std::int32_t>>());
        if (forced.back().empty()) throw std::runtime_error("--forced: request " + std::to_string(b) + " is empty");
      }
    }
    std::vector<std::vector<std::int32_t>> oracle;
    if (!oracle_file.empty()) {
      std::ifstream f(oracle_file);
      const auto j = nlohmann::json::parse(f);
      for (int b = 0; b < seqs; ++b) {
        const auto& r = j.at("requests").at(b);
        if (r.at("prompt").get<std::string>() != prompts[std::size_t(b) % prompts.size()])
          throw std::runtime_error("--oracle: request " + std::to_string(b) + " has another prompt");
        oracle.push_back(r.at("tokens").get<std::vector<std::int32_t>>());
      }
    }
    // forced: (request, output index, margin) of every position where this run's argmax is not the forced token;
    // margin = model.logits[argmax] - model.logits[forced token] by this run's logits
    struct Miss {
      int request, index;
      float margin;
    };
    std::vector<Miss> misses;
    std::size_t compared = 0;
    auto bf16r = [](float x) {
      std::uint32_t u;
      std::memcpy(&u, &x, 4);
      u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
      std::memcpy(&x, &u, 4);
      return x;
    };
    auto score = [&](const float* row, std::int32_t t, bool raw) {
      float l = 0;
      CUDA_CHECK(cudaMemcpy(&l, row + t, 4, cudaMemcpyDeviceToHost));
      return raw ? 30.f * std::tanh(bf16r(l) / 30.f) : l;
    };

    cudaFree(nullptr);
    std::size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    const auto t_load = Clock::now();

    if (reference) {
      m::RuntimeOptions o;
      o.model_dir = model_dir;
      o.ablation = ablation;
      o.max_ctx = ctx;
      o.dense_fp8 = false;
      o.kv_fp8 = kv_fp8;
      o.spec = false;
      o.prefill_chunk = prefill_chunk;
      auto gen = m::make_generator(o);
      std::printf("reference: load %.1fs\n", since(t_load));
      const std::vector<std::int32_t> stop = {end_id};
      const auto t0 = Clock::now();
      std::size_t total_out = 0;
      for (int b = 0; b < seqs; ++b) {
        m::Sampling sampling;
        sampling.temperature = float(temperature);
        sampling.seed = std::uint64_t(seed) + std::uint64_t(b);
        gen->generate(ids[b], tokens, sampling, ignore_eos ? std::span<const std::int32_t>() : std::span<const std::int32_t>(stop),
                      [&](std::span<const std::int32_t> t) {
                        out[b].insert(out[b].end(), t.begin(), t.end());
                        return true;
                      });
        total_out += out[b].size();
      }
      const double wall = since(t0);
      std::printf("reference: %d requests one at a time, %zu output tokens in %.2fs = %.1f tok/s\n", seqs, total_out, wall,
                  double(total_out) / wall);
      report["mode"] = "reference";
      report["output_tokens"] = total_out;
      report["wall_s"] = wall;
    } else if (forced.empty() && oracle.empty() && !accept_all) {
      m::BatchEngineOptions options;
      options.model_dir = model_dir;
      options.ablation = ablation;
      options.slots = slots;
      options.ctx = ctx;
      options.prefill_chunk = prefill_chunk;
      options.spec = spec;
      options.kv_fp8 = kv_fp8;
      options.spec_trace = spec_trace;
      options.spec_threshold = float(spec_threshold);
      options.spec_min = spec_min;
      m::BatchEngine engine(options);
      std::size_t free1 = 0;
      cudaMemGetInfo(&free1, &total);
      std::printf("engine: load %.1fs, GPU %.2f GB (slots %d x %d tokens, spec %d: %.2f GB besides the weights), "
                  "%.2f GB free\n", since(t_load), double(free0 - free1) / 1e9, slots, ctx, spec,
                  double(engine.device_bytes()) / 1e9, double(free1) / 1e9);
      std::fflush(stdout);

      const auto t0 = Clock::now();
      std::vector<m::BatchBackend::Admission> admissions;
      for (int b = 0; b < seqs; ++b) admissions.push_back({b, ids[b], float(temperature)});
      const std::vector<std::int32_t> first = engine.admit(admissions);
      const double prefill_s = since(t0);
      std::vector<m::BatchBackend::Seq> running;
      for (int b = 0; b < seqs; ++b) {
        out[b].push_back(first[b]);
        if (tokens > 1 && (ignore_eos || first[b] != end_id))
          running.push_back({b, first[b], std::int64_t(ids[b].size()), float(temperature)});
      }
      std::vector<std::vector<std::int32_t>> emitted;
      std::vector<double> step_s;
      std::vector<int> step_n;
      const auto t_dec = Clock::now();
      while (!running.empty()) {
        const auto ts = Clock::now();
        engine.step(running, emitted);
        step_s.push_back(since(ts));
        step_n.push_back(int(running.size()));
        std::vector<m::BatchBackend::Seq> still;
        for (std::size_t i = 0; i < running.size(); ++i) {
          const int b = running[i].slot;
          bool finished = false;
          for (std::int32_t t : emitted[i]) {
            out[b].push_back(t);
            finished = int(out[b].size()) >= tokens || (!ignore_eos && t == end_id);
            if (finished) break;
          }
          if (!finished) still.push_back(running[i]);
        }
        running.swap(still);
      }
      const double decode_s = since(t_dec), wall = since(t0);
      std::size_t total_out = 0, seq_steps = 0;
      for (int b = 0; b < seqs; ++b) total_out += out[b].size();
      for (int n : step_n) seq_steps += std::size_t(n);
      double full_ms = 0;
      int full = 0;
      for (std::size_t i = 0; i < step_s.size(); ++i)
        if (step_n[i] == seqs) full_ms += 1e3 * step_s[i], ++full;
      std::printf("engine: %d requests, %zu output tokens; admission %.2fs, decode %.2fs in %zu steps (%.1f ms per step "
                  "with all %d)\n", seqs, total_out, prefill_s, decode_s, step_s.size(), full ? full_ms / full : 0.0,
                  seqs);
      if (spec > 0)
        std::printf("spec: up to %d drafts per step (threshold %.2f), %.3f rows, %.3f tokens per sequence and step "
                    "(%.3f drafts accepted)\n", spec, spec_threshold,
                    double(engine.spec_rows()) / double(engine.spec_steps()),
                    double(total_out - std::size_t(seqs)) / double(seq_steps),
                    double(engine.spec_accepted()) / double(engine.spec_steps()));
      std::printf("AGGREGATE %.1f tok/s (%zu output tokens / %.2fs from the first prefill to the last token)\n",
                  double(total_out) / wall, total_out, wall);
      report["mode"] = "engine";
      report["spec"] = spec;
      report["output_tokens"] = total_out;
      report["wall_s"] = wall;
      report["prefill_s"] = prefill_s;
      report["decode_s"] = decode_s;
      report["aggregate_tps"] = double(total_out) / wall;
      report["full_step_ms"] = full ? full_ms / full : 0.0;
      report["step_n"] = step_n;  // running sequences and seconds of every step
      report["step_s"] = step_s;
    } else {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      m::Weights w = m::load_weights(model_dir, /*dense_fp8=*/false, stream);
      m::Ablation abl = m::load_ablation(ablation);
      m::BatchState st = m::make_batch_state(slots, ctx, kv_fp8);
      m::Prefiller pf(w, abl, prefill_chunk, stream);
      m::BatchDecoder dec(w, abl, st, seqs, stream, spec + 1);
      if (!forced_file.empty()) dec.keep_logits(true);  // the misses' margins
      ingine::DeviceBuffer chunk(std::size_t(prefill_chunk) * 4), logits(std::size_t(m::UNPADDED_VOCAB) * 4), first(4);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::size_t free1 = 0;
      cudaMemGetInfo(&free1, &total);
      std::printf("batch: load %.1fs, GPU %.2f GB (slots %d x %d tokens: caches %.2f GB), %.2f GB free\n", since(t_load),
                  double(free0 - free1) / 1e9, slots, ctx, double(st.device_bytes()) / 1e9, double(free1) / 1e9);
      std::fflush(stdout);

      // the round: every request starts now (prefill each, then decode all together)
      const auto t0 = Clock::now();
      std::vector<std::int32_t> last(seqs);
      for (int b = 0; b < seqs; ++b) {
        m::reset_slot(st, b, stream);
        m::State v = st.view(b);
        const int P = int(ids[b].size());
        for (int s = 0; s < P; s += prefill_chunk) {
          const int T = std::min(prefill_chunk, P - s);
          CUDA_CHECK(cudaMemcpyAsync(chunk.p, ids[b].data() + s, std::size_t(T) * 4, cudaMemcpyHostToDevice, stream));
          pf.run(v, static_cast<const std::int32_t*>(chunk.p), T, s, static_cast<float*>(logits.p), nullptr);
        }
        ingine::kernels::batch::sample(static_cast<const float*>(logits.p), m::UNPADDED_VOCAB, 1, m::UNPADDED_VOCAB, false,
                                       float(temperature), std::uint64_t(seed) * 1000003u + std::uint64_t(b),
                                       static_cast<std::int32_t*>(first.p), stream);
        CUDA_CHECK(cudaMemcpyAsync(&last[b], first.p, 4, cudaMemcpyDeviceToHost, stream));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const double prefill_s = since(t0);
      for (int b = 0; b < seqs; ++b) {
        if (forced.empty()) {
          out[b].push_back(last[b]);
          continue;
        }
        ++compared;
        if (last[b] != forced[b][0]) {
          const float* row = static_cast<const float*>(logits.p);  // the last prefill's: request seqs - 1 only
          misses.push_back({b, 0, b == seqs - 1 ? score(row, last[b], false) - score(row, forced[b][0], false) : -1.f});
        }
        out[b].push_back(forced[b][0]);
      }

      std::vector<int> active;
      for (int b = 0; b < seqs; ++b) {
        const bool more = forced.empty() ? tokens > 1 && (ignore_eos || last[b] != end_id) : forced[b].size() > 1;
        if (more) active.push_back(b);
      }
      // --spec K: rows [anchor, K drafts] per sequence; the longest prefix of drafts equal to the target's argmax is
      // accepted, then the target's token after it (greedy speculative decoding). Drafts: --oracle FILE's tokens at
      // those positions, or (--accept-all, timing only) copies of the anchor with every row kept
      const int M = spec + 1;
      auto draft_of = [&](int b, int k) -> std::int32_t {
        const std::size_t at = out[b].size() + std::size_t(k);
        return !oracle.empty() && at < oracle[b].size() ? oracle[b][at] : out[b].back();
      };
      std::vector<std::int32_t> in_tok, in_slot, keep;
      std::vector<std::int64_t> in_pos;
      std::vector<double> step_s;
      std::vector<int> step_n;
      std::size_t drafts_accepted = 0;
      const auto t_dec = Clock::now();
      std::uint64_t step_index = 0;
      while (!active.empty()) {
        in_tok.clear(), in_slot.clear(), in_pos.clear(), keep.clear();
        for (int b : active) {
          in_tok.push_back(out[b].back());
          for (int k = 0; k < spec; ++k) in_tok.push_back(draft_of(b, k));
          in_slot.push_back(b);
          in_pos.push_back(std::int64_t(ids[b].size() + out[b].size() - 1));  // position of the anchor
        }
        const auto ts = Clock::now();
        const auto got = dec.step(int(active.size()), M, in_tok.data(), in_slot.data(), in_pos.data(),
                                  float(temperature), std::uint64_t(seed) * 0x9E3779B97F4A7C15ull + ++step_index);
        std::vector<int> still;
        for (std::size_t i = 0; i < active.size(); ++i) {
          const int b = active[i];
          if (!forced.empty()) {
            const std::int32_t want = forced[b][out[b].size()];
            ++compared;
            if (got[i] != want) {
              const float* row = dec.logits() + i * m::BatchDecoder::kLogitsLd;
              const bool raw = dec.logits_raw();
              misses.push_back({b, int(out[b].size()), score(row, got[i], raw) - score(row, want, raw)});
            }
            out[b].push_back(want);
            if (out[b].size() < forced[b].size()) still.push_back(b);
            continue;
          }
          const std::int32_t* row_tok = in_tok.data() + i * M;
          const std::int32_t* row_got = got.data() + i * M;
          int a = 0;
          if (accept_all) a = spec;
          else
            while (a < spec && row_tok[1 + a] == row_got[a]) ++a;
          drafts_accepted += std::size_t(a);
          keep.push_back(a + 1);
          bool finished = false;
          for (int k = 0; k <= a && !finished; ++k) {  // the accepted drafts, then the target's token after them
            const std::int32_t t = k < a ? row_tok[1 + k] : row_got[a];
            out[b].push_back(t);
            finished = int(out[b].size()) >= tokens || (!ignore_eos && t == end_id);
          }
          if (!finished) still.push_back(b);
        }
        if (M > 1) dec.rollback(keep.data());
        step_s.push_back(since(ts));
        step_n.push_back(int(active.size()));
        active.swap(still);
      }
      const double decode_s = since(t_dec), wall = since(t0);
      std::size_t total_out = 0, seq_steps = 0;
      for (int b = 0; b < seqs; ++b) total_out += out[b].size();
      for (int n : step_n) seq_steps += std::size_t(n);
      const std::size_t decode_out = total_out - std::size_t(seqs);  // the first tokens come from the prefill
      std::vector<double> full_steps;
      for (std::size_t i = 0; i < step_s.size(); ++i)
        if (step_n[i] == seqs) full_steps.push_back(step_s[i]);
      const double full_step_ms =
          full_steps.empty() ? 0 : 1e3 * std::accumulate(full_steps.begin(), full_steps.end(), 0.0) / double(full_steps.size());
      std::printf("batch: %d requests, %zu output tokens (%zu steps, %.1f ms per step with all %d)\n", seqs, total_out,
                  step_s.size(), full_step_ms, seqs);
      std::printf("batch: prefill %.2fs, decode %.2fs (%.1f tok/s)\n", prefill_s, decode_s, double(decode_out) / decode_s);
      if (spec > 0)
        std::printf("spec: %d drafts per step, %.3f tokens per sequence and step (%.3f drafts accepted)\n", spec,
                    double(decode_out) / double(seq_steps), double(drafts_accepted) / double(seq_steps));
      std::printf("AGGREGATE %.1f tok/s (%zu output tokens / %.2fs from the first prefill to the last token)\n",
                  double(total_out) / wall, total_out, wall);
      report["mode"] = "batch";
      report["output_tokens"] = total_out;
      report["wall_s"] = wall;
      report["prefill_s"] = prefill_s;
      report["decode_s"] = decode_s;
      report["aggregate_tps"] = double(total_out) / wall;
      report["full_step_ms"] = full_step_ms;
      if (!forced.empty()) {
        std::printf("forced: %zu positions, %zu where the argmax differs (top-1 agreement %.3f%%)\n", compared,
                    misses.size(), 100.0 * double(compared - misses.size()) / double(compared));
        std::sort(misses.begin(), misses.end(), [](const Miss& a, const Miss& b) { return a.margin > b.margin; });
        for (const Miss& x : misses)
          std::printf("  request %d output %d: margin %.4f\n", x.request, x.index, double(x.margin));
        nlohmann::json mj = nlohmann::json::array();
        for (const Miss& x : misses) mj.push_back({x.request, x.index, x.margin});
        report["forced_compared"] = compared;
        report["forced_misses"] = mj;
      }
      CUDA_CHECK(cudaStreamDestroy(stream));
    }

    if (!out_file.empty()) {
      nlohmann::json reqs = nlohmann::json::array();
      for (int b = 0; b < seqs; ++b) {
        reqs.push_back({{"prompt", prompts[std::size_t(b) % prompts.size()]},
                        {"prompt_tokens", ids[b].size()},
                        {"tokens", out[b]},
                        {"text", tok.decode(out[b])}});
      }
      report["requests"] = reqs;
      report["seqs"] = seqs;
      report["temperature"] = temperature;
      // a response cut at --tokens may end inside a UTF-8 sequence: replace its bytes instead of failing
      std::ofstream(out_file) << report.dump(1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-batch-bench: %s\n", e.what());
    return 1;
  }
}
