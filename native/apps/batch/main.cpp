// ingine-batch-bench: many requests at once on one GPU (model/batch.h). Each prompt is prefilled into its own slot,
// then all sequences decode together, one token each per step, until each reaches the end of its turn or --tokens.
// Reports the round's aggregate rate (every output token / wall time from the first prefill to the last token: the
// "concurrent aggregate" of Strata's benchmark protocol), the decode-only rate and the step time.
//
//     ingine-batch-bench --seqs 64 --tokens 512
//     ingine-batch-bench --seqs 8 --tokens 128 --reference --out ref.json   (the same prompts one at a time through
//         the single-request Generator: compare its greedy outputs with a batch run's --out)
#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/batch/batch.h"
#include "model/batch.h"
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

// short prompts of the kinds people send (code, explanations, writing, math); repeated when --seqs exceeds them
const std::vector<std::string> kPrompts = {
    "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests.",
    "What is the difference between a process and a thread? Answer in a few paragraphs.",
    "Write a SQL query that returns the top 5 customers by total order value, with their email addresses.",
    "Explain the CAP theorem with a concrete example.",
    "Write a bash script that finds the 10 largest files under a directory and prints their sizes.",
    "Summarise the causes of the French Revolution.",
    "Write a JavaScript debounce function and explain when to use it instead of throttle.",
    "How does TCP congestion control work? Explain slow start and congestion avoidance.",
    "Write a Rust function that parses a comma-separated list of integers and returns their sum, with error handling.",
    "Explain gradient descent to a high-school student.",
    "Write a Go HTTP server with one JSON endpoint that returns the current time.",
    "What are the main differences between SQL and NoSQL databases?",
    "Write a C function that reverses a singly linked list in place.",
    "Explain what a closure is in JavaScript, with two examples.",
    "Solve: a train travels 300 km at 75 km/h and returns at 100 km/h. What is its average speed for the trip?",
    "Write a Python class for a bank account with deposit, withdraw and a transaction history.",
    "Explain how public-key cryptography works, using RSA as the example.",
    "Write a regular expression that validates an IPv4 address and explain each part.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between `git merge` and `git rebase`, and when to use each.",
    "Write a TypeScript interface for a blog post and a function that sorts posts by date.",
    "What is the time complexity of quicksort, and why can it degrade to O(n^2)?",
    "Write a Python script that reads a CSV file and prints the average of a numeric column.",
    "Explain how vaccines train the immune system.",
    "Write a binary search in Java and list its edge cases.",
    "Describe the water cycle in detail.",
    "Write a Dockerfile for a small Python Flask app and explain each line.",
    "Explain the difference between supervised and unsupervised learning, with examples.",
    "Write a function that merges two sorted arrays into one sorted array, in C++.",
    "What causes the seasons on Earth?",
    "Write a Python generator that yields the Fibonacci numbers and a test for it.",
    "Explain how a CPU cache works and why cache misses are expensive.",
    "Write an email to a landlord asking to fix a broken heater, polite but firm.",
    "Prove that the square root of 2 is irrational.",
    "Write a React component that shows a counter with increment and reset buttons.",
    "Explain what Kubernetes does and the role of pods, services and deployments.",
    "Write a Python function that flattens an arbitrarily nested list.",
    "Explain the difference between latency and throughput, with examples from networking and GPUs.",
    "Write a haiku sequence of five haikus about autumn in a city.",
    "Write a Python decorator that retries a function up to three times on exception, with exponential backoff.",
    "How does a transformer language model generate text, step by step?",
    "Write unit tests in pytest for a function that validates email addresses.",
    "Explain the Monty Hall problem and why switching is better.",
    "Write a Makefile for a small C project with a test target.",
    "What is the difference between TCP and UDP, and when would you choose each?",
    "Write a Python script that downloads a web page and counts the most common words.",
    "Explain how garbage collection works in Java, including generations.",
    "Write a cover letter for a junior data analyst position.",
    "Implement a stack using two queues in Python and explain the complexity.",
    "Explain how a bloom filter works and where it is used.",
    "Write a Lua function that deep-copies a table, handling cycles.",
    "Explain what inflation is and how central banks respond to it.",
    "Write a Python asyncio program that fetches three URLs concurrently.",
    "What are the SOLID principles? Give one example for each.",
    "Write a function that computes the edit distance between two strings, with dynamic programming.",
    "Explain how photosynthesis works.",
    "Write a shell one-liner that counts unique IP addresses in an nginx access log, and explain it.",
    "Explain the difference between a mutex and a semaphore.",
    "Write a Python function that converts Roman numerals to integers, with tests.",
    "Describe how a refrigerator works.",
    "Write a minimal event emitter class in JavaScript with on, off and emit.",
    "Explain what a database index is and how a B-tree index speeds up queries.",
};

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = default_model_dir(), ablation, effort = "none", prompts_file, out_file, forced_file;
  int seqs = 64, slots = 0, ctx = 768, tokens = 512, prefill_chunk = 512;
  double temperature = 0.0;
  int seed = 1;
  bool reference = false, ignore_eos = false, kv_fp8 = false;

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
  ap.option("--prompts", "", "JSON array of prompt strings (default: 64 built-in short prompts)", &prompts_file);
  ap.option("--out", "", "write every request's output tokens and text to FILE (JSON)", &out_file);
  ap.flag("--ignore-eos", "always generate --tokens tokens", &ignore_eos);
  ap.flag("--kv-fp8", "FP8 full-attention caches", &kv_fp8);
  ap.flag("--reference", "the same prompts one at a time through the single-request Generator", &reference);
  ap.option("--forced", "",
            "feed the output tokens of FILE (another run's --out) and report where this run's argmax differs from them",
            &forced_file);
  ap.parse(argc, argv);
  if (slots == 0) slots = seqs;

  try {
    namespace m = ingine::model;
    std::vector<std::string> prompts = kPrompts;
    if (!prompts_file.empty()) {
      std::ifstream f(prompts_file);
      prompts = nlohmann::json::parse(f).get<std::vector<std::string>>();
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
      if (int(ids[b].size()) + tokens > ctx) {
        std::fprintf(stderr, "prompt %d: %zu tokens + %d output tokens exceed --ctx %d\n", b, ids[b].size(), tokens, ctx);
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
    } else {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      m::Weights w = m::load_weights(model_dir, /*dense_fp8=*/false, stream);
      m::Ablation abl = m::load_ablation(ablation);
      m::BatchState st = m::make_batch_state(slots, ctx, kv_fp8);
      m::Prefiller pf(w, abl, prefill_chunk, stream);
      m::BatchDecoder dec(w, abl, st, seqs, stream);
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
      std::vector<std::int32_t> in_tok, in_slot;
      std::vector<std::int64_t> in_pos;
      std::vector<double> step_s;
      std::vector<int> step_n;
      const auto t_dec = Clock::now();
      std::uint64_t step_index = 0;
      while (!active.empty()) {
        in_tok.clear(), in_slot.clear(), in_pos.clear();
        for (int b : active) {
          in_tok.push_back(out[b].back());
          in_slot.push_back(b);
          in_pos.push_back(std::int64_t(ids[b].size() + out[b].size() - 1));  // position of the token fed in
        }
        const auto ts = Clock::now();
        const auto got = dec.step(int(active.size()), in_tok.data(), in_slot.data(), in_pos.data(), float(temperature),
                                  std::uint64_t(seed) * 0x9E3779B97F4A7C15ull + ++step_index);
        step_s.push_back(since(ts));
        step_n.push_back(int(active.size()));
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
          out[b].push_back(got[i]);
          if (int(out[b].size()) < tokens && (ignore_eos || got[i] != end_id)) still.push_back(b);
        }
        active.swap(still);
      }
      const double decode_s = since(t_dec), wall = since(t0);
      std::size_t total_out = 0, decode_out = 0;
      for (int b = 0; b < seqs; ++b) total_out += out[b].size();
      for (int n : step_n) decode_out += std::size_t(n);
      std::vector<double> full_steps;
      for (std::size_t i = 0; i < step_s.size(); ++i)
        if (step_n[i] == seqs) full_steps.push_back(step_s[i]);
      const double full_step_ms =
          full_steps.empty() ? 0 : 1e3 * std::accumulate(full_steps.begin(), full_steps.end(), 0.0) / double(full_steps.size());
      std::printf("batch: %d requests, %zu output tokens (%zu steps, %.1f ms per step with all %d)\n", seqs, total_out,
                  step_s.size(), full_step_ms, seqs);
      std::printf("batch: prefill %.2fs, decode %.2fs (%.1f tok/s)\n", prefill_s, decode_s, double(decode_out) / decode_s);
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
      std::ofstream(out_file) << report.dump(1) << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-batch-bench: %s\n", e.what());
    return 1;
  }
}
