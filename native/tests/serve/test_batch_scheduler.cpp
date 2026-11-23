// BatchScheduler unit tests with a backend that emits several tokens per step (like speculative decoding): jobs
// beyond the slots wait and take freed slots, every job's tokens stop at a stop id, at max_new_tokens and at the room
// of its slot, cancellation and admission failures free the slot, close() fails waiting jobs.
#include "engine/batch_scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ingine::engine::BatchScheduler;
using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::model::BatchBackend;
using ingine::model::Sampling;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

// Request q (its prompt[0]) emits q * 1000 + 0, 1, 2, ...: one token at admission, `per_step` per step. Throws on a
// prompt[0] < 0. Checks the scheduler's bookkeeping: distinct slots, positions after the prompt.
class RoundBackend final : public BatchBackend {
 public:
  RoundBackend(int slots, int ctx, int per_step, std::chrono::milliseconds pause = {})
      : slots_(slots), ctx_(ctx), per_step_(per_step), pause_(pause) {}
  std::string model_id() const override { return "round"; }
  int slots() const override { return slots_; }
  int ctx() const override { return ctx_; }
  int lookahead() const override { return per_step_ - 1; }
  std::vector<int32_t> admit(std::span<const Admission> requests) override {
    std::vector<int32_t> first;
    for (const Admission& a : requests) {
      if (a.prompt[0] < 0) throw std::invalid_argument("bad prompt");
      if (a.slot < 0 || a.slot >= slots_) ++bad;
      state_[a.slot] = {a.prompt[0], 1, int(a.prompt.size())};
      first.push_back(a.prompt[0] * 1000);
    }
    return first;
  }
  void step(std::span<Seq> seqs, std::vector<std::vector<int32_t>>& emitted) override {
    if (pause_.count()) std::this_thread::sleep_for(pause_);
    emitted.assign(seqs.size(), {});
    std::set<int> used;
    for (std::size_t i = 0; i < seqs.size(); ++i) {
      Seq& s = seqs[i];
      State& st = state_.at(s.slot);
      if (!used.insert(s.slot).second || s.pending != st.q * 1000 + st.k - 1 || s.pos != st.prompt + st.k - 1 ||
          s.pos + lookahead() >= ctx_)
        ++bad;
      for (int r = 0; r < per_step_; ++r) emitted[i].push_back(st.q * 1000 + st.k++);
      s.pending = emitted[i].back();
      s.pos += per_step_;
    }
    max_running = std::max(max_running.load(), int(seqs.size()));
  }
  std::atomic<int> bad{0}, max_running{0};

 private:
  struct State {
    int q = 0, k = 0, prompt = 0;
  };
  std::map<int, State> state_;
  int slots_, ctx_, per_step_;
  std::chrono::milliseconds pause_;
};

std::vector<int32_t> drain(Job& job, std::optional<std::string>* failure = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(5));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && failure) *failure = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout");
  }
}

std::vector<int32_t> stream(int q, int n) {
  std::vector<int32_t> v;
  for (int k = 0; k < n; ++k) v.push_back(q * 1000 + k);
  return v;
}

std::shared_ptr<Job> job(int q, int max_new, int prompt_len = 1, Sampling sampling = {}) {
  std::vector<int32_t> prompt(std::size_t(prompt_len), q);
  return std::make_shared<Job>(prompt, max_new, sampling);
}

void test_many_jobs() {  // 10 jobs on 4 slots: every stream whole, cut at max_new_tokens mid-step
  RoundBackend backend(4, 1000, 3);
  BatchScheduler s(backend, {}, 16);
  std::vector<std::shared_ptr<Job>> jobs;
  for (int q = 1; q <= 10; ++q) {
    jobs.push_back(job(q, 7));
    CHECK(s.submit(jobs.back()));
  }
  for (int q = 1; q <= 10; ++q) CHECK(drain(*jobs[q - 1]) == stream(q, 7));
  CHECK(backend.max_running <= 4);
  CHECK(backend.bad == 0);
}

void test_stop_and_room() {
  RoundBackend backend(2, 12, 3);  // lookahead 2: a prompt of 5 tokens leaves room for 12 - 2 - 5 = 5 tokens
  BatchScheduler s(backend, {3004}, 4);
  auto stop = job(3, 100);
  auto room = job(4, 100, 5);
  auto none = job(5, 100, 10);
  CHECK(s.submit(stop) && s.submit(room) && s.submit(none));
  CHECK(drain(*stop) == stream(3, 5));  // ends with the stop id
  CHECK(drain(*room) == stream(4, 5));
  std::optional<std::string> failure;
  CHECK(drain(*none, &failure).empty());
  CHECK(failure && failure->find("does not fit") != std::string::npos);
  CHECK(backend.bad == 0);
}

void test_cancel_and_failures() {  // a withdrawn or failed job frees its slot
  RoundBackend backend(1, 1 << 20, 1, std::chrono::milliseconds(2));
  BatchScheduler s(backend, {}, 4);
  auto long_job = job(1, 1 << 30);
  auto bad = job(-1, 5);
  auto top_p = job(2, 5, 1, Sampling{.temperature = 1.f, .top_p = 0.9f});
  auto next = job(3, 4);
  CHECK(s.submit(long_job) && s.submit(bad) && s.submit(top_p) && s.submit(next));
  while (long_job->take(std::chrono::milliseconds(1)).tokens.empty()) {
  }
  s.withdraw(long_job);
  drain(*long_job);
  std::optional<std::string> f1, f2;
  CHECK(drain(*bad, &f1).empty() && f1 && f1->find("bad prompt") != std::string::npos);
  CHECK(drain(*top_p, &f2).empty() && f2 && f2->find("top_p") != std::string::npos);
  CHECK(drain(*next) == stream(3, 4));
}

void test_close() {  // waiting jobs fail, the running one ends
  RoundBackend backend(1, 1 << 20, 1, std::chrono::milliseconds(2));
  auto s = std::make_unique<BatchScheduler>(backend, std::vector<int32_t>{}, 1);
  auto running = job(1, 1 << 30);
  auto waiting = job(2, 5);
  CHECK(s->submit(running));
  while (s->stats().active == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(s->submit(waiting));
  CHECK(!s->submit(job(3, 5)));  // one may wait
  s->close();
  std::optional<std::string> failure;
  CHECK(drain(*waiting, &failure).empty() && failure == std::optional<std::string>("server shutting down"));
  CHECK(!drain(*running).empty());
  CHECK(!s->submit(job(4, 5)));
}

}  // namespace

int main() {
  test_many_jobs();
  test_stop_and_room();
  test_cancel_and_failures();
  test_close();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("batch scheduler: ok\n");
  return 0;
}
