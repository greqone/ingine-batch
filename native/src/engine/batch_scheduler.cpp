#include "engine/batch_scheduler.h"

#include "engine/log.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

namespace ingine::engine {

BatchScheduler::BatchScheduler(model::BatchBackend& backend, std::vector<int32_t> stop_ids, int max_queue)
    : backend_(backend),
      stop_ids_(stop_ids.begin(), stop_ids.end()),
      max_queue_(max_queue),
      thread_([this] { loop(); }) {}

BatchScheduler::~BatchScheduler() { close(); }

bool BatchScheduler::submit(std::shared_ptr<Job> job) {
  {
    std::lock_guard lock(mutex_);
    if (closed_ || int(waiting_.size()) >= max_queue_) return false;
    waiting_.push_back(std::move(job));
  }
  cv_.notify_all();
  return true;
}

void BatchScheduler::withdraw(const std::shared_ptr<Job>& job) {
  job->cancel();
  std::lock_guard lock(mutex_);
  auto it = std::find(waiting_.begin(), waiting_.end(), job);
  if (it != waiting_.end()) waiting_.erase(it);
}

SchedulerStats BatchScheduler::stats() const {
  std::lock_guard lock(mutex_);
  return {int(waiting_.size()), active_ > 0, max_queue_, active_};
}

void BatchScheduler::close() {
  {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto& job : waiting_) {
      job->cancel();
      job->fail("server shutting down");
      job->finish();
    }
    waiting_.clear();
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

bool BatchScheduler::deliver(Running& r, std::span<const int32_t> tokens) const {
  std::vector<int32_t> out;
  bool ended = false;
  for (int32_t t : tokens) {
    if (r.job->cancelled()) break;
    out.push_back(t);
    if (stop_ids_.contains(t) || ++r.produced >= r.limit) {
      ended = true;
      break;
    }
  }
  r.job->push(out);
  return ended || r.job->cancelled();
}

void BatchScheduler::loop() {
  using Seq = model::BatchBackend::Seq;
  std::vector<int> free_slots;
  for (int s = backend_.slots() - 1; s >= 0; --s) free_slots.push_back(s);
  std::vector<Running> running;  // running[i] generates in seqs[i]
  std::vector<Seq> seqs;
  std::vector<std::vector<int32_t>> emitted;
  auto end = [&](std::size_t i) {  // the job of running[i] has ended: its slot is free
    running[i].job->finish();
    free_slots.push_back(seqs[i].slot);
    std::swap(running[i], running.back()), running.pop_back();
    std::swap(seqs[i], seqs.back()), seqs.pop_back();
  };
  auto fail_all = [&](const std::exception& e) {
    server_log().error("batch engine failed: {}", e.what());
    while (!running.empty()) {
      running.back().job->fail(std::string("generation failed: ") + e.what());
      end(running.size() - 1);
    }
  };

  for (;;) {
    std::vector<std::shared_ptr<Job>> admitted;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return closed_ || !running.empty() || (!waiting_.empty() && !free_slots.empty()); });
      if (closed_) break;
      while (!waiting_.empty() && admitted.size() < free_slots.size()) {
        admitted.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
      }
      active_ = int(running.size() + admitted.size());
    }

    // ---- admissions: prefill and first token of every admitted job
    std::vector<model::BatchBackend::Admission> admissions;
    const std::size_t first_new = running.size();
    for (auto& job : admitted) {
      if (job->cancelled()) {
        job->finish();
        continue;
      }
      job->started();
      const int P = int(job->prompt_ids.size());
      const int limit = std::min(job->max_new_tokens, backend_.ctx() - backend_.lookahead() - P);
      std::string refused;
      if (P < 1 || limit < 1)
        refused = "a prompt of " + std::to_string(P) + " tokens does not fit a slot of " +
                  std::to_string(backend_.ctx()) + " positions";
      else if (job->sampling.top_p < 1.f || job->sampling.top_k > 0)
        refused = "batched serving samples by temperature only: top_p and top_k are not supported";
      else if (!job->images.empty())
        refused = "batched serving takes text only";
      if (!refused.empty()) {
        job->fail(refused);
        job->finish();
        continue;
      }
      const int slot = free_slots.back();
      free_slots.pop_back();
      admissions.push_back({slot, job->prompt_ids, job->sampling.temperature});
      running.push_back({job, 0, limit});
      seqs.push_back({slot, 0, P, job->sampling.temperature});
    }
    if (!admissions.empty()) {
      try {
        const std::vector<int32_t> first = backend_.admit(admissions);
        for (std::size_t i = running.size(); i-- > first_new;) {
          seqs[i].pending = first[i - first_new];
          if (deliver(running[i], {&first[i - first_new], 1})) end(i);
        }
      } catch (const std::exception& e) {
        server_log().error("batch admission failed: {}", e.what());
        while (running.size() > first_new) {
          running.back().job->fail(std::string("generation failed: ") + e.what());
          end(running.size() - 1);
        }
      }
    }

    // ---- one step over every running job (cancelled ones end first)
    for (std::size_t i = running.size(); i-- > 0;)
      if (running[i].job->cancelled()) end(i);
    if (running.empty()) continue;
    try {
      backend_.step(seqs, emitted);
      for (std::size_t i = running.size(); i-- > 0;)
        if (deliver(running[i], emitted[i])) end(i);
    } catch (const std::exception& e) {
      fail_all(e);
    }
  }
  // closed: the running jobs end with what they have produced
  for (auto& r : running) {
    r.job->cancel();
    r.job->finish();
  }
}

}  // namespace ingine::engine
