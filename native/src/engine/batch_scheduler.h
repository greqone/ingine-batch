// Batched scheduling (JobQueue): up to backend.slots() jobs generate at once. One worker thread loops: it admits
// waiting jobs into free slots (their prefill and first token), runs one batched step over every running job,
// pushes each job its tokens and frees the slots of the jobs that ended (stop token, max_new_tokens, the room of the
// slot, cancellation, failure).
#pragma once

#include "engine/scheduler.h"
#include "ingine/batch.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

namespace ingine::engine {

class BatchScheduler final : public JobQueue {
 public:
  // max_queue: jobs allowed to wait for a slot.
  BatchScheduler(model::BatchBackend& backend, std::vector<int32_t> stop_ids, int max_queue);
  ~BatchScheduler() override;
  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  bool submit(std::shared_ptr<Job> job) override;
  void withdraw(const std::shared_ptr<Job>& job) override;
  SchedulerStats stats() const override;
  void close() override;

 private:
  struct Running {
    std::shared_ptr<Job> job;
    int produced = 0, limit = 0;
  };
  void loop();
  // Pushes the job its tokens up to a stop token, its limit or its cancellation; true when the job has ended.
  bool deliver(Running& r, std::span<const int32_t> tokens) const;

  model::BatchBackend& backend_;
  const std::unordered_set<int32_t> stop_ids_;
  const int max_queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> waiting_;
  int active_ = 0;
  bool closed_ = false;
  std::thread thread_;
};

}  // namespace ingine::engine
