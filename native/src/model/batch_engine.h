// The batched engine behind batched serving (ingine/batch.h): the weights, the request slots, the Prefiller for
// long admissions, the BatchDecoder for the steps and, with speculative decoding, the BatchDrafter.
#pragma once

#include "core/arena.h"
#include "ingine/batch.h"
#include "model/batch.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ingine::model {

class BatchDrafter;
class Prefiller;

struct BatchEngineOptions {
  std::filesystem::path model_dir;
  std::filesystem::path ablation;  // empty: the original model
  int slots = 64;                  // requests decoded together at most
  int ctx = 4096;                  // positions per slot
  int prefill_chunk = 512;         // prompt tokens per Prefiller call
  int spec = 0;                    // DSpark drafts verified per step (1..7); 0: plain decoding
  bool kv_fp8 = false;             // FP8 E4M3 full-attention caches
};

// BF16 dense weights (the checkpoint's), BF16 sliding caches, the checkpoint's NVFP4 / MXFP8 experts.
//
// Speculative decoding (spec = K): a step verifies [pending token, its K drafts] of every sequence in one
// BatchDecoder step of K + 1 rows and keeps the longest prefix of drafts equal to the tokens the target samples at
// those rows (greedy: its argmax), plus the target's token after them: the target's own samples, so the output is
// distributed as plain decoding at every temperature. The kept rows' taps become draft context and the drafter
// drafts after the new pending tokens (greedy drafts: the most probable match of the target's sample).
class BatchEngine final : public BatchBackend {
 public:
  explicit BatchEngine(const BatchEngineOptions& options);
  ~BatchEngine() override;
  BatchEngine(const BatchEngine&) = delete;
  BatchEngine& operator=(const BatchEngine&) = delete;

  std::string model_id() const override;
  int slots() const override { return st_.slots; }
  int ctx() const override { return st_.ctx; }
  int lookahead() const override { return spec_; }
  // Prompts of up to kStepPrefill tokens: together, in decode steps of up to 8 rows per prompt (a prompt's last
  // block padded, the padding rolled back); longer ones one after the other through the Prefiller. Then the first
  // tokens (and their drafts).
  static constexpr int kStepPrefill = 256;
  std::vector<std::int32_t> admit(std::span<const Admission> requests) override;
  void step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) override;

  std::size_t device_bytes() const;  // caches, drafter and scratch (not the target's weights)
  // Speculative decoding so far: verified sequence steps and the drafts they accepted.
  std::uint64_t spec_steps() const { return spec_steps_; }
  std::uint64_t spec_accepted() const { return spec_accepted_; }

 private:
  struct Stream {  // destroyed after every member that uses it
    cudaStream_t s = nullptr;
    Stream();
    ~Stream();
  } stream_;
  const bool ablated_;
  const int prefill_chunk_, spec_;
  Weights w_;
  Ablation abl_;
  BatchState st_;
  std::unique_ptr<Prefiller> pf_;
  std::unique_ptr<BatchDecoder> dec_;  // 8 rows per sequence at most (admissions, verify steps)
  std::unique_ptr<BatchDrafter> drafter_;
  DeviceBuffer chunk_, logits_, first_;
  DeviceBuffer taps_chunk_;  // speculative: the Prefiller's taps of a chunk
  DeviceBuffer gathered_;    // speculative: the kept rows' taps, by sequence
  DeviceBuffer rows_;        // speculative: int32 rows of the step's taps to gather
  std::uint64_t draws_ = 0;  // seeds the noise of sampled tokens: one per sampling call
  std::uint64_t spec_steps_ = 0, spec_accepted_ = 0;
  std::vector<std::int32_t> drafts_;  // speculative: [slots, GAMMA] drafts after each slot's pending token
  std::vector<std::int32_t> draft_host_;  // speculative: the drafter's [n, GAMMA] output
  std::vector<std::int32_t> tok_, slot_, keep_, rows_host_, anchor_;
  std::vector<std::int64_t> pos_, next_pos_;
  std::vector<float> temp_;

  void prefill_one(const Admission& a, std::int32_t& first);
  void prefill_together(std::span<const Admission> requests, std::span<const std::size_t> which,
                        std::vector<std::int32_t>& first);
  // the kept rows (keep_[i] of sequence i's M rows of the last step, at positions pos_[i]..) become draft context
  void commit_kept(int n, int M);
  // drafts after anchor_[i] at next_pos_[i] for slot_[i], into drafts_
  void draft(int n);
};

}  // namespace ingine::model
