// The batched engine behind batched serving (ingine/batch.h): the weights, the request slots, the Prefiller for
// admissions and the BatchDecoder for the steps.
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

class Prefiller;

struct BatchEngineOptions {
  std::filesystem::path model_dir;
  std::filesystem::path ablation;  // empty: the original model
  int slots = 64;                  // requests decoded together at most
  int ctx = 4096;                  // positions per slot
  int prefill_chunk = 512;         // prompt tokens per prefill call
  bool kv_fp8 = false;             // FP8 E4M3 full-attention caches
};

// BF16 dense weights (the checkpoint's), BF16 sliding caches, the checkpoint's NVFP4 / MXFP8 experts.
class BatchEngine final : public BatchBackend {
 public:
  explicit BatchEngine(const BatchEngineOptions& options);
  ~BatchEngine() override;
  BatchEngine(const BatchEngine&) = delete;
  BatchEngine& operator=(const BatchEngine&) = delete;

  std::string model_id() const override;
  int slots() const override { return st_.slots; }
  int ctx() const override { return st_.ctx; }
  int lookahead() const override { return 0; }
  // Admissions one after the other through the Prefiller, then the first tokens.
  std::vector<std::int32_t> admit(std::span<const Admission> requests) override;
  void step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) override;

  std::size_t device_bytes() const;  // caches and scratch (not the weights)

 private:
  struct Stream {  // destroyed after every member that uses it
    cudaStream_t s = nullptr;
    Stream();
    ~Stream();
  } stream_;
  const bool ablated_;
  const int prefill_chunk_;
  Weights w_;
  Ablation abl_;
  BatchState st_;
  std::unique_ptr<Prefiller> pf_;
  std::unique_ptr<BatchDecoder> dec_;
  DeviceBuffer chunk_, logits_, first_;
  std::uint64_t draws_ = 0;  // seeds the noise of sampled tokens: one per sampling call
  std::vector<std::int32_t> tok_, slot_;
  std::vector<std::int64_t> pos_;
  std::vector<float> temp_;
};

}  // namespace ingine::model
