#include "model/batch_engine.h"

#include "core/device.h"
#include "kernels/batch/batch.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

// the Gumbel noise of sampling call `draw`
std::uint64_t seed_of(std::uint64_t draw) { return draw * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull; }

}  // namespace

BatchEngine::Stream::Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
BatchEngine::Stream::~Stream() {
  if (s) cudaStreamDestroy(s);
}

BatchEngine::BatchEngine(const BatchEngineOptions& o)
    : ablated_(!o.ablation.empty()),
      prefill_chunk_(o.prefill_chunk),
      w_(load_weights(o.model_dir, /*dense_fp8=*/false, stream_.s)),
      abl_(load_ablation(o.ablation)),
      st_(make_batch_state(o.slots, o.ctx, o.kv_fp8)),
      pf_(std::make_unique<Prefiller>(w_, abl_, o.prefill_chunk, stream_.s)),
      dec_(std::make_unique<BatchDecoder>(w_, abl_, st_, o.slots, stream_.s)),
      chunk_(std::size_t(std::max(o.prefill_chunk, 1)) * 4),
      logits_(std::size_t(UNPADDED_VOCAB) * 4),
      first_(4) {
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
}

BatchEngine::~BatchEngine() { cudaStreamSynchronize(stream_.s); }

std::string BatchEngine::model_id() const { return ablated_ ? "inkling-2-mini-abliterated" : "inkling-2-mini"; }

std::size_t BatchEngine::device_bytes() const {
  return st_.device_bytes() + dec_->device_bytes() + pf_->device_bytes() + chunk_.bytes + logits_.bytes + first_.bytes;
}

std::vector<std::int32_t> BatchEngine::admit(std::span<const Admission> requests) {
  const cudaStream_t s = stream_.s;
  std::vector<std::int32_t> first(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const Admission& a = requests[i];
    if (a.slot < 0 || a.slot >= st_.slots) throw std::invalid_argument("BatchEngine::admit: slot out of range");
    const int P = int(a.prompt.size());
    if (P < 1 || P + lookahead() + 1 > st_.ctx)
      throw std::invalid_argument("BatchEngine::admit: a prompt of " + std::to_string(P) +
                                  " tokens does not fit a slot of " + std::to_string(st_.ctx) + " positions");
    reset_slot(st_, a.slot, s);
    State v = st_.view(a.slot);
    for (int p = 0; p < P; p += prefill_chunk_) {
      const int T = std::min(prefill_chunk_, P - p);
      CUDA_CHECK(cudaMemcpyAsync(chunk_.p, a.prompt.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
      pf_->run(v, static_cast<const std::int32_t*>(chunk_.p), T, p, static_cast<float*>(logits_.p), nullptr);
    }
    kernels::batch::sample(static_cast<const float*>(logits_.p), UNPADDED_VOCAB, 1, UNPADDED_VOCAB, false,
                           a.temperature, seed_of(++draws_), static_cast<std::int32_t*>(first_.p), s);
    CUDA_CHECK(cudaMemcpyAsync(&first[i], first_.p, 4, cudaMemcpyDeviceToHost, s));
  }
  CUDA_CHECK(cudaStreamSynchronize(s));
  return first;
}

void BatchEngine::step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) {
  const int n = int(seqs.size());
  emitted.resize(std::size_t(n));
  if (n == 0) return;
  tok_.resize(std::size_t(n)), slot_.resize(std::size_t(n)), pos_.resize(std::size_t(n)), temp_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    tok_[i] = seqs[i].pending, slot_[i] = seqs[i].slot, pos_[i] = seqs[i].pos, temp_[i] = seqs[i].temperature;
  }
  const auto got = dec_->step(n, 1, tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_));
  for (int i = 0; i < n; ++i) {
    emitted[i].assign(1, got[std::size_t(i)]);
    seqs[i].pending = got[std::size_t(i)];
    seqs[i].pos += 1;
  }
}

}  // namespace ingine::model
