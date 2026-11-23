#include "model/batch_engine.h"

#include "core/device.h"
#include "kernels/batch/batch.h"
#include "model/batch_drafter.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

constexpr int G = BatchDrafter::GAMMA;
constexpr int TAP_ROW = NUM_TAPS * HIDDEN;  // bf16 per row of taps

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
      spec_(o.spec),
      w_(load_weights(o.model_dir, /*dense_fp8=*/false, stream_.s)),
      abl_(load_ablation(o.ablation)),
      st_(make_batch_state(o.slots, o.ctx, o.kv_fp8)),
      pf_(std::make_unique<Prefiller>(w_, abl_, o.prefill_chunk, stream_.s)),
      dec_(std::make_unique<BatchDecoder>(w_, abl_, st_, o.slots, stream_.s, 8)),
      chunk_(std::size_t(std::max(o.prefill_chunk, 1)) * 4),
      logits_(std::size_t(UNPADDED_VOCAB) * 4),
      first_(4) {
  if (spec_ < 0 || spec_ > G) throw std::invalid_argument("BatchEngine: spec must be in 0..7");
  if (spec_ > 0) {
    drafter_ = std::make_unique<BatchDrafter>(o.model_dir, w_, o.slots, o.ctx, o.slots, stream_.s);
    taps_chunk_ = DeviceBuffer(std::size_t(prefill_chunk_) * TAP_ROW * 2);
    gathered_ = DeviceBuffer(std::size_t(o.slots) * 8 * TAP_ROW * 2);
    rows_ = DeviceBuffer(std::size_t(o.slots) * 8 * 4);
    drafts_.assign(std::size_t(o.slots) * G, 0);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
}

BatchEngine::~BatchEngine() { cudaStreamSynchronize(stream_.s); }

std::string BatchEngine::model_id() const { return ablated_ ? "inkling-2-mini-abliterated" : "inkling-2-mini"; }

std::size_t BatchEngine::device_bytes() const {
  return st_.device_bytes() + dec_->device_bytes() + pf_->device_bytes() + chunk_.bytes + logits_.bytes + first_.bytes +
         (drafter_ ? drafter_->device_bytes() : 0) + taps_chunk_.bytes + gathered_.bytes + rows_.bytes;
}

std::vector<std::int32_t> BatchEngine::admit(std::span<const Admission> requests) {
  std::vector<std::int32_t> first(requests.size());
  std::vector<std::size_t> together;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const Admission& a = requests[i];
    if (a.slot < 0 || a.slot >= st_.slots) throw std::invalid_argument("BatchEngine::admit: slot out of range");
    const int P = int(a.prompt.size());
    if (P < 1 || P + lookahead() + 1 > st_.ctx)
      throw std::invalid_argument("BatchEngine::admit: a prompt of " + std::to_string(P) +
                                  " tokens does not fit a slot of " + std::to_string(st_.ctx) + " positions");
    reset_slot(st_, a.slot, stream_.s);
    if (drafter_) drafter_->reset_slot(a.slot);
    if (P <= kStepPrefill && P + 8 <= st_.ctx) together.push_back(i);
    else prefill_one(a, first[i]);
  }
  if (!together.empty()) prefill_together(requests, together, first);
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
  if (drafter_ && !requests.empty()) {  // drafts after the first tokens
    const int n = int(requests.size());
    slot_.resize(std::size_t(n)), anchor_.resize(std::size_t(n)), next_pos_.resize(std::size_t(n));
    for (int i = 0; i < n; ++i)
      slot_[i] = requests[i].slot, anchor_[i] = first[i], next_pos_[i] = std::int64_t(requests[i].prompt.size());
    draft(n);
  }
  return first;
}

void BatchEngine::prefill_one(const Admission& a, std::int32_t& first) {
  const cudaStream_t s = stream_.s;
  const int P = int(a.prompt.size());
  State v = st_.view(a.slot);
  for (int p = 0; p < P; p += prefill_chunk_) {
    const int T = std::min(prefill_chunk_, P - p);
    CUDA_CHECK(cudaMemcpyAsync(chunk_.p, a.prompt.data() + p, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
    Tensor taps = rm(taps_chunk_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
    pf_->run(v, static_cast<const std::int32_t*>(chunk_.p), T, p, static_cast<float*>(logits_.p),
             drafter_ ? &taps : nullptr);
    if (drafter_) {
      const std::int32_t slot = a.slot, count = T;
      const std::int64_t start = p;
      drafter_->commit(1, &slot, &start, &count, static_cast<const __nv_bfloat16*>(taps_chunk_.p));
    }
  }
  kernels::batch::sample(static_cast<const float*>(logits_.p), UNPADDED_VOCAB, 1, UNPADDED_VOCAB, false, a.temperature,
                         seed_of(++draws_), static_cast<std::int32_t*>(first_.p), s);
  CUDA_CHECK(cudaMemcpyAsync(&first, first_.p, 4, cudaMemcpyDeviceToHost, s));
}

void BatchEngine::prefill_together(std::span<const Admission> requests, std::span<const std::size_t> which,
                                   std::vector<std::int32_t>& first) {
  // per step, every prompt with tokens left feeds its next M = min(8, the longest remainder) of them, a shorter
  // remainder padded with its last token; each prompt keeps its real rows (rollback of the padding)
  std::vector<int> done(which.size(), 0);
  std::vector<std::size_t> member;  // the step's sequence i is which[member[i]]
  for (;;) {
    int M = 0;
    for (std::size_t j = 0; j < which.size(); ++j)
      M = std::max(M, std::min(int(requests[which[j]].prompt.size()) - done[j], 8));
    if (M <= 0) break;
    tok_.clear(), slot_.clear(), pos_.clear(), temp_.clear(), keep_.clear(), member.clear();
    for (std::size_t j = 0; j < which.size(); ++j) {
      const Admission& a = requests[which[j]];
      const int left = int(a.prompt.size()) - done[j];
      if (left <= 0) continue;
      const int k = std::min(left, M);
      for (int r = 0; r < M; ++r) tok_.push_back(a.prompt[std::size_t(done[j] + std::min(r, k - 1))]);
      slot_.push_back(a.slot), pos_.push_back(done[j]), temp_.push_back(a.temperature), keep_.push_back(k);
      member.push_back(j);
    }
    const int n = int(slot_.size());
    const auto got = dec_->step(n, M, tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_));
    for (int i = 0; i < n; ++i) {
      const std::size_t j = member[std::size_t(i)];
      done[j] += keep_[std::size_t(i)];
      if (done[j] == int(requests[which[j]].prompt.size()))
        first[which[j]] = got[std::size_t(i) * std::size_t(M) + std::size_t(keep_[std::size_t(i)]) - 1];
    }
    dec_->rollback(keep_.data());
    if (drafter_) commit_kept(n, M);
  }
}

void BatchEngine::commit_kept(int n, int M) {
  rows_host_.clear();
  for (int i = 0; i < n; ++i)
    for (int r = 0; r < keep_[std::size_t(i)]; ++r) rows_host_.push_back(i * M + r);
  CUDA_CHECK(cudaMemcpyAsync(rows_.p, rows_host_.data(), rows_host_.size() * 4, cudaMemcpyHostToDevice, stream_.s));
  kernels::batch::gather_rows(dec_->taps(), TAP_ROW, static_cast<const std::int32_t*>(rows_.p), int(rows_host_.size()),
                              static_cast<__nv_bfloat16*>(gathered_.p), stream_.s);
  drafter_->commit(n, slot_.data(), pos_.data(), keep_.data(), static_cast<const __nv_bfloat16*>(gathered_.p));
}

void BatchEngine::draft(int n) {
  const BatchDrafter::Drafts d =
      drafter_->draft(n, slot_.data(), anchor_.data(), next_pos_.data(), 0.f, seed_of(++draws_));
  draft_host_.resize(std::size_t(n) * G);
  CUDA_CHECK(cudaMemcpyAsync(draft_host_.data(), d.tokens, draft_host_.size() * 4, cudaMemcpyDeviceToHost, stream_.s));
  CUDA_CHECK(cudaStreamSynchronize(stream_.s));
  for (int i = 0; i < n; ++i)
    std::copy_n(draft_host_.begin() + std::ptrdiff_t(i) * G, G, drafts_.begin() + std::ptrdiff_t(slot_[i]) * G);
}

void BatchEngine::step(std::span<Seq> seqs, std::vector<std::vector<std::int32_t>>& emitted) {
  const int n = int(seqs.size());
  emitted.resize(std::size_t(n));
  if (n == 0) return;
  const int K = spec_, M = K + 1;  // rows per sequence: the pending token and its drafts
  tok_.clear(), slot_.resize(std::size_t(n)), pos_.resize(std::size_t(n)), temp_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    const Seq& q = seqs[i];
    tok_.push_back(q.pending);
    for (int k = 0; k < K; ++k) tok_.push_back(drafts_[std::size_t(q.slot) * G + std::size_t(k)]);
    slot_[i] = q.slot, pos_[i] = q.pos, temp_[i] = q.temperature;
  }
  const auto got = dec_->step(n, M, tok_.data(), slot_.data(), pos_.data(), temp_.data(), seed_of(++draws_));
  if (K == 0) {
    for (int i = 0; i < n; ++i) {
      emitted[i].assign(1, got[std::size_t(i)]);
      seqs[i].pending = got[std::size_t(i)];
      seqs[i].pos += 1;
    }
    return;
  }
  keep_.resize(std::size_t(n)), anchor_.resize(std::size_t(n)), next_pos_.resize(std::size_t(n));
  for (int i = 0; i < n; ++i) {
    const std::int32_t* row = tok_.data() + std::size_t(i) * M;    // pending, drafts
    const std::int32_t* out = got.data() + std::size_t(i) * M;     // the target's token after each row
    int a = 0;
    while (a < K && row[1 + a] == out[a]) ++a;
    emitted[i].assign(row + 1, row + 1 + a);
    emitted[i].push_back(out[a]);
    keep_[i] = a + 1;
    spec_accepted_ += std::uint64_t(a);
    seqs[i].pending = anchor_[i] = out[a];
    seqs[i].pos = next_pos_[i] = pos_[i] + a + 1;
  }
  spec_steps_ += std::uint64_t(n);
  dec_->rollback(keep_.data());
  commit_kept(n, M);
  draft(n);
}

}  // namespace ingine::model
