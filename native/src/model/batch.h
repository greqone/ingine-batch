// Batched decoding: request slots with their own caches, and one decode step over many sequences at once (one
// token each), for a server that runs many requests together.
#pragma once

#include "core/arena.h"
#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// The caches of `slots` request slots (one model.State each) for contexts of up to `ctx` tokens (prompt and
// generated): sliding rings of min(ctx, WINDOW) rows (a ring below WINDOW never wraps), full-attention caches of ctx
// rows, conv states. Slot s's memory is slot 0's plus s times the slot stride of each tensor.
struct BatchState {
  int slots = 0, ctx = 0, ring_rows = 0;
  bool kv_fp8 = false;
  Tensor k_ring, v_ring;       // bf16 [slots * 40, ring_rows, 8, 128]
  Tensor ring_pos;             // int64 [slots * 40, ring_rows], -1 = empty
  Tensor k_full, v_full;       // bf16 or fp8 e4m3 [slots * 8, ctx, 8, 128]
  Tensor conv_k, conv_v;       // fp32 [slots * 48, 3, 1024]
  Tensor conv_attn, conv_mlp;  // fp32 [slots * 48, 3, H]
  std::vector<DeviceBuffer> storage;

  // slot strides, in elements
  std::int64_t ring_stride() const { return std::int64_t(NUM_SLIDING_VISITS) * ring_rows * KV_DIM; }
  std::int64_t kpos_stride() const { return std::int64_t(NUM_SLIDING_VISITS) * ring_rows; }
  std::int64_t full_stride() const { return std::int64_t(NUM_FULL_VISITS) * ctx * KV_DIM; }
  std::int64_t conv_kv_stride() const { return std::int64_t(NUM_VISITS) * (CONV_K - 1) * KV_DIM; }
  std::int64_t conv_h_stride() const { return std::int64_t(NUM_VISITS) * (CONV_K - 1) * HIDDEN; }
  std::size_t device_bytes() const;

  // model.State over slot s's memory (the Prefiller runs on it); owns nothing
  State view(int s) const;
};

BatchState make_batch_state(int slots, int ctx, bool kv_fp8);
// model.State.reset for one slot: ring positions -1, conv states 0. Stream-ordered.
void reset_slot(BatchState& st, int s, cudaStream_t stream);

// One decode step for n sequences, one token each: token tokens[b] at position pos[b] of slot slots[b]. The visit
// sequence is the Prefiller's with n rows (fast.FastDecoder.decode's block-state wiring, the mix and message-gate
// kernels, cuBLASLt GEMMs over the dense rows (BF16, or FP8 rows dequantized per visit), the grouped MoE GEMMs); the
// attention is the decode attention kernel over each sequence's slot (attn::DecodeArgs::batch), the short
// convolutions continue each slot's conv state. Then the head over the n rows (one GEMM over BF16 rows, else the
// Prefiller's row head per row) and one token per row (kernels::batch::sample).
class BatchDecoder {
 public:
  BatchDecoder(const Weights& w, const Ablation& abl, BatchState& st, int max_seqs, cudaStream_t stream);
  ~BatchDecoder();
  BatchDecoder(const BatchDecoder&) = delete;
  BatchDecoder& operator=(const BatchDecoder&) = delete;

  // The step for n <= max_seqs sequences (distinct slots; each position < ctx; tokens, slots and pos are host
  // arrays). temperature <= 0: greedy. Returns the n sampled tokens (host memory owned by the decoder, valid until
  // the next step); synchronises the stream.
  std::span<const std::int32_t> step(int n, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, float temperature, std::uint64_t seed);

  int max_seqs() const { return max_seqs_; }
  std::size_t device_bytes() const;
  // the last step's logits (device fp32, row b at b * kLogitsLd): head outputs l when logits_raw()
  // (model.logits = 30 tanh(bf16(l) / 30)), else model.logits
  static constexpr int kLogitsLd = (UNPADDED_VOCAB + 7) / 8 * 8;
  const float* logits() const { return logits_; }
  bool logits_raw() const { return !w_.unembed.fp8(); }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  BatchState& st_;
  int max_seqs_;
  cudaStream_t stream_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  DeviceBuffer attn_ws_;
  std::vector<std::int32_t> sampled_host_;

  // inputs and activations (rows = max_seqs), see batch.cpp
  std::int32_t *tokens_, *slots_, *sampled_;
  std::int64_t *pos_, *ids_;
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  __nv_bfloat16 *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  __nv_bfloat16 *proj16_, *attn_out_, *wo_out_;
  float* proj32_;
  __nv_bfloat16 *gin_, *g16_;
  float *fpre_, *gpre_;
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  float* logits_;
  __nv_bfloat16* wscratch_;
};

}  // namespace ingine::model
