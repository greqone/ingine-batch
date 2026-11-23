// Batched decoding: request slots with their own caches, and one decode step over many sequences at once (one
// token each, or a speculative verify of several), for a server that runs many requests together.
#pragma once

#include "core/arena.h"
#include "kernels/batch/batch.h"
#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::batch {
class TunedGemm;
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

// One decode step for n sequences of M consecutive rows each (M = 1: plain decoding; M > 1: a speculative verify
// of the anchor and M - 1 drafts): rows b*M .. b*M+M-1 are tokens at positions pos[b] .. pos[b]+M-1 of slot
// slots[b]. The visit sequence is the Prefiller's with n*M rows (fast.FastDecoder.decode's block-state wiring, the
// mix and message-gate kernels, GEMMs over the dense rows (BF16, or FP8 rows dequantized per visit; algorithms
// measured in place, kernels::batch::TunedGemm), the grouped MoE GEMMs); the attention is the decode attention
// kernel over each sequence's slot (attn::DecodeArgs::batch, M rows causal), the short convolutions continue each
// slot's conv state. Then the head over every row and one token per row (kernels::batch::sample).
//
// With max_m > 1 a step of M > 1 rows also keeps what speculative decoding needs: the rows' DSpark taps
// (taps()), and the conv inputs plus the conv rows the step overwrites, so that rollback() can keep only each
// sequence's accepted rows (engine._rollback). The caches need no rollback: a slot's rings never wrap, and the
// rows written past a sequence's accepted length are never read before they are written again.
class BatchDecoder {
 public:
  BatchDecoder(const Weights& w, const Ablation& abl, BatchState& st, int max_seqs, cudaStream_t stream,
               int max_m = 1);
  ~BatchDecoder();
  BatchDecoder(const BatchDecoder&) = delete;
  BatchDecoder& operator=(const BatchDecoder&) = delete;

  // The step for n <= max_seqs sequences, sequence b with m[b] rows (1..max_m) at positions pos[b] .. pos[b] + m[b]
  // - 1 of slot slots[b] (distinct slots; positions below ctx); tokens [sum m] hold the sequences' rows back to back.
  // m, tokens, slots, pos and temperature [n] are host arrays; temperature <= 0: greedy. Returns the sampled tokens,
  // one per row (row r's token follows row r; host memory owned by the decoder, valid until the next step);
  // synchronises the stream.
  std::span<const std::int32_t> step(int n, const std::int32_t* m, const std::int32_t* tokens,
                                     const std::int32_t* slots, const std::int64_t* pos, const float* temperature,
                                     std::uint64_t seed);
  // every sequence with M rows
  std::span<const std::int32_t> step(int n, int M, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, const float* temperature, std::uint64_t seed) {
    return step(n, std::vector<std::int32_t>(std::size_t(n), M).data(), tokens, slots, pos, temperature, seed);
  }
  // every sequence with M rows at the same temperature
  std::span<const std::int32_t> step(int n, int M, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, float temperature, std::uint64_t seed) {
    return step(n, M, tokens, slots, pos, std::vector<float>(std::size_t(n), temperature).data(), seed);
  }
  std::span<const std::int32_t> step(int n, const std::int32_t* tokens, const std::int32_t* slots,
                                     const std::int64_t* pos, float temperature, std::uint64_t seed) {
    return step(n, 1, tokens, slots, pos, temperature, seed);
  }
  // After a step: sequence b keeps its first commit[b] rows (1..m[b]; host array of the step's n): its slot's conv
  // states become those after them. Stream-ordered.
  void rollback(const std::int32_t* commit);
  // The last step's DSpark taps (max_m > 1, a step with more than one row for some sequence): device bf16 [rows, 4,
  // H] (model.TAP_VISITS partial blocks, then the final normed hidden), the Prefiller's layout.
  const __nv_bfloat16* taps() const { return taps_; }

  int max_seqs() const { return max_seqs_; }
  int max_m() const { return max_m_; }
  std::size_t device_bytes() const;
  // the last step's logits (device fp32, row b at b * kLogitsLd): head outputs l when logits_raw()
  // (model.logits = 30 tanh(bf16(l) / 30)), else model.logits
  static constexpr int kLogitsLd = (UNPADDED_VOCAB + 7) / 8 * 8;
  const float* logits() const { return logits_; }
  bool logits_raw() const { return !w_.unembed.fp8(); }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);
  kernels::batch::ConvStates conv_states() const;  // slot 0's conv states and the slot strides

  const Weights& w_;
  const Ablation& abl_;
  BatchState& st_;
  int max_seqs_, max_m_;
  int last_n_ = 0;            // the last step's sequences (rollback)
  bool last_verify_ = false;  // ... and whether it kept the conv history (some sequence had more than one row)
  cudaStream_t stream_;
  std::unique_ptr<kernels::batch::TunedGemm> gemm_;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  DeviceBuffer attn_ws_;
  std::vector<std::int32_t> sampled_host_;
  std::vector<float> temps_host_;            // per row
  std::vector<std::int32_t> rows_host_, last_m_;  // the last step's row offsets [n + 1] and rows per sequence

  // inputs and activations (rows = max_seqs * max_m), see batch.cpp
  std::int32_t *tokens_, *slots_, *sampled_;
  std::int32_t* rows_;  // row offsets, device [max_seqs + 1]
  float* temps_;        // per row
  std::int64_t *pos_, *ids_;
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  __nv_bfloat16 *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  __nv_bfloat16 *proj16_, *attn_out_, *wo_out_;
  float* proj32_;
  __nv_bfloat16 *gin_, *g16_;
  float* fg_;  // the gates' forget | gain products, fp32 [3, rows, 2H]
  // per layer (loop layers): the gates' forget16 | gain16 rows, bf16 [G, 2H, GATE_RANK] (G = 3 qkv, 1 mlp)
  std::array<const __nv_bfloat16*, LAYERS> fg_qkv_{}, fg_mlp_{};
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  float* logits_;
  __nv_bfloat16* wscratch_;
  // speculative verify (max_m > 1): taps, conv inputs per visit, conv rows the step overwrites, accepted rows
  __nv_bfloat16* taps_ = nullptr;
  __nv_bfloat16 *hist_k_ = nullptr, *hist_v_ = nullptr, *hist_attn_ = nullptr, *hist_mlp_ = nullptr;
  float* conv_backup_ = nullptr;
  std::int32_t* commit_ = nullptr;
};

}  // namespace ingine::model
