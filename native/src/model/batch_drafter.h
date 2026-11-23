// Batched DSpark drafter: the Drafter (model/drafter.h) for many sequences at once, each with its draft context
// in a slot. The weights are shared: every projection, MLP and head is one GEMM over all n x 8 block rows of a
// call (n x 7 for the head); attention runs per sequence over its slot's context and its own 8 bidirectional
// block rows (kernels/batch/draft.h).
//
// * Per-slot state carried between rounds: layers 0 and 2 keep BF16 rings of min(ctx, WINDOW) rows (row =
//   position % rows, with the position of each row: a draft at P attends to P - WINDOW <= pos < P), layer 1 an
//   E4M3 ring of ctx rows (attends to every row below min(P, ctx)), and f_rnn (the Markov head's input from the
//   last committed row). The block rows' keys/values stay in per-call scratch.
// * Shortlist head: the SHORTLIST_ROWS most frequent tokens and the specials, as the Drafter's (its context
//   rows, note_tokens, are not kept here). The Markov head's fold of hidden_out is computed for those rows only.
// * Precision of the block's projections (q/k/v, o, gate/up, down) and the shortlist head:
//   MXFP8    weights and activations as MXFP8 (E4M3, an E8M0 scale per 32 elements), block-scaled FP8 GEMMs.
//   DRAFTER  the Drafter's FP8-row weights dequantized to BF16, BF16 activations and GEMMs: the Drafter's drafts
//            up to accumulation order (for checking this class against it).
//   The context k/v projections of commits are BF16 GEMMs in both (the Drafter's large-commit path), the Markov
//   head's projections BF16.
#pragma once

#include "kernels/batch/draft.h"
#include "kernels/draft/draft.h"
#include "model/model_data.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}
namespace ingine::kernels::bdraft {
class MxGemm;
}

namespace ingine::model {

class BatchDrafter {
 public:
  static constexpr int BLOCK = kernels::draft::BLOCK;  // 8 rows per drafted block
  static constexpr int GAMMA = kernels::draft::GAMMA;  // 7 drafts per round
  static constexpr int SHORTLIST_ROWS = 32768;         // ranked tokens in the shortlist head
  static constexpr int COMMIT_PIECE = 2048;            // commit rows per GEMM pass

  enum class Precision { MXFP8, DRAFTER };

  // mtp.safetensors from model_dir; `w` supplies the embedding, its norm and the unembedding (and must outlive
  // the drafter). slots x ctx positions; draft() and commit() take up to max_seqs sequences.
  BatchDrafter(const std::filesystem::path& model_dir, const Weights& w, int slots, int ctx, int max_seqs,
               cudaStream_t stream, Precision precision = Precision::MXFP8);
  ~BatchDrafter();
  BatchDrafter(const BatchDrafter&) = delete;
  BatchDrafter& operator=(const BatchDrafter&) = delete;

  // Empty draft context of a slot: no valid local-ring rows, f_rnn zero (as a new slot). Stream-ordered.
  void reset_slot(int slot);

  // Target taps become draft context: sequence i of n has count[i] >= 0 rows at positions start[i].. of slot
  // slots[i]; feats = device bf16 [sum count, 4, H] (sequence i's rows after those of sequences < i), the
  // Prefiller's taps. Host arrays (read before return); n <= slots, distinct slots.
  void commit(int n, const std::int32_t* slots, const std::int64_t* start, const std::int32_t* count,
              const __nv_bfloat16* feats);

  // gamma (1..GAMMA) drafts after anchor token anchors[i] at position P[i] of slot slots[i] (every position < P[i]
  // committed), n <= max_seqs: a block of 1 + gamma rows (anchor + gamma masks; the bidirectional attention sees only
  // these, so the drafts differ from a longer block's) and gamma Markov steps. Greedy at temperature <= 0, else
  // Gumbel-max at the temperature with noise gumbel(seed, i, step, candidate) (vary the seed per call). Device
  // outputs in fixed buffers, overwritten by the next draft().
  struct Drafts {
    const std::int32_t* tokens;  // int32 [n, 7]: columns < gamma
    const float* conf;           // fp32 [n, 7]: max softmax probability per row (as Drafter::Draft::conf)
  };
  Drafts draft(int n, const std::int32_t* slots, const std::int32_t* anchors, const std::int64_t* P,
               float temperature, std::uint64_t seed, int gamma = GAMMA);

  int slots() const { return slots_; }
  int ctx() const { return ctx_; }
  int max_seqs() const { return max_seqs_; }
  std::size_t device_bytes() const;  // weights, slot state, scratch and the GEMM workspaces

 private:
  struct Weight {  // [rows, K]: BF16 (DRAFTER) or its Mx copy (MXFP8)
    const __nv_bfloat16* x = nullptr;
    kernels::bdraft::Mx mx{};
  };
  struct Layer {
    const __nv_bfloat16 *in_norm, *post_norm, *ctx_norm, *q_norm, *k_norm, *k_norm_ctx;
    Weight qkv, wo, gate_up, down;  // [5120, H], [H, 3072], [2I, H], [H, I]
    const __nv_bfloat16* wkv_ctx;   // BF16 [2048, H]: k_proj_ctx | v_proj_ctx
    kernels::bdraft::Ring ring;
  };
  // an activation matrix: BF16 (DRAFTER) or its Mx copy (MXFP8)
  struct Act {
    __nv_bfloat16* x = nullptr;
    kernels::bdraft::Mx mx{};
  };

  void* alloc(std::size_t bytes);
  Act act(int rows, int K);
  Weight weight(const __nv_bfloat16* x, int rows, int K);  // from BF16 rows (MXFP8: quantized, x not kept)
  // loads the weights; returns hidden_out (bf16 weight [H, 256], then bias [H]) for the shortlist fold
  DeviceBuffer load(const std::filesystem::path& model_dir);
  void gemm(const __nv_bfloat16* a, int lda, const __nv_bfloat16* b, int ldb, void* c, int ldc, bool c_fp32, int M,
            int N, int K);
  // out bf16 [M, ldc] = in [M, K] . w [N, K]^T
  void linear(const Act& in, const Weight& w, __nv_bfloat16* out, int ldc, int M, int N, int K);

  const Weights& w_;
  cudaStream_t stream_;
  int slots_, ctx_, max_seqs_;
  Precision precision_;
  int inter_ = 0, short_rows_ = 0, short_ld_ = 0;
  std::size_t bytes_ = 0;
  std::vector<DeviceBuffer> storage_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  std::unique_ptr<kernels::bdraft::MxGemm> mx_gemm_;  // MXFP8

  std::array<Layer, 3> layers_{};
  const __nv_bfloat16* alpha_ = nullptr;  // bf16 [4, 4] softmax(fuse_weight)
  const __nv_bfloat16 *final_norm_ = nullptr, *mask_embed_ = nullptr, *f_norm_ = nullptr;
  const __nv_bfloat16 *w1_ = nullptr, *rnn_w_ = nullptr, *rnn_b_ = nullptr;
  __nv_bfloat16 *fold_w_ = nullptr, *fold_b_ = nullptr;  // by shortlist row
  Weight short_head_;                                     // [short_ld_, H]: short_rows_ rows, zero padding
  std::int32_t* short_map_ = nullptr;                     // shortlist row -> token id
  __nv_bfloat16* f_rnn_ = nullptr;                        // bf16 [slots, H]

  // per-call inputs (int32 device arrays) and scratch
  std::int32_t* d_params_ = nullptr;  // draft: slot | anchor | P [max_seqs] each; commit: slot | start | count | row0
  Act xn_, attn_o_, act_, head_in_;  // GEMM inputs (head_in_.x = h_)
  __nv_bfloat16 *res_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *delta_ = nullptr, *gu_ = nullptr, *kb_ = nullptr,
                *vb_ = nullptr, *h_ = nullptr, *f_in_ = nullptr, *base_ = nullptr;
  float *attn_ws_ = nullptr, *part_h_ = nullptr, *part_f_ = nullptr, *cand_val_ = nullptr;
  std::int32_t* cand_idx_ = nullptr;
  std::int32_t* tokens_ = nullptr;
  float* conf_ = nullptr;
  __nv_bfloat16 *commit_xin_ = nullptr, *commit_kv_ = nullptr;
  std::vector<std::int32_t> host_params_;
};

}  // namespace ingine::model
