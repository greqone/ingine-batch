// Kernels of the batched DSpark drafter (model/batch_drafter.*): the arithmetic of kernels/draft (dspark.py)
// for n sequences per call, each with its draft context in a slot. A drafted block is R = 1 + gamma rows per
// sequence (anchor + gamma masks, 2 <= R <= 8): rows b * R .. b * R + R - 1 belong to sequence b of the call;
// draft step i of sequence b is row b * gamma + i of the head and Markov arrays. Per-sequence inputs are device
// arrays; the dense projections between these kernels are GEMMs with BF16 outputs (rounded once): BF16 GEMMs over
// BF16 inputs, or MXFP8 GEMMs (kernels/batch/mx_gemm.h) over the Mx copies the producing kernels write of their
// outputs. Plain launches on `stream`; they never allocate or synchronise.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::bdraft {

// Per-sequence inputs of a draft, int32 device arrays [n].
struct DraftSeqs {
  const std::int32_t* slot = nullptr;
  const std::int32_t* anchor = nullptr;  // token at position P
  const std::int32_t* P = nullptr;       // every position < P committed
};

// Per-sequence inputs of a commit, int32 device arrays [n]: sequence s owns rows row0[s] .. row0[s] + count[s] - 1
// of the commit (ascending row0), at positions start[s] ..
struct CommitSeqs {
  const std::int32_t* slot = nullptr;
  const std::int32_t* start = nullptr;
  const std::int32_t* count = nullptr;
  const std::int32_t* row0 = nullptr;
  int n = 0;
};

// One draft layer's context over all slots: keys/values [slots, 8 heads, rows, 128], BF16 or E4M3 (fp8, no scale:
// the value rounded to nearest even, saturated). Row = position % rows. Local layers (pos != null) record each row's
// position (int32 [slots, rows], -1 = empty) and attend to positions P - 1024 <= pos < P; the global layer
// attends to every row below min(P, rows).
struct Ring {
  void* k = nullptr;
  void* v = nullptr;
  std::int32_t* pos = nullptr;
  int rows = 0;
  bool fp8 = false;
};

// MXFP8 copy of a row-major matrix [rows, K] (K % 128 == 0): q float8_e4m3fn [rows, K] and sf, one E8M0 scale
// (2^(byte - 127)) per 32 consecutive elements of a row, in cuBLASLt's VEC32_UE8M0 tile layout (mx_sf_index):
// x ~ q * scale with scale = 2^ceil(log2(amax / 448)) of the 32 elements (q saturated at +-448). Null q: none.
struct Mx {
  std::uint8_t* q = nullptr;
  std::uint8_t* sf = nullptr;
};
// bytes of sf for [rows, K]: tiles of 128 rows x 4 scales (512 bytes)
inline std::size_t mx_sf_bytes(int rows, int K) {
  return std::size_t((rows + 127) / 128) * std::size_t((K / 32 + 3) / 4) * 512;
}
// byte offset of the scale of (row, k-block kb = k / 32) in sf
__host__ __device__ inline std::size_t mx_sf_index(int row, int kb, int K) {
  const int tiles_k = (K / 32 + 3) / 4;
  return (std::size_t(row / 128) * tiles_k + kb / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + kb % 4;
}
// out = the Mx copy of x bf16 [rows, K] (weights at load time).
void mx_quantize(const __nv_bfloat16* x, int rows, int K, Mx out, cudaStream_t stream);

// ---- commit ----------------------------------------------------------------------------------------------

// Rows [row_base, row_base + rows) of a commit (feats bf16 [rows, 4, H], the piece's rows):
// xin[l, r] = mix_rmsnorm(feats[r], alpha[l], ctx_norm[l]) (bf16 [3, rows, H]), and for the last row of a
// sequence f_rnn[slot] = mix_rmsnorm(feats[r], alpha[3], f_norm) (f_rnn bf16 [slots, H]). alpha bf16 [4, 4].
void commit_mix(const __nv_bfloat16* feats, int rows, int row_base, const CommitSeqs& seqs, const __nv_bfloat16* alpha,
                const __nv_bfloat16* const ctx_norm[3], const __nv_bfloat16* f_norm, __nv_bfloat16* xin,
                __nv_bfloat16* f_rnn, cudaStream_t stream);
// kv[l] bf16 [rows, 2048] (k | v of layer l) of the same rows into the rings: k = bf16(rope(bf16(rms_head(k) *
// k_norm[l]))), v as is, at row position % rings[l].rows of the sequence's slot. A sequence's rows more than
// rings[l].rows before its end are skipped (later rows take their place).
void commit_store(const __nv_bfloat16* const kv[3], int rows, int row_base, const CommitSeqs& seqs,
                  const __nv_bfloat16* const k_norm[3], const Ring rings[3], cudaStream_t stream);

// ---- draft block -----------------------------------------------------------------------------------------

// Block input rows (R per sequence): res[b * R] = bf16(rms(embed[anchor]) * embed_norm), res[b * R + 1..R-1] =
// mask_embed; xn = bf16(rms(res) * in_norm) (bf16 [n * R, H], if not null; its Mx copy to xn_mx); f_in[b] =
// f_rnn[slot] (bf16 [n, H]).
void embed_block(int n, int R, const DraftSeqs& seqs, const __nv_bfloat16* embed, const float* embed_norm,
                 const __nv_bfloat16* mask_embed, const __nv_bfloat16* in_norm, const __nv_bfloat16* f_rnn,
                 __nv_bfloat16* res, __nv_bfloat16* xn, Mx xn_mx, __nv_bfloat16* f_in, cudaStream_t stream);

// q/k/v of the block rows, qkv bf16 [n * R, 5120]: q, k = bf16(rope(bf16(rms_head * norm))) at positions P + r;
// q -> q_out bf16 [n * R, 24, 128]; k, v -> kb, vb [n * R, 8, 128] (E4M3 if fp8, else BF16).
void qkv_rope(const __nv_bfloat16* qkv, int n, int R, const DraftSeqs& seqs, const __nv_bfloat16* q_norm,
              const __nv_bfloat16* k_norm, __nv_bfloat16* q_out, void* kb, void* vb, bool fp8, cudaStream_t stream);

// Bidirectional block attention of each sequence (scale 1/sqrt(128), GQA head h -> kv head h / 3) over its
// slot's ring (see Ring) and its R block rows (kb, vb as qkv_rope wrote them, the ring's precision); out bf16
// [n * R, 3072] (if not null; its Mx copy to out_mx). max_keys: the most keys of any sequence (ring rows it may
// read + R), bounding the chunks. ws: attention_workspace_floats(max_seqs) floats.
std::size_t attention_workspace_floats(int max_seqs);
void block_attention(const __nv_bfloat16* q, int n, int R, const DraftSeqs& seqs, const Ring& ring, const void* kb,
                     const void* vb, int max_keys, float* ws, __nv_bfloat16* out, Mx out_mx, cudaStream_t stream);

// res = bf16(res + delta); xn = bf16(rms(res) * w), rows of H (xn if not null; its Mx copy to xn_mx). With h_out
// (bf16 [n * (R - 1), H]) / h_mx the normed rows 1..R-1 of each block of R rows also go there, compacted (row
// b * R + i -> b * (R - 1) + i - 1).
void add_rmsnorm(__nv_bfloat16* res, const __nv_bfloat16* delta, const __nv_bfloat16* w, __nv_bfloat16* xn, Mx xn_mx,
                 int rows, int R, __nv_bfloat16* h_out, Mx h_mx, cudaStream_t stream);

// out = bf16(silu(gate) * up), gu bf16 [rows, 2I] (gate | up), out bf16 [rows, I] (if not null; its Mx copy to
// out_mx).
void silu_mul(const __nv_bfloat16* gu, int rows, int inter, __nv_bfloat16* out, Mx out_mx, cudaStream_t stream);

// ---- Markov head -----------------------------------------------------------------------------------------

// Top-1024 of each row of base bf16 [rows, S] (row stride ld, a multiple of 8; S >= 1024): the elements above
// the 1024th value by ascending index, then of those equal to it the lowest indices, ascending
// (kernels::draft::topk_candidates' order). cand_val fp32 [rows, 1024]; cand_idx int32 [rows, 1024] = the
// element's index (shortlist row).
void topk_candidates(const __nv_bfloat16* base, int rows, int S, int ld, float* cand_val, std::int32_t* cand_idx,
                     cudaStream_t stream);

// The gamma (1..7) steps of every sequence (kernels::draft markov_step / markov_logits / markov_finish), a cluster
// of two CTAs per sequence.
// Step i: prev = anchor (i = 0) or the token drafted at step i - 1; raw = bf16(bf16(rnn_w[:, :512] [state_i,
// W1[prev]] + part_h[b * gamma + i] + part_f[b]) + rnn_b) (part_h fp32 [n * gamma, 768], part_f fp32 [n, 768]: the
// h and f_rnn parts of the projection), the gates give state_{i+1} (state_0 = 0) and the latent; candidate k's
// logit = bf16(30 tanh(bf16(cand_val + bf16(fold_b + bf16(fold_w . latent))) / 30)) (fold_w bf16 [S, 256], fold_b
// bf16 [S] by shortlist row; cand_* rows b * gamma + i); the pick = argmax of the logits (temperature <= 0) or of
// logits / T + Gumbel noise gumbel(seed, b, i, k), lowest k on ties. tokens int32 [n, 7] = map[cand_idx of the
// pick]; conf fp32 [n, 7] = max softmax probability of the step's logits (columns i < gamma written).
void markov_draft(int n, int gamma, const DraftSeqs& seqs, const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b,
                  const __nv_bfloat16* w1, const float* part_h, const float* part_f, const float* cand_val,
                  const std::int32_t* cand_idx, const std::int32_t* map, const __nv_bfloat16* fold_w,
                  const __nv_bfloat16* fold_b, float temperature, std::uint64_t seed, std::int32_t* tokens, float* conf,
                  cudaStream_t stream);

}  // namespace ingine::kernels::bdraft
