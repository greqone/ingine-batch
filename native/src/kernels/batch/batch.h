// Kernels of batched decoding (src/model/batch.*): n sequences per step, M consecutive rows each (rows b*M ..
// b*M+M-1 belong to sequence b), every sequence in its own request slot.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::batch {

// model.static_conv per sequence, with the arithmetic of prefill::static_conv: sequence b's rows continue slot
// slots[b]'s conv state (fp32 [3, C] at state + slots[b] * slot_stride, oldest row first):
// out[r, c] = (sum_j w[c, j] * ext[r + j] + x[r, c]) / sqrt(5), ext = [state, the sequence's rows of x];
// then the state holds the last three inputs. x bf16 [n * M, ld], w fp32 [C, 4], out fp32 [n * M, C].
void conv(const __nv_bfloat16* x, int ld, int n, int M, int C, const float* w, float* state, std::int64_t slot_stride,
          const std::int32_t* slots, float* out, cudaStream_t stream);

// dst[r, c] = float(src[r, c]) for r < rows, c < cols.
void widen(const __nv_bfloat16* src, int ld_src, int rows, int cols, float* dst, int ld_dst, cudaStream_t stream);

// One token per row of logits fp32 [n, ld], v < V. raw: head outputs l_v = w_v . x, scored as model.logits
// s_v = 30 tanh(bf16(l_v) / 30); else the rows are model.logits already (the Prefiller's). Row r's temperature
// temps[r] (device fp32 [n]) or `temperature`: <= 0 is the argmax of s (lowest v on ties), else a sample of
// softmax(s / T) by the Gumbel-max trick, the noise a function of (seed, row, v). out int32 [n].
void sample(const float* logits, int ld, int n, int V, bool raw, const float* temps, std::uint64_t seed,
            std::int32_t* out, cudaStream_t stream);
void sample(const float* logits, int ld, int n, int V, bool raw, float temperature, std::uint64_t seed,
            std::int32_t* out, cudaStream_t stream);

// Speculative verify of M rows per sequence (engine._rollback, per slot): the conv states of every visit, slot 0's
// pointers plus slots[b] * stride (elements); fp32 [48, 3, C] per slot, oldest row first.
struct ConvStates {
  float *k, *v;        // C = 1024 (the attention's k/v convs)
  float *attn, *mlp;   // C = H (the post-attention and post-MLP convs)
  std::int64_t kv_stride, h_stride;
};
// The conv inputs of a verify step: bf16 [48, n * M, C] per conv (the BF16 values the convs read).
struct ConvHistory {
  __nv_bfloat16 *k, *v, *attn, *mlp;
};
// Floats of conv_save's backup per sequence: rows 1, 2 of the four states of the 48 visits.
constexpr std::int64_t conv_backup_floats() { return 48LL * 2 * (2 * 1024 + 2 * 2304); }

// hist_k / hist_v bf16 [rows, 1024] = columns [k_off, k_off + 1024) / [v_off, ...) of proj bf16 [rows, ld].
void keep_kv(const __nv_bfloat16* proj, int ld, int rows, int k_off, int v_off, __nv_bfloat16* hist_k,
             __nv_bfloat16* hist_v, cudaStream_t stream);
// backup fp32 [n, conv_backup_floats()] = rows 1, 2 of slot slots[b]'s conv states (before a verify step).
void conv_save(const ConvStates& st, int n, const std::int32_t* slots, float* backup, cudaStream_t stream);
// After a verify step of M rows per sequence that keeps commit[b] (1..M) rows of sequence b: slot slots[b]'s
// states = rows commit[b] .. commit[b] + 2 of [rows 0..2 before the step, its M inputs] (rows 1, 2 from the
// backup, the inputs from hist rows b * M ..); nothing where commit[b] == M. commit: device int32 [n].
void conv_rollback(const ConvStates& st, int n, const std::int32_t* slots, const float* backup,
                   const ConvHistory& hist, int M, const std::int32_t* commit, cudaStream_t stream);
// dst row i = src row rows[i], rows of row_elems bf16 (a multiple of 8); rows: device int32 [count].
void gather_rows(const __nv_bfloat16* src, int row_elems, const std::int32_t* rows, int count, __nv_bfloat16* dst,
                 cudaStream_t stream);

}  // namespace ingine::kernels::batch
