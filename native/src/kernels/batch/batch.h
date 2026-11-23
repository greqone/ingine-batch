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
// s_v = 30 tanh(bf16(l_v) / 30); else the rows are model.logits already (the Prefiller's). temperature <= 0: argmax
// of s (lowest v on ties); else a sample of softmax(s / temperature) by the Gumbel-max trick, the noise a function
// of (seed, row, v). out int32 [n].
void sample(const float* logits, int ld, int n, int V, bool raw, float temperature, std::uint64_t seed,
            std::int32_t* out, cudaStream_t stream);

}  // namespace ingine::kernels::batch
