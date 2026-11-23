// Routed-expert GEMMs of the batched step (decode / verify rows: T up to a few hundred tokens, ~4..32 per expert).
// Same contracts and arithmetic as prefill::moe_gate_up / moe_down (BF16 activations times experts decoded exactly
// as model.decode_experts, FP32 accumulation, BF16 act and down rows); the accumulation order differs. Order per
// MoE: moe_select, moe_gate_up, moe_down (each a programmatic dependent of the one before: launch them back to back
// on one stream), then prefill::moe_combine. Launchers run on `stream`, never allocate or synchronise.
#pragma once

#include "kernels/prefill/prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ingine::kernels::batch {

// (token, expert) pairs per expert tile: 16 up to T = 128, else 32. INGINE_BATCH_MOE_TILE (16 or 32) overrides it.
int moe_tile_rows(int T);
// tile-list capacity (MoeWork::tiles) for every T up to max_T
int moe_tiles_capacity(int max_T);

// prefill::moe_topk and prefill::moe_route's routing (MoeWork::counts need not be zeroed): idx / wts [T, 8] exactly
// as prefill::moe_topk, counts [128], expert-sorted pair lists and tiles of moe_tile_rows(T) pairs. Two kernels,
// the second (and moe_gate_up) launched as programmatic dependents, so launch latencies overlap.
void moe_select(const float* logits, const float* bias, int T, int* idx, float* wts, const prefill::MoeWork& w,
                cudaStream_t stream);

// prefill::moe_gate_up on moe_select's tiles: act[pair] = bf16(softcap(g, 4) * sigmoid(g) * softcap(u, 25)),
// g/u = bf16(W13_e . x[token]); x bf16 [T, H], act bf16 [T * 8, 1152] by sorted pair.
void moe_gate_up(const prefill::MoeExperts& e, const __nv_bfloat16* x, int T, const prefill::MoeWork& w,
                 __nv_bfloat16* act, cudaStream_t stream);
// prefill::moe_down on moe_select's tiles: down[t * 8 + slot] = bf16(W2_e . act[pair]), bf16 [T * 8, H]. Stages its
// first weights while moe_gate_up finishes.
void moe_down(const prefill::MoeExperts& e, const __nv_bfloat16* act, int T, const prefill::MoeWork& w,
              __nv_bfloat16* down, cudaStream_t stream);

}  // namespace ingine::kernels::batch
