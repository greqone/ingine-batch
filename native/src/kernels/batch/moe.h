// Routed-expert GEMMs of the batched step (decode / verify rows: T up to a few hundred tokens, ~4..32 per expert).
// Same contracts and arithmetic as prefill::moe_gate_up / moe_down (BF16 activations times experts decoded exactly
// as model.decode_experts, FP32 accumulation, BF16 act and down rows); the accumulation order differs. Routing:
// prefill::moe_topk, then moe_route below (tiles of moe_tile_rows(T) pairs, which these kernels need), and
// prefill::moe_combine. Launchers run on `stream`, never allocate or synchronise.
#pragma once

#include "kernels/prefill/prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ingine::kernels::batch {

// (token, expert) pairs per expert tile: 16 up to T = 128, else 64. INGINE_BATCH_MOE_TILE (16 or 64) overrides it.
int moe_tile_rows(int T);
// tile-list capacity (MoeWork::tiles) for every T up to max_T
int moe_tiles_capacity(int max_T);

// Expert-sorted pair lists and tiles of moe_tile_rows(T) pairs from idx and counts (prefill::moe_route's layout).
void moe_route(const int* idx, int T, const prefill::MoeWork& w, cudaStream_t stream);

// prefill::moe_gate_up on moe_route's tiles: act[pair] = bf16(softcap(g, 4) * sigmoid(g) * softcap(u, 25)),
// g/u = bf16(W13_e . x[token]); x bf16 [T, H], act bf16 [T * 8, 1152] by sorted pair.
void moe_gate_up(const prefill::MoeExperts& e, const __nv_bfloat16* x, int T, const prefill::MoeWork& w,
                 __nv_bfloat16* act, cudaStream_t stream);
// prefill::moe_down on moe_route's tiles: down[t * 8 + slot] = bf16(W2_e . act[pair]), bf16 [T * 8, H]. Launched as
// a programmatic dependent of moe_gate_up (stages its first weights while moe_gate_up finishes): call it right
// after moe_gate_up on the same stream.
void moe_down(const prefill::MoeExperts& e, const __nv_bfloat16* act, int T, const prefill::MoeWork& w,
              __nv_bfloat16* down, cudaStream_t stream);

}  // namespace ingine::kernels::batch
