// Decode attention for a batch of sequences at short to medium contexts (attn_decode with batch > 1): the
// DecodeArgs contract of attn.h, one block per (key split, KV head, sequence) instead of a 16-block cluster per
// KV head. Graph capturable; programmatic dependent launch as attn_decode.
#pragma once

#include "kernels/attn/attn.h"

#include <cstddef>

namespace ingine::kernels::attn {

// Workspace bytes attn_decode_batched needs for up to max_m tokens of up to max_batch sequences (zeroed once).
std::size_t batched_workspace_bytes(int max_m, int max_batch);

// Key splits per (sequence, KV head) the launch uses for this batch and cache size (1 = no merge pass).
int batched_splits(int batch, std::int64_t cache_rows);

void attn_decode_batched(const DecodeArgs& a, cudaStream_t stream);

}  // namespace ingine::kernels::attn
