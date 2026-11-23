// Attention core of one layer visit for M = 1..8 consecutive decode tokens (the PyTorch extension's kernel ported
// to a raw-pointer launcher; contract in native/KERNELS.md, area `attn`): q/rel post-processing, key/value convs +
// state update, KV cache write, split-K attention with relative bias, sink and output gate, in one kernel.
// Always launched with programmatic dependent launch (as the Python binding does); graph capturable.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::attn {

struct DecodeArgs {
  const float* proj = nullptr;  // fp32 [M, R]: R = 8600 (sliding) or 8664 (full), SEG order
  int M = 1;                    // 1..8
  const float* q_scale = nullptr;   // fp32 [24, 128]
  const float* r_scale = nullptr;   // fp32 [24, 16]
  const float* rel_proj = nullptr;  // fp32 [16, 1024]
  const float* k_conv = nullptr;    // sliding: fp32 [1024, 4] (conv1d order); full: k_static [4, 1024] by lag
  const float* v_conv = nullptr;
  float* conv_k_state = nullptr;    // fp32 [3, 1024]
  float* conv_v_state = nullptr;
  void* k_cache = nullptr;          // [rows, 8, 128] bf16, or float8_e4m3fn (full layers only)
  void* v_cache = nullptr;
  bool cache_fp8 = false;
  // full: max_ctx; sliding: the ring's rows, 1024, or fewer for a ring that never wraps (positions stay below
  // its rows). At most 2^30.
  std::int64_t cache_rows = 0;
  std::int64_t* kpos = nullptr;     // sliding: int64 [cache_rows] slot positions (-1 = empty); full: unused
  const std::int64_t* pos = nullptr;  // int64 device, pos[b] = position of sequence b's first token
  int window = 0;                   // 1024 (sliding ring) or 0 (full layer, linear cache)
  bool log_scaling = false;         // full layers: q, rel *= tau(p)
  void* workspace = nullptr;        // workspace_bytes(M, batch) bytes, zeroed once at allocation
  std::size_t workspace_size = 0;
  __nv_bfloat16* out = nullptr;     // bf16 [batch * M, 3072]
  // A batch of sequences (one per grid z). Sequence b owns rows b*M .. b*M+M-1 of proj and out and position
  // pos[b], and reads and writes the caches, kpos and conv states of slot slots[b] (b without a table): the
  // pointers above are slot 0's, slot s's are them plus s * the stride (in elements).
  int batch = 1;
  const std::int32_t* slots = nullptr;
  std::int64_t cache_stride = 0, kpos_stride = 0, conv_stride = 0;
};

// Bytes of workspace attn_decode needs for up to max_m tokens of up to max_batch sequences: a counter per
// sequence (zero the workspace once; the kernel resets its counters itself) and the split partials. One
// workspace serves every layer of a stream-ordered decode step.
std::size_t workspace_bytes(int max_m, int max_batch = 1);

// How many of the kernel's 16-block clusters (one per KV head, 8 per call) fit on the GPU at once.
int max_active_clusters(int M, int window);

void attn_decode(const DecodeArgs& a, cudaStream_t stream);

}  // namespace ingine::kernels::attn
