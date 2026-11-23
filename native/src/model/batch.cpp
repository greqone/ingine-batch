// Batched decoding (batch.h): request slots and one decode step over many sequences, wired as Prefiller::run with
// one row per sequence. Differences from the prefill: the projections are widened to the FP32 rows of the decode
// attention kernel, which attends over each sequence's own slot (positions, caches, conv states), the short
// convolutions continue each slot's state, and the head runs on every row.
#include "model/batch.h"

#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/attn/attn.h"
#include "kernels/batch/batch.h"
#include "kernels/batch/dense.h"
#include "kernels/batch/moe.h"
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace pk = kernels::prefill;
namespace dense = kernels::dense;
namespace attn = kernels::attn;
namespace kb = kernels::batch;

namespace {

constexpr int H = HIDDEN;
constexpr int LD_V = BatchDecoder::kLogitsLd;  // row stride of the logits (aligned rows for cuBLASLt)

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const __nv_bfloat16* bf16p(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }

Tensor zeros(std::vector<DeviceBuffer>& storage, DType dt, std::initializer_list<std::int32_t> shape) {
  std::size_t n = 1;
  for (std::int32_t d : shape) n *= std::size_t(d);
  storage.emplace_back(n * dtype_size(dt));
  CUDA_CHECK(cudaMemset(storage.back().p, 0, storage.back().bytes));
  return rm(storage.back().p, dt, shape);
}

// fast.Buffers.blocks (as prefill.cpp)
std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

}  // namespace

// ------------------------------------------------------------------------------------------------- state

std::size_t BatchState::device_bytes() const {
  std::size_t n = 0;
  for (const auto& b : storage) n += b.bytes;
  return n;
}

State BatchState::view(int s) const {
  if (s < 0 || s >= slots) throw std::out_of_range("BatchState::view: slot " + std::to_string(s));
  State v;
  v.max_ctx = ctx;
  v.kv_fp8 = kv_fp8;
  v.ring_rows = ring_rows;
  const std::size_t es = kv_fp8 ? 1 : 2;
  auto at = [](const Tensor& t, std::size_t bytes) { return static_cast<char*>(t.data) + bytes; };
  v.k_ring = rm(at(k_ring, std::size_t(s) * ring_stride() * 2), DType::BF16, {NUM_SLIDING_VISITS, ring_rows, KV_HEADS, HEAD_DIM});
  v.v_ring = rm(at(v_ring, std::size_t(s) * ring_stride() * 2), DType::BF16, {NUM_SLIDING_VISITS, ring_rows, KV_HEADS, HEAD_DIM});
  v.ring_pos = rm(at(ring_pos, std::size_t(s) * kpos_stride() * 8), DType::I64, {NUM_SLIDING_VISITS, ring_rows});
  v.k_full = rm(at(k_full, std::size_t(s) * full_stride() * es), k_full.dtype, {NUM_FULL_VISITS, ctx, KV_HEADS, HEAD_DIM});
  v.v_full = rm(at(v_full, std::size_t(s) * full_stride() * es), v_full.dtype, {NUM_FULL_VISITS, ctx, KV_HEADS, HEAD_DIM});
  v.conv_k = rm(at(conv_k, std::size_t(s) * conv_kv_stride() * 4), DType::FP32, {NUM_VISITS, CONV_K - 1, KV_DIM});
  v.conv_v = rm(at(conv_v, std::size_t(s) * conv_kv_stride() * 4), DType::FP32, {NUM_VISITS, CONV_K - 1, KV_DIM});
  v.conv_attn = rm(at(conv_attn, std::size_t(s) * conv_h_stride() * 4), DType::FP32, {NUM_VISITS, CONV_K - 1, H});
  v.conv_mlp = rm(at(conv_mlp, std::size_t(s) * conv_h_stride() * 4), DType::FP32, {NUM_VISITS, CONV_K - 1, H});
  v.length = 0;
  return v;
}

BatchState make_batch_state(int slots, int ctx, bool kv_fp8) {
  if (slots < 1 || slots > 4096) throw std::invalid_argument("slots must be in 1..4096");
  if (ctx < 2 || ctx > (1 << 20)) throw std::invalid_argument("slot context must be in 2..2^20");
  BatchState b;
  b.slots = slots;
  b.ctx = ctx;
  b.kv_fp8 = kv_fp8;
  b.ring_rows = std::min(ctx, WINDOW);
  const DType full = kv_fp8 ? DType::FP8_E4M3FN : DType::BF16;
  b.k_ring = zeros(b.storage, DType::BF16, {slots * NUM_SLIDING_VISITS, b.ring_rows, KV_HEADS, HEAD_DIM});
  b.v_ring = zeros(b.storage, DType::BF16, {slots * NUM_SLIDING_VISITS, b.ring_rows, KV_HEADS, HEAD_DIM});
  b.ring_pos = zeros(b.storage, DType::I64, {slots * NUM_SLIDING_VISITS, b.ring_rows});
  CUDA_CHECK(cudaMemset(b.ring_pos.data, 0xff, b.storage.back().bytes));  // -1
  b.k_full = zeros(b.storage, full, {slots * NUM_FULL_VISITS, ctx, KV_HEADS, HEAD_DIM});
  b.v_full = zeros(b.storage, full, {slots * NUM_FULL_VISITS, ctx, KV_HEADS, HEAD_DIM});
  b.conv_k = zeros(b.storage, DType::FP32, {slots * NUM_VISITS, CONV_K - 1, KV_DIM});
  b.conv_v = zeros(b.storage, DType::FP32, {slots * NUM_VISITS, CONV_K - 1, KV_DIM});
  b.conv_attn = zeros(b.storage, DType::FP32, {slots * NUM_VISITS, CONV_K - 1, H});
  b.conv_mlp = zeros(b.storage, DType::FP32, {slots * NUM_VISITS, CONV_K - 1, H});
  return b;
}

void reset_slot(BatchState& st, int s, cudaStream_t stream) {
  const State v = st.view(s);
  CUDA_CHECK(cudaMemsetAsync(v.ring_pos.data, 0xff, std::size_t(st.kpos_stride()) * 8, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_k.data, 0, std::size_t(st.conv_kv_stride()) * 4, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_v.data, 0, std::size_t(st.conv_kv_stride()) * 4, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_attn.data, 0, std::size_t(st.conv_h_stride()) * 4, stream));
  CUDA_CHECK(cudaMemsetAsync(v.conv_mlp.data, 0, std::size_t(st.conv_h_stride()) * 4, stream));
}

// ------------------------------------------------------------------------------------------------ decoder

BatchDecoder::BatchDecoder(const Weights& w, const Ablation& abl, BatchState& st, int max_seqs, cudaStream_t stream,
                           int max_m)
    : w_(w), abl_(abl), st_(st), max_seqs_(max_seqs), max_m_(max_m), stream_(stream),
      gemm_(std::make_unique<kb::TunedGemm>()) {
  if (max_seqs < 1 || max_seqs > st.slots) throw std::invalid_argument("BatchDecoder: max_seqs must be in 1..slots");
  if (max_m < 1 || max_m > 8) throw std::invalid_argument("BatchDecoder: max_m must be in 1..8");
  const std::size_t S = std::size_t(max_seqs), N = S * std::size_t(max_m);  // sequences, rows
  auto alloc = [&](std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  };
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto bf16 = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };
  auto i64 = [&](std::size_t n) { return static_cast<std::int64_t*>(alloc(n * 8)); };
  tokens_ = i32(N), slots_ = i32(S), sampled_ = i32(N), rows_ = i32(S + 1);
  temps_ = f32(N);
  pos_ = i64(S), ids_ = i64(N);
  B0_ = f32(N * H), B1_ = f32(N * H);
  for (int p = 0; p < PASSES; ++p) B2_[p] = f32(N * H), E_[p] = f32(N * H);
  hid_ = f32(N * H), part_ = f32(N * H), y_ = f32(N * H), y2_ = f32(N * H);
  xin_ = bf16(3 * N * H), xm_ = bf16(N * H), xf_ = bf16(N * H);
  mixf_ = f32(3 * N * H), msgs_ = f32(4 * N * H), mixm_ = f32(N * H);
  proj16_ = bf16(N * PROJ_ROWS_FULL);
  attn_out_ = bf16(N * HEADS * HEAD_DIM), wo_out_ = bf16(N * H);
  gin_ = bf16(3 * N * 2 * H), g16_ = bf16(3 * N * GATE_RANK);
  fg_ = f32(3 * N * 2 * H);
  router_logits_ = f32(N * EXPERTS), wts_ = f32(N * TOP_K);
  idx_ = i32(N * TOP_K);
  counts_ = i32(EXPERTS), tiles_n_ = i32(1), pair_token_ = i32(N * TOP_K), pair_dst_ = i32(N * TOP_K);
  tiles_ = static_cast<int4*>(alloc(std::size_t(kb::moe_tiles_capacity(int(N))) * sizeof(int4)));
  act_ = bf16(N * TOP_K * INTER), down_ = bf16(N * TOP_K * H), moe_sum_ = bf16(N * H);
  logits_ = f32(N * LD_V);
  // FP8 dense rows (dense_fp8): dequantized per visit as the Prefiller does
  wscratch_ = w.dense_fp8 ? bf16(std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM)) : nullptr;
  if (max_m > 1) {
    taps_ = bf16(N * NUM_TAPS * H);
    hist_k_ = bf16(NUM_VISITS * N * KV_DIM), hist_v_ = bf16(NUM_VISITS * N * KV_DIM);
    hist_attn_ = bf16(NUM_VISITS * N * H), hist_mlp_ = bf16(NUM_VISITS * N * H);
    commit_ = i32(S);
  }
  for (auto& b : storage_) b.fill(0);
  // the gates' forget16 | gain16 rows per gate (one GEMM of N = 2H per gate), after the weights' uploads
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::size_t gate_elems = std::size_t(H) * GATE_RANK;
  auto* fg_rows = bf16(std::size_t(LOOP_END - LOOP_START) * 4 * 2 * gate_elems);
  for (int l = LOOP_START; l < LOOP_END; ++l) {
    const Layer& layer = w.layers[l];
    __nv_bfloat16* dst = fg_rows + std::size_t(l - LOOP_START) * 4 * 2 * gate_elems;
    fg_qkv_[l] = dst, fg_mlp_[l] = dst + 3 * 2 * gate_elems;
    const std::pair<const Tensor*, const Tensor*> stacks[] = {{&layer.qkv_gate_forget16, &layer.qkv_gate_gain16},
                                                             {&layer.mlp1_gate_forget16, &layer.mlp1_gate_gain16}};
    for (int s = 0; s < 2; ++s) {
      const int G = s == 0 ? 3 : 1;
      __nv_bfloat16* d = dst + (s == 0 ? 0 : 3 * 2 * gate_elems);
      for (int half = 0; half < 2; ++half)
        CUDA_CHECK(cudaMemcpy2D(d + half * gate_elems, 2 * gate_elems * 2,
                                half == 0 ? stacks[s].first->data : stacks[s].second->data, gate_elems * 2,
                                gate_elems * 2, G, cudaMemcpyDeviceToDevice));
    }
  }
  attn_ws_ = DeviceBuffer(attn::workspace_bytes(max_m, max_seqs));
  attn_ws_.fill(0);  // the kernel resets its counters itself
  sampled_host_.resize(N);
  temps_host_.resize(N);
}

BatchDecoder::~BatchDecoder() = default;

std::size_t BatchDecoder::device_bytes() const {
  std::size_t n = attn_ws_.bytes;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

const float* const* BatchDecoder::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const float* const*>(it->second.p);
}

kb::ConvStates BatchDecoder::conv_states() const {
  const State s0 = st_.view(0);
  return {static_cast<float*>(s0.conv_k.data),    static_cast<float*>(s0.conv_v.data),
          static_cast<float*>(s0.conv_attn.data), static_cast<float*>(s0.conv_mlp.data),
          st_.conv_kv_stride(),                   st_.conv_h_stride()};
}

void BatchDecoder::accept(const std::int32_t* kept) {
  if (!last_verify_) return;  // every sequence had one row: its states advanced in the step
  for (int b = 0; b < last_n_; ++b)
    if (kept[b] < 1 || kept[b] > last_m_[std::size_t(b)])
      throw std::invalid_argument("BatchDecoder::accept: kept rows outside 1..m");
  CUDA_CHECK(cudaMemcpyAsync(commit_, kept, std::size_t(last_n_) * 4, cudaMemcpyHostToDevice, stream_));
  kb::conv_commit(conv_states(), last_n_, slots_, {hist_k_, hist_v_, hist_attn_, hist_mlp_},
                  rows_host_[std::size_t(last_n_)], rows_, commit_, stream_);
}

std::span<const std::int32_t> BatchDecoder::step(int n, const std::int32_t* m, const std::int32_t* tokens,
                                                 const std::int32_t* slots, const std::int64_t* pos,
                                                 const float* temperature, std::uint64_t seed) {
  if (n < 1 || n > max_seqs_) throw std::invalid_argument("BatchDecoder::step: n outside 1..max_seqs");
  rows_host_.resize(std::size_t(n) + 1), last_m_.assign(m, m + n);
  rows_host_[0] = 0;
  int M = 0;  // the most rows of a sequence
  for (int b = 0; b < n; ++b) {
    if (m[b] < 1 || m[b] > max_m_) throw std::invalid_argument("BatchDecoder::step: rows outside 1..max_m");
    if (slots[b] < 0 || slots[b] >= st_.slots) throw std::invalid_argument("BatchDecoder::step: slot out of range");
    if (pos[b] < 0 || pos[b] + m[b] > st_.ctx)
      throw std::invalid_argument("BatchDecoder::step: positions outside the slot context");
    rows_host_[std::size_t(b) + 1] = rows_host_[std::size_t(b)] + m[b];
    M = std::max(M, m[b]);
  }
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const int T = rows_host_[std::size_t(n)];  // rows: m[b] per sequence
  const bool verify = M > 1;  // keep taps, conv inputs and the overwritten conv rows (max_m > 1)
  const std::size_t slab = std::size_t(T) * H;
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);
  const State s0 = st_.view(0);  // per-visit base pointers; sequence b's slot adds slots[b] * stride

  CUDA_CHECK(cudaMemcpyAsync(tokens_, tokens, std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(slots_, slots, std::size_t(n) * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(pos_, pos, std::size_t(n) * 8, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(rows_, rows_host_.data(), (std::size_t(n) + 1) * 4, cudaMemcpyHostToDevice, s));
  for (int b = 0; b < n; ++b)
    std::fill(temps_host_.begin() + rows_host_[std::size_t(b)], temps_host_.begin() + rows_host_[std::size_t(b) + 1],
              temperature[b]);
  CUDA_CHECK(cudaMemcpyAsync(temps_, temps_host_.data(), std::size_t(T) * 4, cudaMemcpyHostToDevice, s));
  last_n_ = n, last_verify_ = verify;

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 __nv_bfloat16* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  // B rows of a Dense matrix as BF16: the matrix itself, or its FP8 rows dequantized (weights.dequant)
  auto bf16_rows = [&](const Dense& d) -> const __nv_bfloat16* {
    if (!d.fp8()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    return wscratch_;
  };
  // G gates at once: the down projections as one strided GEMM, then forget | gain (fg rows, N = 2H) as one
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const __nv_bfloat16* fg_rows,
                   const float* norm_w, __nv_bfloat16* out) {
    pk::gate_inputs(rows, G, T, gin_, s);
    const std::int64_t TG = std::int64_t(T) * GATE_RANK;
    gemm_->run(gin_, 2 * H, bf16p(down), 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s,
               {G, std::int64_t(T) * 2 * H, std::int64_t(GATE_RANK) * 2 * H, TG});
    gemm_->run(g16_, GATE_RANK, fg_rows, GATE_RANK, fg_, 2 * H, true, T, 2 * H, GATE_RANK, s,
               {G, TG, std::int64_t(2) * H * GATE_RANK, std::int64_t(T) * 2 * H});
    kb::gate_apply(rows, G, fg_, T, norm_w, out, slab, s);
  };

  pk::widen_ids(tokens_, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  int prev_v = -1, tap = 0;

  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer, p = VISITS[v].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
      base = part_, write = y2_;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = mlp_dirs + std::size_t(pv) * H;
    const float* w_prev = mlp_w + 4 * std::size_t(pv);
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
      if (i == 0) write = nullptr;
    }
    // ---- attention read(s)
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
      pk::GateRows rows;
      for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
      gates(rows, 3, layer.qkv_gate_down, fg_qkv_[l], f32p(layer.attn_norm3), xin_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }
    if (tap < 3 && v == TAP_VISITS[tap]) {
      if (verify) pk::store_tap(hid, T, taps_ + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
      ++tap;
    }

    // ---- projections (fast.proj_layout): one GEMM for a single input, else q | gate | rel | sink from input 0,
    // then k, v (inputs 1, 2) as one strided GEMM, then k_dyn, v_dyn likewise; BF16-rounded, then the FP32 rows the
    // decode attention kernel reads
    const int R = proj_rows(layer.full);
    const __nv_bfloat16* wp = bf16_rows(layer.wproj);
    const bool single = !layer.full && !msgs;
    if (single) {
      gemm_->run(xin_, H, wp, H, proj16_, R, false, T, R, H, s);
    } else {
      static_assert(SEG_INPUT[SEG_Q] == 0 && SEG_INPUT[SEG_SINK] == 0 && SEG_INPUT[SEG_K] == 1 &&
                    SEG_INPUT[SEG_V] == 2 && SEG_INPUT[SEG_K_DYN] == 1 && SEG_INPUT[SEG_V_DYN] == 2);
      static_assert(SEG_ROWS[SEG_K] == SEG_ROWS[SEG_V] && SEG_ROWS[SEG_K_DYN] == SEG_ROWS[SEG_V_DYN]);
      gemm_->run(xin_, H, wp, H, proj16_, R, false, T, SEG_BEGIN[SEG_K], H, s);
      for (const int seg : {SEG_K, SEG_K_DYN}) {
        if (SEG_BEGIN[seg] >= R) break;
        const int rows = SEG_ROWS[seg];
        gemm_->run(xin_ + slab, H, wp + std::size_t(SEG_BEGIN[seg]) * H, H, proj16_ + SEG_BEGIN[seg], R, false, T,
                   rows, H, s, {2, std::int64_t(slab), std::int64_t(rows) * H, rows});
      }
    }
    if (verify)  // the k/v conv inputs (the states are written after acceptance)
      kb::keep_kv(proj16_, R, T, SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], hist_k_ + std::size_t(v) * T * KV_DIM,
                  hist_v_ + std::size_t(v) * T * KV_DIM, s);

    // ---- attention: every sequence over its own slot
    attn::DecodeArgs a;
    a.proj16 = proj16_;
    a.conv_write = !verify;
    a.M = M;
    a.q_scale = f32p(layer.q_scale);
    a.r_scale = f32p(layer.r_scale);
    a.rel_proj = f32p(layer.rel_proj);
    a.conv_k_state = s0.conv_k_at(v);
    a.conv_v_state = s0.conv_v_at(v);
    a.pos = pos_;
    a.workspace = attn_ws_.p;
    a.workspace_size = attn_ws_.bytes;
    a.out = attn_out_;
    a.batch = n;
    a.rows = rows_;
    a.slots = slots_;
    a.conv_stride = st_.conv_kv_stride();
    if (layer.full) {
      const int fi = FULL_INDEX[v];
      a.k_conv = f32p(layer.k_static);
      a.v_conv = f32p(layer.v_static);
      a.k_cache = s0.k_full_at(fi).data;
      a.v_cache = s0.v_full_at(fi).data;
      a.cache_fp8 = st_.kv_fp8;
      a.cache_rows = st_.ctx;
      a.kpos = nullptr;
      a.window = 0;
      a.log_scaling = true;
      a.cache_stride = st_.full_stride();
    } else {
      const int ri = RING_INDEX[v];
      a.k_conv = f32p(layer.k_conv);
      a.v_conv = f32p(layer.v_conv);
      a.k_cache = s0.k_ring_at(ri).data;
      a.v_cache = s0.v_ring_at(ri).data;
      a.cache_fp8 = false;
      a.cache_rows = st_.ring_rows;
      a.kpos = s0.ring_pos_at(ri);
      a.window = WINDOW;
      a.log_scaling = false;
      a.cache_stride = st_.ring_stride();
      a.kpos_stride = st_.kpos_stride();
    }
    attn::attn_decode(a, s);

    // ---- output projection + post-attention conv (each slot's state); a verify keeps the conv inputs
    __nv_bfloat16* const wo_dst = verify ? hist_attn_ + std::size_t(v) * slab : wo_out_;
    gemm_->run(attn_out_, HEADS * HEAD_DIM, bf16_rows(layer.wo), HEADS * HEAD_DIM, wo_dst, H, false, T, H,
               HEADS * HEAD_DIM, s);
    kb::conv(wo_dst, H, n, rows_, H, f32p(layer.attn_conv), s0.conv_attn_at(v), st_.conv_h_stride(), slots_, !verify,
             y_,
             s);

    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
          f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
      pk::GateRows rows;
      rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
      gates(rows, 1, layer.mlp1_gate_down, fg_mlp_[l], f32p(layer.mlp_norm1), xm_);
    } else {
      mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
          f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
    }

    // ---- routed experts (model._moe_grouped) + post-MLP conv (each slot's state)
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    kb::moe_select(router_logits_, f32p(layer.router_bias), T, idx_, wts_, work, s);
    const Experts& ex = layer.experts;
    const pk::MoeExperts me{ex.fmt == ExpertFormat::NVFP4 ? 0 : 1, ex.w13.data, ex.w13_scale.data,
                            f32p(ex.w13_factor), ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor)};
    kb::moe_gate_up(me, xm_, T, work, act_, s);
    kb::moe_down(me, act_, T, work, down_, s);
    __nv_bfloat16* const moe_dst = verify ? hist_mlp_ + std::size_t(v) * slab : moe_sum_;
    pk::moe_combine(down_, wts_, T, moe_dst, s);
    kb::conv(moe_dst, H, n, rows_, H, f32p(layer.mlp_conv), s0.conv_mlp_at(v), st_.conv_h_stride(), slots_, !verify,
             y2_,
             s);
    prev_v = v;
  }

  // ---- final read, taps, the head of every row, one token per row
  mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H, mlp_w + 4 * std::size_t(prev_v), nullptr,
      f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr);
  if (verify)
    CUDA_CHECK(cudaMemcpy2DAsync(taps_ + std::size_t(NUM_TAPS - 1) * H, std::size_t(NUM_TAPS) * H * 2, xf_,
                                 std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, s));
  if (w.unembed.fp8()) {  // model.logits row by row (FP8 rows dequantized in the kernel)
    for (int b = 0; b < T; ++b)
      pk::head_logits(xf_ + std::size_t(b) * H, w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB,
                      logits_ + std::size_t(b) * LD_V, s);
  } else {
    gemm_->run(xf_, H, bf16p(w.unembed.w), H, logits_, LD_V, true, T, UNPADDED_VOCAB, H, s);
  }
  kb::sample(logits_, LD_V, T, UNPADDED_VOCAB, logits_raw(), temps_, seed, sampled_, s);
  CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled_, std::size_t(T) * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  gemm_->collect();
  return {sampled_host_.data(), std::size_t(T)};
}

}  // namespace ingine::model
