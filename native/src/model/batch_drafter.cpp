#include "model/batch_drafter.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/batch/mx_gemm.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/shortlist.h"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kbd = kernels::bdraft;
namespace kdense = kernels::dense;

namespace {

constexpr int H = kd::H;
constexpr std::size_t GEMM_WORKSPACE = std::size_t(32) << 20;  // kernels::prefill::Gemm's
constexpr int SPECIALS = 199990;  // <|unused|>, <|endoftext|> and the added tokens: always in the shortlist

bool is_local(int l) { return l != 1; }  // dspark.LOCAL = (0, 2)

std::vector<float> bf16_to_float(const std::vector<std::byte>& raw) {
  std::vector<float> out(raw.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint16_t h;
    std::memcpy(&h, raw.data() + 2 * i, 2);
    const std::uint32_t u = std::uint32_t(h) << 16;
    std::memcpy(&out[i], &u, 4);
  }
  return out;
}

}  // namespace

void* BatchDrafter::alloc(std::size_t bytes) {
  storage_.emplace_back(bytes);
  bytes_ += bytes;
  return storage_.back().p;
}

std::size_t BatchDrafter::device_bytes() const {
  return bytes_ + GEMM_WORKSPACE + (mx_gemm_ ? kbd::MxGemm::WORKSPACE_BYTES : 0);
}

BatchDrafter::Act BatchDrafter::act(int rows, int K) {
  Act a;
  if (precision_ == Precision::MXFP8) {
    a.mx.q = static_cast<std::uint8_t*>(alloc(std::size_t(rows) * K));
    a.mx.sf = static_cast<std::uint8_t*>(alloc(kbd::mx_sf_bytes(rows, K)));
  } else {
    a.x = static_cast<__nv_bfloat16*>(alloc(std::size_t(rows) * K * 2));
  }
  return a;
}

BatchDrafter::Weight BatchDrafter::weight(const __nv_bfloat16* x, int rows, int K) {
  Weight w;
  if (precision_ == Precision::MXFP8) {
    w.mx.q = static_cast<std::uint8_t*>(alloc(std::size_t(rows) * K));
    w.mx.sf = static_cast<std::uint8_t*>(alloc(kbd::mx_sf_bytes(rows, K)));
    kbd::mx_quantize(x, rows, K, w.mx, stream_);
  } else {
    // the Drafter's FP8 rows (kernels::dense::fp8_rows), dequantized as weights.dequant
    auto* out = static_cast<__nv_bfloat16*>(alloc(std::size_t(rows) * K * 2));
    DeviceBuffer q(std::size_t(rows) * K), scale(std::size_t(rows) * 4);
    kdense::fp8_rows(x, rows, K, q.p, static_cast<float*>(scale.p), stream_);
    kdense::dequant_rows(q.p, static_cast<const float*>(scale.p), rows, K, out, stream_);
    w.x = out;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // the caller frees x
  return w;
}

BatchDrafter::BatchDrafter(const std::filesystem::path& model_dir, const Weights& w, int slots, int ctx, int max_seqs,
                           cudaStream_t stream, Precision precision)
    : w_(w), stream_(stream), slots_(slots), ctx_(ctx), max_seqs_(max_seqs), precision_(precision),
      gemm_(std::make_unique<kernels::prefill::Gemm>()) {
  if (slots < 1 || max_seqs < 1 || ctx < 2 * BLOCK) throw std::invalid_argument("BatchDrafter: bad slots/ctx/max_seqs");
  if (precision_ == Precision::MXFP8) mx_gemm_ = std::make_unique<kbd::MxGemm>();
  const int V = w.unembed.rows();
  const DeviceBuffer hidden = load(model_dir);

  auto bf16s = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto f32s = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto i32s = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * 4)); };

  // the shortlist head: the SHORTLIST_ROWS most frequent tokens and the specials by ascending id (the Drafter's
  // rows, so equal base logits break ties the same way); fold_w / fold_b of those rows
  {
    std::vector<std::uint8_t> in(V, 0);
    int taken = 0;
    for (std::uint32_t t : kd::draft_shortlist_ranks())
      if (taken < SHORTLIST_ROWS && t < std::uint32_t(V) && !in[t]) in[t] = 1, ++taken;
    for (int t = SPECIALS; t < V; ++t) in[t] = 1;
    std::vector<std::int32_t> list;
    for (int t = 0; t < V; ++t)
      if (in[t]) list.push_back(t);
    short_rows_ = int(list.size());
    short_ld_ = (short_rows_ + 63) / 64 * 64;  // aligned GEMM columns (the padding rows are zero, never read)
    const std::size_t S = std::size_t(short_rows_);
    DeviceBuffer ids(S * 4), head(std::size_t(short_ld_) * H * 2);
    CUDA_CHECK(cudaMemcpyAsync(ids.p, list.data(), S * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(head.p, 0, std::size_t(short_ld_) * H * 2, stream_));
    auto* rows = static_cast<__nv_bfloat16*>(head.p);
    short_map_ = i32s(S);
    const auto* hw = static_cast<const __nv_bfloat16*>(hidden.p);
    const auto* hb = hw + std::size_t(H) * kd::RANK;
    fold_w_ = bf16s(S * kd::RANK);
    fold_b_ = bf16s(S);
    if (w.unembed.fp8()) {
      DeviceBuffer q(S * H), scale(S * 4);
      kd::gather_head_rows(w.unembed.w.data, static_cast<const float*>(w.unembed.scale.data), H,
                           static_cast<const std::int32_t*>(ids.p), short_rows_, q.p, static_cast<float*>(scale.p),
                           short_map_, stream_);
      kd::fold_head(q.p, static_cast<const float*>(scale.p), short_rows_, hw, hb, fold_w_, fold_b_, stream_);
      kdense::dequant_rows(q.p, static_cast<const float*>(scale.p), short_rows_, H, rows, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    } else {
      kd::gather_head_rows(w.unembed.w.data, nullptr, std::size_t(H) * 2, static_cast<const std::int32_t*>(ids.p),
                           short_rows_, rows, nullptr, short_map_, stream_);
      kd::fold_head(rows, nullptr, short_rows_, hw, hb, fold_w_, fold_b_, stream_);
    }
    if (precision_ == Precision::MXFP8) {
      short_head_ = weight(rows, short_ld_, H);
    } else {  // the unembedding rows as the Drafter's shortlist head holds them
      auto* d = bf16s(std::size_t(short_ld_) * H);
      CUDA_CHECK(cudaMemcpyAsync(d, rows, std::size_t(short_ld_) * H * 2, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      short_head_.x = d;
    }
  }

  // slot state
  const int local_rows = std::min(kd::WINDOW, ctx_);
  for (int l = 0; l < 3; ++l) {
    kbd::Ring& R = layers_[l].ring;
    R.rows = is_local(l) ? local_rows : ctx_;
    R.fp8 = !is_local(l);  // the Drafter's global ring is E4M3, its local rings BF16
    const std::size_t bytes = std::size_t(slots_) * R.rows * kd::KV_DIM * (R.fp8 ? 1 : 2);
    R.k = alloc(bytes);
    R.v = alloc(bytes);
    CUDA_CHECK(cudaMemsetAsync(R.k, 0, bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(R.v, 0, bytes, stream_));
    if (is_local(l)) {
      R.pos = i32s(std::size_t(slots_) * R.rows);
      CUDA_CHECK(cudaMemsetAsync(R.pos, 0xff, std::size_t(slots_) * R.rows * 4, stream_));
    }
  }
  f_rnn_ = bf16s(std::size_t(slots_) * H);
  CUDA_CHECK(cudaMemsetAsync(f_rnn_, 0, std::size_t(slots_) * H * 2, stream_));

  // scratch
  const int M = max_seqs_ * BLOCK, MH = max_seqs_ * GAMMA;
  d_params_ = i32s(std::max<std::size_t>(3 * std::size_t(max_seqs_), 4 * std::size_t(slots_)));
  xn_ = act(M, H);
  attn_o_ = act(M, kd::Q_DIM);
  act_ = act(M, inter_);
  h_ = bf16s(std::size_t(MH) * H);
  head_in_ = precision_ == Precision::MXFP8 ? act(MH, H) : Act{};
  head_in_.x = h_;
  res_ = bf16s(std::size_t(M) * H);
  qkv_ = bf16s(std::size_t(M) * kd::QKV_ROWS);
  q_ = bf16s(std::size_t(M) * kd::Q_DIM);
  delta_ = bf16s(std::size_t(M) * H);
  gu_ = bf16s(std::size_t(M) * 2 * inter_);
  kb_ = bf16s(std::size_t(M) * kd::KV_DIM);
  vb_ = bf16s(std::size_t(M) * kd::KV_DIM);
  attn_ws_ = f32s(kbd::attention_workspace_floats(max_seqs_));
  f_in_ = bf16s(std::size_t(max_seqs_) * H);
  base_ = bf16s(std::size_t(MH) * short_ld_);
  part_h_ = f32s(std::size_t(MH) * kd::RNN_OUT);
  part_f_ = f32s(std::size_t(max_seqs_) * kd::RNN_OUT);
  cand_val_ = f32s(std::size_t(MH) * kd::CANDIDATES);
  cand_idx_ = i32s(std::size_t(MH) * kd::CANDIDATES);
  tokens_ = i32s(MH);
  conf_ = f32s(MH);
  commit_xin_ = bf16s(std::size_t(3) * COMMIT_PIECE * H);
  commit_kv_ = bf16s(std::size_t(3) * COMMIT_PIECE * 2 * kd::KV_DIM);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

BatchDrafter::~BatchDrafter() = default;

DeviceBuffer BatchDrafter::load(const std::filesystem::path& model_dir) {
  SafetensorsFile f(model_dir / "mtp.safetensors");
  const std::uint64_t base = f.data_offset();
  DeviceBuffer raw(f.file_size() - base);
  {
    std::vector<UploadRequest> reqs;
    for (const SafetensorsTensor& t : f.tensors())
      reqs.push_back(upload_request(f, t, static_cast<std::byte*>(raw.p) + (t.offset - base)));
    DeviceUploader up(UploadOptions{std::size_t(16) << 20, 4, 4});
    up.upload(reqs, stream_);
  }
  using Parts = std::initializer_list<std::pair<std::string, std::vector<std::int64_t>>>;
  // Copies tensors (concatenated along rows) to dst.
  auto gather = [&](Parts parts, std::byte* dst) {
    std::size_t off = 0;
    for (const auto& [name, shape] : parts) {
      const SafetensorsTensor& t = f.at(name);
      if (t.dtype != "BF16" || t.shape != shape) throw std::runtime_error("mtp.safetensors: unexpected " + name);
      CUDA_CHECK(cudaMemcpyAsync(dst + off, static_cast<const std::byte*>(raw.p) + (t.offset - base), t.bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
      off += t.bytes;
    }
    return off;
  };
  auto bytes_of = [&](Parts parts) {
    std::size_t bytes = 0;
    for (const auto& [name, shape] : parts) bytes += f.at(name).bytes;
    return bytes;
  };
  auto persist = [&](Parts parts) {
    auto* dst = static_cast<std::byte*>(alloc(bytes_of(parts)));
    gather(parts, dst);
    return reinterpret_cast<const __nv_bfloat16*>(dst);
  };
  // a projection [rows, K] at the class's precision
  auto proj = [&](Parts parts) {
    const std::int64_t K = parts.begin()->second.at(1);
    const std::size_t bytes = bytes_of(parts);
    DeviceBuffer cat(bytes);
    gather(parts, static_cast<std::byte*>(cat.p));
    return weight(static_cast<const __nv_bfloat16*>(cat.p), int(bytes / 2 / std::size_t(K)), int(K));
  };
  const std::int64_t h = H;
  inter_ = int(f.at("layers.0.mlp.gate_proj.weight").shape.at(0));
  const std::int64_t I = inter_;

  {
    const SafetensorsTensor& t = f.at("fuse_weight");
    if (t.dtype != "BF16" || t.shape != std::vector<std::int64_t>{4, 4})
      throw std::runtime_error("mtp.safetensors: unexpected fuse_weight");
    const std::vector<float> fw = bf16_to_float(f.read("fuse_weight"));
    __nv_bfloat16 host[16];
    for (int r = 0; r < 4; ++r) {
      float m = -INFINITY;
      for (int c = 0; c < 4; ++c) m = std::max(m, fw[r * 4 + c]);
      float e[4], s = 0.f;
      for (int c = 0; c < 4; ++c) s += (e[c] = std::exp(fw[r * 4 + c] - m));
      for (int c = 0; c < 4; ++c) host[r * 4 + c] = __float2bfloat16_rn(e[c] / s);
    }
    auto* d = static_cast<__nv_bfloat16*>(alloc(sizeof(host)));
    CUDA_CHECK(cudaMemcpyAsync(d, host, sizeof(host), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    alpha_ = d;
  }
  for (int l = 0; l < 3; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const std::string a = p + "self_attn.";
    Layer& L = layers_[l];
    L.in_norm = persist({{p + "input_layernorm.weight", {h}}});
    L.post_norm = persist({{p + "post_attention_layernorm.weight", {h}}});
    L.ctx_norm = persist({{p + "dflare_norm.weight", {h}}});
    L.q_norm = persist({{a + "q_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm = persist({{a + "k_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm_ctx = persist({{a + "k_norm_ctx.weight", {kd::HEAD_DIM}}});
    L.qkv = proj({{a + "q_proj.weight", {kd::Q_DIM, h}}, {a + "k_proj.weight", {kd::KV_DIM, h}},
                  {a + "v_proj.weight", {kd::KV_DIM, h}}});
    L.wo = proj({{a + "o_proj.weight", {h, kd::Q_DIM}}});
    L.gate_up = proj({{p + "mlp.gate_proj.weight", {I, h}}, {p + "mlp.up_proj.weight", {I, h}}});
    L.down = proj({{p + "mlp.down_proj.weight", {h, I}}});
    L.wkv_ctx = persist({{a + "k_proj_ctx.weight", {kd::KV_DIM, h}}, {a + "v_proj_ctx.weight", {kd::KV_DIM, h}}});
  }
  final_norm_ = persist({{"norm.weight", {h}}});
  mask_embed_ = persist({{"mask_embed", {h}}});
  const std::string mh = "markov_head.";
  const std::int64_t w1_rows = f.at(mh + "markov_w1.weight").shape.at(0);
  if (w1_rows < w_.unembed.rows()) throw std::runtime_error("mtp.safetensors: markov_w1 has fewer rows than the vocabulary");
  w1_ = persist({{mh + "markov_w1.weight", {w1_rows, kd::RANK}}});
  rnn_w_ = persist({{mh + "rnn_proj.weight", {kd::RNN_OUT, kd::RNN_IN}}});
  rnn_b_ = persist({{mh + "rnn_proj.bias", {kd::RNN_OUT}}});
  f_norm_ = persist({{mh + "dflare_norm.weight", {h}}});
  // hidden_out (weight [H, 256], then bias [H]) for the constructor's fold of the shortlist rows
  DeviceBuffer hidden(std::size_t(H) * (kd::RANK + 1) * 2);
  gather({{mh + "hidden_out.weight", {h, kd::RANK}}, {mh + "hidden_out.bias", {h}}}, static_cast<std::byte*>(hidden.p));
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // raw is freed on return
  return hidden;
}

void BatchDrafter::gemm(const __nv_bfloat16* a, int lda, const __nv_bfloat16* b, int ldb, void* c, int ldc, bool c_fp32,
                        int M, int N, int K) {
  gemm_->run(a, lda, b, ldb, c, ldc, c_fp32, M, N, K, stream_);
}

void BatchDrafter::linear(const Act& in, const Weight& w, __nv_bfloat16* out, int ldc, int M, int N, int K) {
  if (precision_ == Precision::MXFP8) mx_gemm_->run(in.mx.q, in.mx.sf, w.mx.q, w.mx.sf, out, ldc, M, N, K, stream_);
  else gemm(in.x, K, w.x, K, out, ldc, false, M, N, K);
}

void BatchDrafter::reset_slot(int slot) {
  if (slot < 0 || slot >= slots_) throw std::invalid_argument("BatchDrafter::reset_slot: slot out of range");
  for (const Layer& L : layers_)
    if (L.ring.pos != nullptr)
      CUDA_CHECK(cudaMemsetAsync(L.ring.pos + std::size_t(slot) * L.ring.rows, 0xff, std::size_t(L.ring.rows) * 4,
                                 stream_));
  CUDA_CHECK(cudaMemsetAsync(f_rnn_ + std::size_t(slot) * H, 0, std::size_t(H) * 2, stream_));
}

void BatchDrafter::commit(int n, const std::int32_t* slots, const std::int64_t* start, const std::int32_t* count,
                          const __nv_bfloat16* feats) {
  if (n < 0 || n > slots_) throw std::invalid_argument("BatchDrafter::commit: n out of range");
  // per-sequence arrays of the sequences with rows: slot | start | count | row0, slots_ entries each
  host_params_.assign(4 * std::size_t(slots_), 0);
  int m = 0, total = 0;
  for (int i = 0; i < n; ++i) {
    if (slots[i] < 0 || slots[i] >= slots_ || start[i] < 0 || count[i] < 0 || start[i] + count[i] > (std::int64_t(1) << 30))
      throw std::invalid_argument("BatchDrafter::commit: bad slot/start/count");
    if (count[i] == 0) continue;
    host_params_[m] = slots[i];
    host_params_[slots_ + m] = std::int32_t(start[i]);
    host_params_[2 * slots_ + m] = count[i];
    host_params_[3 * slots_ + m] = total;
    total += count[i];
    ++m;
  }
  if (total == 0) return;
  // pageable source: staged before cudaMemcpyAsync returns
  CUDA_CHECK(cudaMemcpyAsync(d_params_, host_params_.data(), host_params_.size() * 4, cudaMemcpyHostToDevice, stream_));
  const kbd::CommitSeqs S{d_params_, d_params_ + slots_, d_params_ + 2 * slots_, d_params_ + 3 * slots_, m};
  const __nv_bfloat16* ctx_norm[3] = {layers_[0].ctx_norm, layers_[1].ctx_norm, layers_[2].ctx_norm};
  const __nv_bfloat16* k_norm[3] = {layers_[0].k_norm_ctx, layers_[1].k_norm_ctx, layers_[2].k_norm_ctx};
  const kbd::Ring rings[3] = {layers_[0].ring, layers_[1].ring, layers_[2].ring};
  for (int r0 = 0; r0 < total; r0 += COMMIT_PIECE) {
    const int rows = std::min(COMMIT_PIECE, total - r0);
    kbd::commit_mix(feats + std::size_t(r0) * 4 * H, rows, r0, S, alpha_, ctx_norm, f_norm_, commit_xin_, f_rnn_,
                    stream_);
    const __nv_bfloat16* kv[3];
    for (int l = 0; l < 3; ++l) {
      __nv_bfloat16* out = commit_kv_ + std::size_t(l) * rows * 2 * kd::KV_DIM;
      gemm(commit_xin_ + std::size_t(l) * rows * H, H, layers_[l].wkv_ctx, H, out, 2 * kd::KV_DIM, false, rows,
           2 * kd::KV_DIM, H);
      kv[l] = out;
    }
    kbd::commit_store(kv, rows, r0, S, k_norm, rings, stream_);
  }
}

BatchDrafter::Drafts BatchDrafter::draft(int n, const std::int32_t* slots, const std::int32_t* anchors,
                                         const std::int64_t* P, float temperature, std::uint64_t seed) {
  if (n < 1 || n > max_seqs_) throw std::invalid_argument("BatchDrafter::draft: n out of range");
  const int V = w_.unembed.rows();
  host_params_.assign(3 * std::size_t(max_seqs_), 0);
  std::array<int, 3> max_keys{};
  for (int i = 0; i < n; ++i) {
    if (slots[i] < 0 || slots[i] >= slots_ || anchors[i] < 0 || anchors[i] >= V || P[i] < 0 || P[i] > (1 << 30))
      throw std::invalid_argument("BatchDrafter::draft: bad slot/anchor/P");
    host_params_[i] = slots[i];
    host_params_[max_seqs_ + i] = anchors[i];
    host_params_[2 * max_seqs_ + i] = std::int32_t(P[i]);
    for (int l = 0; l < 3; ++l)
      max_keys[l] = std::max(max_keys[l], int(std::min<std::int64_t>(P[i], layers_[l].ring.rows)) + BLOCK);
  }
  CUDA_CHECK(cudaMemcpyAsync(d_params_, host_params_.data(), host_params_.size() * 4, cudaMemcpyHostToDevice, stream_));
  const kbd::DraftSeqs S{d_params_, d_params_ + max_seqs_, d_params_ + 2 * max_seqs_};
  const int M = n * BLOCK, MH = n * GAMMA, I = inter_;

  kbd::embed_block(n, S, static_cast<const __nv_bfloat16*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                   mask_embed_, layers_[0].in_norm, f_rnn_, res_, xn_.x, xn_.mx, f_in_, stream_);
  for (int l = 0; l < 3; ++l) {
    const Layer& L = layers_[l];
    linear(xn_, L.qkv, qkv_, kd::QKV_ROWS, M, kd::QKV_ROWS, H);
    kbd::qkv_rope(qkv_, n, S, L.q_norm, L.k_norm, q_, kb_, vb_, L.ring.fp8, stream_);
    kbd::block_attention(q_, n, S, L.ring, kb_, vb_, max_keys[l], attn_ws_, attn_o_.x, attn_o_.mx, stream_);
    linear(attn_o_, L.wo, delta_, H, M, H, kd::Q_DIM);
    kbd::add_rmsnorm(res_, delta_, L.post_norm, xn_.x, xn_.mx, M, nullptr, {}, stream_);
    linear(xn_, L.gate_up, gu_, 2 * I, M, 2 * I, H);
    kbd::silu_mul(gu_, M, I, act_.x, act_.mx, stream_);
    linear(act_, L.down, delta_, H, M, H, I);
    if (l < 2) kbd::add_rmsnorm(res_, delta_, layers_[l + 1].in_norm, xn_.x, xn_.mx, M, nullptr, {}, stream_);
    else kbd::add_rmsnorm(res_, delta_, final_norm_, nullptr, {}, M, h_, head_in_.mx, stream_);
  }
  // Markov head over the rows' shortlist logits: candidates, the projection's h and f_rnn parts, 7 steps
  linear(head_in_, short_head_, base_, short_ld_, MH, short_ld_, H);
  kbd::topk_candidates(base_, MH, short_rows_, short_ld_, cand_val_, cand_idx_, stream_);
  gemm(h_, H, rnn_w_ + 2 * kd::RANK, kd::RNN_IN, part_h_, kd::RNN_OUT, true, MH, kd::RNN_OUT, H);
  gemm(f_in_, H, rnn_w_ + 2 * kd::RANK + H, kd::RNN_IN, part_f_, kd::RNN_OUT, true, n, kd::RNN_OUT, H);
  kbd::markov_draft(n, S, rnn_w_, rnn_b_, w1_, part_h_, part_f_, cand_val_, cand_idx_, short_map_, fold_w_, fold_b_,
                    temperature, seed, tokens_, conf_, stream_);
  return {tokens_, conf_};
}

}  // namespace ingine::model
