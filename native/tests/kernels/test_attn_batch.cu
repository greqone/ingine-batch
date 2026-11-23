// Batched decode attention (attn_decode with batch > 1, src/kernels/attn/attn_batch.cu) against the cluster
// kernel on the same batch (attn_decode_cluster, which test_attn checks against an FP64 oracle). Random
// inputs; batches of 2, 7 and 64 sequences on a permuted slot table, M = 1, 2, 4, 8, per-sequence positions
// from {0, 1, 37, 600, 1500, 5000} (+ an offset for later sequences): 1024-row sliding rings (empty, partial,
// wrapped, with stale speculative entries), 640-row rings that never wrap (positions past them: no-op), and
// full BF16 / E4M3 caches (rows past each position hold random bytes). Output within two BF16 ulps
// (accumulation order differs); caches, kpos and conv states bit-identical. Ragged rows (DecodeArgs::rows):
// batches of 7 and 64 with random row counts in 1..M (M = 2, 4, 8) against the cluster kernel run on each
// sequence alone with M = its count.
// `--bench`: kernel time per call at batch 64 (old vs new), sliding and full BF16 layers, M = 1, 4, 8,
// contexts 64, 300, 640 and 2048, the L2 flushed before every call. `--bench-ragged`: uniform M = 3 and 5
// against ragged rows in 1..5 at contexts 300 and 640. `--bench-prod [case]`: the production regime (below).
#include "kernels/oracle.h"

#include "kernels/attn/attn.h"
#include "kernels/attn/attn_batch.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace ingine::test;
namespace attn = ingine::kernels::attn;

namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, C = HKV * D;
constexpr int OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_KDYN = 8600;
cudaStream_t g_stream = nullptr;
int g_bench_call = 1;  // the benches' batched call: 1 = fp32 proj, 2 = bf16 proj16 (--proj16)

__device__ std::uint64_t mix64(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// cache rows [n_rows][C]: N(0, 1) values as BF16 or E4M3, or random bytes where garbage[row]
__global__ void fill_kv(std::uint8_t* buf, std::size_t n_rows, int fp8, std::uint64_t seed, const std::uint8_t* garbage) {
  const std::size_t n = n_rows * C;
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
    const std::uint64_t h = mix64(seed + 0x9E3779B97F4A7C15ull * (i + 1));
    if (garbage[i / C]) {
      if (fp8) buf[i] = std::uint8_t(h);
      else reinterpret_cast<std::uint16_t*>(buf)[i] = std::uint16_t(h);
      continue;
    }
    const float u1 = float((h >> 40) + 1) * (1.f / 16777217.f), u2 = float(h & 0xffffff) * (1.f / 16777216.f);
    const float x = sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
    if (fp8) buf[i] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    else reinterpret_cast<__nv_bfloat16*>(buf)[i] = __float2bfloat16_rn(x);
  }
}

// reads n floats (clean L2 lines: evicts everything else without write-backs)
__global__ void read_all(const float* p, std::size_t n, float* sink) {
  float s = 0.f;
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
    s += __ldcg(p + i);
  if (s == 1234.5f) *sink = s;
}

enum class Kind { ring, small_ring, full_bf16, full_fp8 };

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::ring: return "sliding ring 1024";
    case Kind::small_ring: return "sliding ring 640";
    case Kind::full_bf16: return "full bf16";
    default: return "full fp8";
  }
}

// One batch: inputs, both kernels' state copies and arguments.
struct Batch {
  int B = 0, M = 0, nslots = 0, rows = 0, R = 0, T = 0;  // T: proj / out rows
  bool full = false, fp8 = false;
  std::vector<std::int64_t> pos;
  std::vector<std::int32_t> slots;
  std::vector<std::int32_t> mb, row0;  // ragged: rows per sequence, row offsets [B + 1] (empty: M each)
  std::vector<float> proj, q_scale, r_scale, rel_proj, kconv, vconv, ck0, cv0;
  std::vector<std::int64_t> kpos0;  // [nslots][rows]
  std::optional<DeviceArray<float>> dproj, dqs, drs, drp, dkc, dvc;
  std::optional<DeviceArray<__nv_bfloat16>> dproj16;  // the same projection rows as bf16
  std::optional<DeviceArray<std::int64_t>> dpos;
  std::optional<DeviceArray<std::int32_t>> dslots, drows;
  // per call (0 = cluster reference, 1 = batched, 2 = batched from bf16 proj16, 3 = batched with conv_write
  // false), each on its own copy of the state
  static constexpr int NV = 4;
  std::optional<DeviceArray<std::uint8_t>> k[NV], v[NV], ws[NV];
  std::optional<DeviceArray<float>> ck[NV], cv[NV];
  std::optional<DeviceArray<std::int64_t>> kp[NV];
  std::optional<DeviceArray<__nv_bfloat16>> out[NV];

  attn::DecodeArgs args(int i) const {
    attn::DecodeArgs a;
    if (i == 2) a.proj16 = dproj16->get();
    else a.proj = dproj->get();
    a.M = M;
    a.q_scale = dqs->get();
    a.r_scale = drs->get();
    a.rel_proj = drp->get();
    a.k_conv = dkc->get();
    a.v_conv = dvc->get();
    a.conv_k_state = ck[i]->get();
    a.conv_v_state = cv[i]->get();
    a.k_cache = k[i]->get();
    a.v_cache = v[i]->get();
    a.cache_fp8 = fp8;
    a.cache_rows = rows;
    a.kpos = full ? nullptr : kp[i]->get();
    a.pos = dpos->get();
    a.window = full ? 0 : WIN;
    a.log_scaling = full;
    a.workspace = ws[i]->get();
    a.workspace_size = ws[i]->count;
    a.out = out[i]->get();
    a.batch = B;
    a.slots = dslots->get();
    a.cache_stride = std::int64_t(rows) * C;
    a.kpos_stride = rows;
    a.conv_stride = 3 * C;
    if (i >= 1 && drows) a.rows = drows->get();
    if (i == 3) a.conv_write = false;
    return a;
  }

  // the reference for ragged rows: the cluster kernel on each sequence alone, with M = its row count
  void reference_per_sequence(cudaStream_t s) const {
    const std::size_t esize = fp8 ? 1 : 2;
    for (int b = 0; b < B; ++b) {
      attn::DecodeArgs a = args(0);
      const std::size_t sl = std::size_t(slots[b]);
      a.M = mb[b];
      a.proj += std::size_t(row0[b]) * R;
      a.out += std::size_t(row0[b]) * HQ * D;
      a.pos += b;
      a.k_cache = static_cast<std::uint8_t*>(a.k_cache) + sl * rows * C * esize;
      a.v_cache = static_cast<std::uint8_t*>(a.v_cache) + sl * rows * C * esize;
      a.conv_k_state += sl * 3 * C;
      a.conv_v_state += sl * 3 * C;
      if (a.kpos) a.kpos += sl * rows;
      a.batch = 1, a.slots = nullptr;
      attn::attn_decode_cluster(a, s);
    }
  }
};

// ring slot positions before the call: in order up to p0 (wrapped once p0 > rows), plus `stale` entries at
// p0 .. p0 + stale - 1 left by a rejected speculative call
std::vector<std::int64_t> ring_kpos(int rows, std::int64_t p0, int stale) {
  std::vector<std::int64_t> kp(rows, -1);
  for (std::int64_t q = std::max<std::int64_t>(0, p0 - rows); q < p0; ++q) kp[q % rows] = q;
  for (int i = 0; i < stale; ++i)
    if (p0 + i < rows || rows == WIN) kp[(p0 + i) % rows] = p0 + i;
  return kp;
}

// a batch of pos.size() sequences at positions pos with M rows each, or ms[b] rows (ragged, max M); small
// rings hold small_rows rows
void make_batch(Batch& bt, Rng& rng, Kind kind, int M, const std::vector<std::int64_t>& pos, int small_rows = 640,
                const std::vector<std::int32_t>& ms = {}) {
  const int B = int(pos.size());
  bt.B = B, bt.M = M, bt.nslots = B + 3;
  bt.mb = ms, bt.row0.assign(1, 0);
  for (int b = 0; b < B; ++b) bt.row0.push_back(bt.row0.back() + (ms.empty() ? M : ms[b]));
  bt.T = bt.row0.back();
  bt.full = kind == Kind::full_bf16 || kind == Kind::full_fp8;
  bt.fp8 = kind == Kind::full_fp8;
  bt.R = bt.full ? 8664 : 8600;
  bt.pos = pos;
  const std::int64_t pmax = *std::max_element(bt.pos.begin(), bt.pos.end());
  bt.rows = kind == Kind::ring ? WIN : kind == Kind::small_ring ? small_rows : int(pmax) + 24;
  bt.slots.resize(B);
  for (int b = 0; b < B; ++b) bt.slots[b] = (b * 7 + 2) % bt.nslots;  // gcd(7, B + 3) = 1: distinct
  // inputs (as test_attn)
  bt.proj.resize(std::size_t(bt.T) * bt.R);
  for (std::size_t i = 0; i < bt.proj.size(); ++i) {
    const int r = int(i % bt.R);
    double s = 1.0;
    if (r >= OFF_REL && r < OFF_SINK) s = 3.0;
    else if (r >= OFF_SINK && r < OFF_K) s = 2.0;
    else if (r >= OFF_KDYN) s = 0.3;
    bt.proj[i] = round_bf16(float(rng.normal() * s));
  }
  bt.q_scale = randn(rng, HQ * D, 0.3), bt.r_scale = randn(rng, HQ * DREL, 0.3);
  for (auto& v : bt.q_scale) v = round_bf16(v + 1.f);
  for (auto& v : bt.r_scale) v = round_bf16(v + 1.f);
  bt.rel_proj = randn_bf16(rng, DREL * EXT, 0.5);
  bt.kconv = randn_bf16(rng, 4 * C, 0.3), bt.vconv = randn_bf16(rng, 4 * C, 0.3);
  bt.ck0 = randn_bf16(rng, std::size_t(bt.nslots) * 3 * C, 1.0), bt.cv0 = randn_bf16(rng, std::size_t(bt.nslots) * 3 * C, 1.0);
  // slot state: ring positions (stale entries on every third sequence), garbage rows
  bt.kpos0.assign(std::size_t(bt.nslots) * bt.rows, -1);
  std::vector<std::uint8_t> garbage(std::size_t(bt.nslots) * bt.rows, 1);
  for (int b = 0; b < B; ++b) {
    const int sl = bt.slots[b];
    if (bt.full) {
      for (int r = 0; r < std::min<std::int64_t>(bt.pos[b], bt.rows); ++r) garbage[std::size_t(sl) * bt.rows + r] = 0;
    } else {
      const auto kp = ring_kpos(bt.rows, bt.pos[b], b % 3 == 2 ? 6 : 0);
      for (int r = 0; r < bt.rows; ++r) {
        bt.kpos0[std::size_t(sl) * bt.rows + r] = kp[r];
        garbage[std::size_t(sl) * bt.rows + r] = kp[r] < 0;
      }
    }
  }
  bt.dproj.emplace(bt.proj), bt.dqs.emplace(bt.q_scale), bt.drs.emplace(bt.r_scale), bt.drp.emplace(bt.rel_proj);
  bt.dproj16.emplace(to_bf16(bt.proj));
  bt.dkc.emplace(bt.kconv), bt.dvc.emplace(bt.vconv);
  bt.dpos.emplace(bt.pos), bt.dslots.emplace(bt.slots);
  if (!ms.empty()) bt.drows.emplace(bt.row0);
  DeviceArray<std::uint8_t> dg(garbage);
  const std::size_t n_rows = std::size_t(bt.nslots) * bt.rows, bytes = n_rows * C * (bt.fp8 ? 1 : 2);
  const std::uint64_t seed = rng.next();
  for (int i = 0; i < Batch::NV; ++i) {
    bt.k[i].emplace(bytes), bt.v[i].emplace(bytes);
    fill_kv<<<1024, 256>>>(bt.k[i]->get(), n_rows, bt.fp8, seed, dg.get());
    fill_kv<<<1024, 256>>>(bt.v[i]->get(), n_rows, bt.fp8, seed ^ 0x5555, dg.get());
    INGINE_TEST_CUDA(cudaGetLastError());
    bt.ck[i].emplace(bt.ck0), bt.cv[i].emplace(bt.cv0);
    bt.kp[i].emplace(bt.kpos0);
    // the batched call gets a workspace sized for any width and a larger batch (as a decoder allocates it)
    bt.ws[i].emplace(i ? attn::workspace_bytes(8, B + 1) : attn::cluster_workspace_bytes(M, B));
    bt.out[i].emplace(std::size_t(bt.T) * HQ * D);
  }
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
}

template <class T>
std::size_t count_diff(const DeviceArray<T>& a, const DeviceArray<T>& b) {
  const auto x = a.download(), y = b.download();
  std::size_t n = 0;
  for (std::size_t i = 0; i < x.size(); ++i) n += std::memcmp(&x[i], &y[i], sizeof(T)) != 0;
  return n;
}

struct Worst {
  double abs = 0, rel = 0;
};

// ragged: every sequence gets a random row count in 1..M (one of them M), checked against the cluster kernel
// on each sequence alone with M = its count
bool run(Rng& rng, Kind kind, int B, int M, int phase, Worst& worst, bool ragged = false) {
  static const std::vector<std::int64_t> positions = {0, 1, 37, 600, 1500, 5000};
  std::vector<std::int64_t> pos(B);
  for (int b = 0; b < B; ++b) pos[b] = positions[(b + phase) % positions.size()] + b / int(positions.size());
  std::vector<std::int32_t> ms;
  if (ragged) {
    for (int b = 0; b < B; ++b) ms.push_back(1 + int(rng.next() % std::uint64_t(M)));
    ms[rng.next() % std::uint64_t(B)] = M;
  }
  Batch bt;
  make_batch(bt, rng, kind, M, pos, 640, ms);
  int active = 0;  // sequences within capacity
  for (int b = 0; b < B; ++b) active += kind != Kind::small_ring || pos[b] + (ragged ? ms[b] : M) <= bt.rows;
  if (ragged) bt.reference_per_sequence(g_stream);
  else attn::attn_decode_cluster(bt.args(0), g_stream);
  for (int i = 1; i < Batch::NV; ++i) attn::attn_decode(bt.args(i), g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  INGINE_TEST_CUDA(cudaGetLastError());
  // output: within two BF16 ulps of the reference (accumulation order differs: rounding flips of bf16(o) and
  // bf16(o * gate)), plus 2^-12 absolute for values that cancel to near zero (|o| is O(1) per term)
  const auto ref = bf16_download(*bt.out[0]), got = bf16_download(*bt.out[1]);
  double max_abs = 0, max_rel = 0;
  std::size_t bad = 0, off = 0, wi = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(double(got[i]) - ref[i]), r = std::fabs(double(ref[i]));
    max_abs = std::max(max_abs, d);
    if (d > 0) ++off;
    if (d > 0 && d / r > max_rel) max_rel = d / r, wi = i;
    if (!(d <= r / 64 + 1.0 / 4096)) ++bad;
  }
  worst.abs = std::max(worst.abs, max_abs), worst.rel = std::max(worst.rel, max_rel);
  const std::size_t dk = count_diff(*bt.k[0], *bt.k[1]), dv = count_diff(*bt.v[0], *bt.v[1]);
  const std::size_t dck = count_diff(*bt.ck[0], *bt.ck[1]), dcv = count_diff(*bt.cv[0], *bt.cv[1]);
  const std::size_t dkp = bt.full ? 0 : count_diff(*bt.kp[0], *bt.kp[1]);
  // the call advanced the conv states, unless every sequence was past its cache
  const std::size_t moved = count_diff(*bt.ck[1], DeviceArray<float>(bt.ck0));
  // bf16 projections: every output and state bit as from fp32; conv_write = false: the same outputs and caches,
  // the conv states untouched
  const std::size_t d16 = count_diff(*bt.out[1], *bt.out[2]) + count_diff(*bt.k[1], *bt.k[2]) +
                          count_diff(*bt.v[1], *bt.v[2]) + count_diff(*bt.ck[1], *bt.ck[2]) +
                          count_diff(*bt.cv[1], *bt.cv[2]) + (bt.full ? 0 : count_diff(*bt.kp[1], *bt.kp[2]));
  const std::size_t dnw = count_diff(*bt.out[1], *bt.out[3]) + count_diff(*bt.k[1], *bt.k[3]) +
                          count_diff(*bt.v[1], *bt.v[3]) + (bt.full ? 0 : count_diff(*bt.kp[1], *bt.kp[3])) +
                          count_diff(*bt.ck[3], DeviceArray<float>(bt.ck0)) +
                          count_diff(*bt.cv[3], DeviceArray<float>(bt.cv0));
  const bool ok = bad == 0 && dk == 0 && dv == 0 && dck == 0 && dcv == 0 && dkp == 0 &&
                  (moved > 0) == (active > 0) && d16 == 0 && dnw == 0;
  std::printf("%s %s B=%d M=%d%s phase=%d splits=%d: out max_abs=%.3e max_rel=%.3e (got %.6g ref %.6g) differ=%zu/%zu "
              "(out of tolerance: %zu); cache k/v bytes differ %zu/%zu, conv k/v %zu/%zu, kpos %zu; proj16 vs fp32 "
              "%zu differ, conv_write=false %zu differ\n",
              ok ? "PASS" : "FAIL", kind_name(kind), B, M, ragged ? " ragged" : "", phase,
              attn::batched_splits(B, bt.rows), max_abs, max_rel, double(got[wi]), double(ref[wi]), off, ref.size(), bad,
              dk, dv, dck, dcv, dkp, d16, dnw);
  return ok;
}

// the projection rows the benched call reads into the L2 (the kernel before attention writes them in a step)
void touch_proj(const Batch& bt, DeviceArray<float>& dummy) {
  if (g_bench_call == 2)
    read_all<<<680, 256, 0, g_stream>>>(reinterpret_cast<const float*>(bt.dproj16->get()), bt.dproj16->count / 2,
                                        dummy.get());
  else read_all<<<680, 256, 0, g_stream>>>(bt.dproj->get(), bt.dproj->count, dummy.get());
}

int bench(int only_m, int only_ctx) {  // 0 = all
  const double bw = 1.79e12;
  // before every call: a 192 MB read (twice the L2), then proj (written by the kernel before attention in a step)
  DeviceArray<float> flush(std::size_t(48) << 20), dummy(1);
  cudaEvent_t e0, e1;
  INGINE_TEST_CUDA(cudaEventCreate(&e0));
  INGINE_TEST_CUDA(cudaEventCreate(&e1));
  Rng rng(7);
  std::printf("kernel time per call, batch 64 (us), %s proj; floor = KV bytes at %.2f TB/s\n",
              g_bench_call == 2 ? "bf16" : "fp32", bw / 1e12);
  for (Kind kind : {Kind::ring, Kind::full_bf16})
    for (int M : {1, 4, 8})
      for (int ctx : {64, 300, 640, 2048}) {
        if ((only_m && M != only_m) || (only_ctx && ctx != only_ctx)) continue;
        Batch bt;
        // sliding rings of min(ctx + 16, 1024) rows, as BatchState for contexts up to ctx + 16
        const Kind k = kind == Kind::ring && ctx + 16 < WIN ? Kind::small_ring : kind;
        make_batch(bt, rng, k, M, std::vector<std::int64_t>(64, ctx), ctx + 16);
        double t[2] = {0, 0};
        constexpr int REPS = 20;
        for (int rep = 0; rep < REPS + 2; ++rep)
          for (int i = 0; i < 2; ++i) {
            read_all<<<680, 256, 0, g_stream>>>(flush.get(), flush.count, dummy.get());
            touch_proj(bt, dummy);
            INGINE_TEST_CUDA(cudaEventRecord(e0, g_stream));
            if (i == 0) attn::attn_decode_cluster(bt.args(0), g_stream);
            else attn::attn_decode(bt.args(g_bench_call), g_stream);
            INGINE_TEST_CUDA(cudaEventRecord(e1, g_stream));
            INGINE_TEST_CUDA(cudaEventSynchronize(e1));
            float ms = 0;
            INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, e0, e1));
            if (rep >= 2) t[i] += ms * 1e3 / REPS;
          }
        const int keys = kind == Kind::ring ? std::min(ctx, WIN) : ctx;
        const double floor_us = 64.0 * keys * 4096 / bw * 1e6;
        std::printf("%-12s M=%d ctx=%5d: old %8.1f  new %8.1f  floor %6.1f  target %6.1f  splits %d\n",
                    kind == Kind::ring ? "sliding" : "full bf16", M, ctx, t[0], t[1], floor_us, 1.3 * floor_us + 10,
                    attn::batched_splits(64, bt.rows));
      }
  return 0;
}

// batched kernel time per call at batch 64: uniform M = 3, 5 and 8 against ragged rows uniformly random in
// 1..5, with the kernel for M = 5 (the step's largest) and for M = 8; contexts 300 and 640
int bench_ragged() {
  DeviceArray<float> flush(std::size_t(48) << 20), dummy(1);
  cudaEvent_t e0, e1;
  INGINE_TEST_CUDA(cudaEventCreate(&e0));
  INGINE_TEST_CUDA(cudaEventCreate(&e1));
  Rng rng(11);
  std::printf("batched kernel time per call, batch 64 (us), %s proj\n", g_bench_call == 2 ? "bf16" : "fp32");
  for (Kind kind : {Kind::ring, Kind::full_bf16})
    for (int ctx : {300, 640}) {
      const Kind k = kind == Kind::ring ? Kind::small_ring : kind;
      std::vector<std::int32_t> ms(64);
      for (auto& m : ms) m = 1 + int(rng.next() % 5);
      constexpr int NC = 5;  // uniform M = 3, 5, 8; ragged with max 5, ragged with max 8
      const int Ms[NC] = {3, 5, 8, 5, 8};
      double t[NC] = {};
      for (int c = 0; c < NC; ++c) {
        Batch bt;
        make_batch(bt, rng, k, Ms[c], std::vector<std::int64_t>(64, ctx), ctx + 16,
                   c >= 3 ? ms : std::vector<std::int32_t>{});
        constexpr int REPS = 20;
        for (int rep = 0; rep < REPS + 2; ++rep) {
          read_all<<<680, 256, 0, g_stream>>>(flush.get(), flush.count, dummy.get());
          touch_proj(bt, dummy);
          INGINE_TEST_CUDA(cudaEventRecord(e0, g_stream));
          attn::attn_decode(bt.args(g_bench_call), g_stream);
          INGINE_TEST_CUDA(cudaEventRecord(e1, g_stream));
          INGINE_TEST_CUDA(cudaEventSynchronize(e1));
          float ms_ = 0;
          INGINE_TEST_CUDA(cudaEventElapsedTime(&ms_, e0, e1));
          if (rep >= 2) t[c] += ms_ * 1e3 / REPS;
        }
      }
      int total = 0;
      for (int m : ms) total += m;
      std::printf("%-9s ctx=%4d: M=3 %6.1f  M=5 %6.1f  M=8 %6.1f  ragged 1..5 (mean %.2f): max-5 kernel %6.1f, "
                  "max-8 kernel %6.1f\n",
                  kind == Kind::ring ? "sliding" : "full bf16", ctx, t[0], t[1], t[2], total / 64.0, t[3], t[4]);
    }
  return 0;
}

// --bench-prod [case]: the batched kernel as a decode step calls it (bf16 proj16, conv_write = false), L2 flushed
// before every call; us per call and GB/s of the K/V rows read (4 KB per sequence and cached key). Batch 64 with
// ragged rows (spec: 3..5 rows, mean 3.7, M = 5; wide: 1..8, mean ~3.75, M = 8) at mean contexts 128..640 (+-48
// per sequence, as a step's sequences differ) on 640-row sliding rings and full BF16 caches, then other regimes:
// batch 1 / 8 / 24 / 44 / 64, uniform M = 1 and 8, long contexts with key splits, a wrapped 1024-row ring.
// `case`: only the case of that index (0-based, in print order).
int bench_prod(int only) {
  const double bw = 1.79e12;
  DeviceArray<float> flush(std::size_t(48) << 20), dummy(1);
  cudaEvent_t e0, e1;
  INGINE_TEST_CUDA(cudaEventCreate(&e0));
  INGINE_TEST_CUDA(cudaEventCreate(&e1));
  Rng rng(17);
  auto time_one = [&](const char* label, Kind kind, int M, const std::vector<std::int64_t>& pos,
                      const std::vector<std::int32_t>& ms, int ring_rows) {
    static int ncase = -1;
    if (++ncase != only && only >= 0) return;
    Batch bt;
    make_batch(bt, rng, kind, M, pos, ring_rows, ms);
    attn::DecodeArgs a = bt.args(2);
    a.conv_write = false;
    constexpr int REPS = 30;
    double t = 0;
    for (int rep = 0; rep < REPS + 3; ++rep) {
      read_all<<<680, 256, 0, g_stream>>>(flush.get(), flush.count, dummy.get());
      read_all<<<680, 256, 0, g_stream>>>(reinterpret_cast<const float*>(bt.dproj16->get()), bt.dproj16->count / 2,
                                          dummy.get());
      INGINE_TEST_CUDA(cudaEventRecord(e0, g_stream));
      attn::attn_decode(a, g_stream);
      INGINE_TEST_CUDA(cudaEventRecord(e1, g_stream));
      INGINE_TEST_CUDA(cudaEventSynchronize(e1));
      float ms_ = 0;
      INGINE_TEST_CUDA(cudaEventElapsedTime(&ms_, e0, e1));
      if (rep >= 3) t += ms_ * 1e3 / REPS;
    }
    double keys = 0, rows = 0, ctx = 0;
    for (int b = 0; b < bt.B; ++b) {
      keys += double(bt.full ? pos[b] : std::min<std::int64_t>(pos[b], WIN));
      rows += ms.empty() ? M : ms[b];
      ctx += double(pos[b]) / bt.B;
    }
    const double gbs = keys * 4096 / (t * 1e-6) / 1e9;
    std::printf("%-26s %-7s M=%d rows/seq %.2f ctx %5.0f: %7.1f us %7.1f GB/s %5.1f%%  splits %d\n", label,
                bt.full ? "full" : "sliding", M, rows / bt.B, ctx, t, gbs, gbs * 1e9 / bw * 100,
                attn::batched_splits(bt.B, bt.rows));
  };
  // batch 64, ragged
  std::vector<std::int32_t> spec(64), wide(64);
  for (int b = 0; b < 64; ++b) {
    const int u = int(rng.next() % 20);
    spec[b] = u < 9 ? 3 : u < 17 ? 4 : 5;  // 45 / 40 / 15 %
    wide[b] = rng.next() % 2 ? 1 + int(rng.next() % 8) : 1 + int(rng.next() % 5);
  }
  spec[5] = 5, wide[5] = 8;
  for (Kind kind : {Kind::small_ring, Kind::full_bf16})
    for (int ctx : {128, 256, 384, 512, 640}) {
      std::vector<std::int64_t> pos(64);
      for (int b = 0; b < 64; ++b) pos[b] = ctx - 48 + int((std::uint32_t(b + 1) * 2654435761u >> 13) % 97);
      time_one("batch 64 ragged spec 3..5", kind, 5, pos, spec, std::max(640, ctx + 64));
      time_one("batch 64 ragged wide 1..8", kind, 8, pos, wide, std::max(640, ctx + 64));
    }
  // other regimes (uniform rows)
  struct Reg {
    int B, M, ctx;
    Kind kind;
  };
  const Reg regs[] = {
      {64, 1, 384, Kind::small_ring}, {64, 8, 384, Kind::small_ring}, {64, 1, 640, Kind::full_bf16},
      {64, 8, 640, Kind::full_bf16},  {64, 4, 2048, Kind::full_bf16}, {64, 4, 3000, Kind::ring},
      {44, 4, 560, Kind::full_bf16},  {24, 4, 560, Kind::small_ring}, {8, 1, 640, Kind::small_ring},
      {8, 8, 640, Kind::full_bf16},   {8, 4, 4096, Kind::full_bf16},  {8, 8, 4096, Kind::ring},
      {1, 1, 640, Kind::small_ring},  {1, 8, 640, Kind::full_bf16},   {1, 1, 4096, Kind::full_bf16},
      {1, 8, 4096, Kind::full_bf16},  {1, 8, 4096, Kind::ring},
  };
  for (const Reg& r : regs) {
    char label[64];
    std::snprintf(label, sizeof label, "batch %d uniform", r.B);
    time_one(label, r.kind, r.M, std::vector<std::int64_t>(r.B, r.ctx), {}, std::max(640, r.ctx + 48));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--proj16") g_bench_call = 2;
  if (argc > 1 && std::string(argv[1]) == "--bench")  // --bench [M [ctx]]
    return bench(argc > 2 ? std::atoi(argv[2]) : 0, argc > 3 ? std::atoi(argv[3]) : 0);
  if (argc > 1 && std::string(argv[1]) == "--bench-ragged") return bench_ragged();
  if (argc > 1 && std::string(argv[1]) == "--bench-prod") return bench_prod(argc > 2 ? std::atoi(argv[2]) : -1);
  Rng rng(9090);
  bool ok = true;
  Worst worst;
  int phase = 0;
  for (Kind kind : {Kind::ring, Kind::small_ring, Kind::full_bf16, Kind::full_fp8})
    for (int B : {2, 7, 64})
      for (int M : {1, 2, 4, 8}) ok &= run(rng, kind, B, M, phase++ % 6, worst);
  for (Kind kind : {Kind::ring, Kind::small_ring, Kind::full_bf16, Kind::full_fp8})
    for (int B : {7, 64})
      for (int M : {2, 4, 8}) ok &= run(rng, kind, B, M, phase++ % 6, worst, true);
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("worst output difference: abs %.3e rel %.3e\n", worst.abs, worst.rel);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
