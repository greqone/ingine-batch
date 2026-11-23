#include "kernels/batch/dense.h"

#include <stdexcept>
#include <string>

namespace ingine::kernels::batch {
namespace {

constexpr int H = prefill::kHidden;
constexpr int ROW_T = H / 4;  // 576 threads, float4 each
constexpr int ROW_W = ROW_T / 32;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
  return std::uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (std::uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// ---- BF16 GEMM: C[M, N] = A[M, K] . B[N, K]^T
//
// Warp tile (MF * 16) x (NF * 8), WM x WN warps. Both operands are K-contiguous rows; a stage holds TM rows of A and
// TN rows of B, 64 elements (128 bytes = 8 chunks of 16 bytes, whole lines from DRAM) each, chunk c of row r at
// chunk c ^ (r & 7) so the 8 rows of an ldmatrix hit distinct banks. cp.async in STAGES stages (rows past M / N
// zero-filled), one barrier per stage, four k16 MMA steps per stage. The row tiles of A are the fastest grid
// dimension, so the CTAs that share B's rows run together and B streams from DRAM once.

constexpr int BK = 64;

struct MmaArgs {
  const __nv_bfloat16* A;
  const __nv_bfloat16* B;
  void* C;
  int lda, ldb, ldc, M, N, K;
  int c_fp32;
};

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int swz(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int MF_, int NF_, int WM_, int WN_, int STAGES_>
struct MmaCfg {
  static constexpr int MF = MF_, NF = NF_, WM = WM_, WN = WN_, STAGES = STAGES_;
  static constexpr int TM = WM * MF * 16, TN = WN * NF * 8, NT = WM * WN * 32;
  static constexpr int A_BYTES = TM * BK * 2, STAGE = (TM + TN) * BK * 2, SMEM = STAGES * STAGE;
  static_assert(NF % 2 == 0 && SMEM <= 96 * 1024);
};

template <class C>
__global__ void __launch_bounds__(C::NT) mma_gemm_kernel(const MmaArgs p) {
  extern __shared__ __align__(128) std::uint8_t smem[];
  constexpr int MF = C::MF, NF = C::NF, TM = C::TM, TN = C::TN, NT = C::NT, STAGES = C::STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warp_m = warp / C::WN, warp_n = warp % C::WN;
  const int mtile = blockIdx.x, ntile = blockIdx.y;
  const __nv_bfloat16* A = p.A + std::size_t(mtile) * TM * p.lda;
  const __nv_bfloat16* B = p.B + std::size_t(ntile) * TN * p.ldb;
  const int m_rows = min(TM, p.M - mtile * TM), n_rows = min(TN, p.N - ntile * TN);
  const int nk = p.K / BK;

  auto load_a = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE;
    for (int i = tid; i < TM * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < m_rows;
      cp_async16(smem_addr(base + swz(r, c)), A + std::size_t(ok ? r : 0) * p.lda + kt * BK + c * 8, ok);
    }
  };
  auto load_b = [&](int stage, int kt) {
    std::uint8_t* base = smem + stage * C::STAGE + C::A_BYTES;
    for (int i = tid; i < TN * 8; i += NT) {
      const int r = i >> 3, c = i & 7;
      const bool ok = r < n_rows;
      cp_async16(smem_addr(base + swz(r, c)), B + std::size_t(ok ? r : 0) * p.ldb + kt * BK + c * 8, ok);
    }
  };

  float acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // the weights do not depend on the previous kernel (programmatic dependent launch): their first stages load
  // before waiting for it
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s)
    if (s < nk) load_b(s, s);
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_a(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < nk; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < nk) load_a(next % STAGES, next), load_b(next % STAGES, next);
    cp_commit();
    const std::uint32_t a_base = smem_addr(smem + (kt % STAGES) * C::STAGE), b_base = a_base + C::A_BYTES;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      std::uint32_t af[MF][4], bf[NF][2];
#pragma unroll
      for (int i = 0; i < MF; ++i)
        ldsm_x4(af[i], a_base + swz(warp_m * MF * 16 + i * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < NF / 2; ++j) {
        std::uint32_t r[4];
        ldsm_x4(r, b_base + swz(warp_n * NF * 8 + j * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1)));
        bf[2 * j][0] = r[0], bf[2 * j][1] = r[1], bf[2 * j + 1][0] = r[2], bf[2 * j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) mma_bf16(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
  }
  cp_wait<0>();
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");

  // thread holds rows lane / 4 (+ 8), columns 2 (lane % 4) (+ 1) of each m16 x n8 block
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const int col = warp_n * NF * 8 + j * 8 + (lane & 3) * 2;
      if (col >= n_rows) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = warp_m * MF * 16 + i * 16 + (lane >> 2) + h * 8;
        if (row >= m_rows) continue;
        const std::size_t off = std::size_t(mtile * TM + row) * p.ldc + ntile * TN + col;
        if (p.c_fp32)
          *reinterpret_cast<float2*>(static_cast<float*>(p.C) + off) = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        else
          *reinterpret_cast<std::uint32_t*>(static_cast<__nv_bfloat16*>(p.C) + off) =
              pack_bf16x2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
      }
    }
}

// configs (cfg order): 64 x 128 (8 warps), 64 x 64 (4 warps), 32 x 128 (8 warps); two stages, so that two CTAs share
// an SM (one computes while the other waits)
using Cfg0 = MmaCfg<2, 4, 2, 4, 2>;
using Cfg1 = MmaCfg<2, 4, 2, 2, 2>;
using Cfg2 = MmaCfg<1, 4, 2, 4, 2>;

template <class C>
void launch(const MmaArgs& a, cudaStream_t stream) {
  static const bool attr = [] {
    return cudaFuncSetAttribute(mma_gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) ==
           cudaSuccess;
  }();
  if (!attr) throw std::runtime_error("batch mma_gemm: shared memory attribute");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((a.M + C::TM - 1) / C::TM, (a.N + C::TN - 1) / C::TN);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute pdl;
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &pdl;
  cfg.numAttrs = 1;
  if (cudaLaunchKernelEx(&cfg, mma_gemm_kernel<C>, a) != cudaSuccess)
    throw std::runtime_error("batch mma_gemm: launch failed");
}

// prefill gate_apply_kernel (same operations), gate blockIdx.y, f / g rows at fg + (g * T + t) * 2H (+ H); the rows
// indexed in parameter space (no local copy)
__global__ void __launch_bounds__(ROW_T)
    gate_apply_kernel(const __grid_constant__ prefill::GateRows rows, const float* __restrict__ fg, int T,
                      const float* __restrict__ norm_w, std::uint16_t* __restrict__ out, std::size_t out_stride) {
  const int t = blockIdx.x, g = blockIdx.y, c = threadIdx.x * 4;
  const std::size_t off = std::size_t(t) * H + c;
  const float* f = fg + (std::size_t(g) * T + t) * 2 * H + c;
  const float4 h4 = *reinterpret_cast<const float4*>(rows.h[g] + off);
  const float4 m4 = *reinterpret_cast<const float4*>(rows.msg[g] + off);
  const float4 f4 = *reinterpret_cast<const float4*>(f), g4 = *reinterpret_cast<const float4*>(f + H);
  const float hv[4] = {h4.x, h4.y, h4.z, h4.w}, mv[4] = {m4.x, m4.y, m4.z, m4.w};
  const float fv[4] = {f4.x, f4.y, f4.z, f4.w}, gv[4] = {g4.x, g4.y, g4.z, g4.w};
  float y[4];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {  // (1 - forget) * hidden + gain * message
    y[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(fv[i])), hv[i]), __fmul_rn(tanhf(gv[i]), mv[i]));
    ss += y[i] * y[i];
  }
  // block sum as prefill's block_sum: warp sums, then the warps' sums in order
  __shared__ float red[ROW_W];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  ss = 0.f;
  for (int w = 0; w < ROW_W; ++w) ss += red[w];
  const float r = rsqrtf(ss * (1.f / H) + kEps);
  const float4 w4 = *reinterpret_cast<const float4*>(norm_w + std::size_t(g) * H + c);
  *reinterpret_cast<uint2*>(out + g * out_stride + off) =
      make_uint2(pack_bf16x2(y[0] * r * w4.x, y[1] * r * w4.y), pack_bf16x2(y[2] * r * w4.z, y[3] * r * w4.w));
}

}  // namespace

void mma_gemm(int cfg, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32,
              int M, int N, int K, cudaStream_t stream) {
  const MmaArgs a{A, B, C, lda, ldb, ldc, M, N, K, c_fp32 ? 1 : 0};
  switch (cfg) {
    case 0: launch<Cfg0>(a, stream); break;
    case 1: launch<Cfg1>(a, stream); break;
    case 2: launch<Cfg2>(a, stream); break;
    default: throw std::invalid_argument("batch mma_gemm: config " + std::to_string(cfg));
  }
}

void gate_apply(const prefill::GateRows& rows, int G, const float* fg, int T, const float* norm_w, __nv_bfloat16* out,
                std::size_t out_stride, cudaStream_t stream) {
  gate_apply_kernel<<<dim3(T, G), ROW_T, 0, stream>>>(rows, fg, T, norm_w, reinterpret_cast<std::uint16_t*>(out),
                                                       out_stride);
}

}  // namespace ingine::kernels::batch
