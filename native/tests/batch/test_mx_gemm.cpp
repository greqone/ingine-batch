// MXFP8 copies (kernels::bdraft::mx_quantize) and the block-scaled GEMM (kernels::bdraft::MxGemm): the quantized
// values decode (E4M3 x 2^(E8M0 - 127), scales at mx_sf_index) to within E4M3 rounding of the inputs, and the GEMM
// equals the FP64 product of the decoded operands up to FP32 accumulation and the BF16 output rounding. Blocks of
// 32 get random power-of-two magnitudes, so a misplaced scale shows.
#include "common/test_util.h"
#include "kernels/batch/draft.h"
#include "kernels/batch/mx_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;
namespace kbd = ingine::kernels::bdraft;

namespace {

int g_failures = 0;

std::vector<__nv_bfloat16> random_rows(Rng& r, int rows, int K) {
  std::vector<__nv_bfloat16> x(std::size_t(rows) * K);
  for (int i = 0; i < rows; ++i)
    for (int b = 0; b < K / 32; ++b) {
      const double mag = std::ldexp(1.0, int(r.next() % 17) - 8);
      for (int k = 0; k < 32; ++k) x[std::size_t(i) * K + 32 * b + k] = float_to_bf16(float(r.normal() * mag));
    }
  return x;
}

// decoded Mx copy, and whether every element is within E4M3 rounding (half an ulp of 3 mantissa bits, relative
// to the block's scale) of x
std::vector<double> decode(const DeviceArray<uint8_t>& q, const DeviceArray<uint8_t>& sf, const std::vector<__nv_bfloat16>& x,
                           int rows, int K, bool& close) {
  const std::vector<uint8_t> hq = q.download(), hs = sf.download();
  std::vector<double> out(std::size_t(rows) * K);
  close = true;
  for (int i = 0; i < rows; ++i)
    for (int k = 0; k < K; ++k) {
      const double scale = std::ldexp(1.0, int(hs[kbd::mx_sf_index(i, k / 32, K)]) - 127);
      const double v = double(e4m3_to_float(hq[std::size_t(i) * K + k])) * scale;
      const double ref = double(bf16_to_float(x[std::size_t(i) * K + k]));
      out[std::size_t(i) * K + k] = v;
      // |v| <= 448 scale; E4M3 normals have 3 mantissa bits (relative step 2^-3), subnormals a step of 2^-9 scale
      if (std::fabs(v - ref) > std::fabs(ref) * 0.0625 + 0.001953125 * scale * 1.0001) close = false;
    }
  return out;
}

void run(int M, int N, int K, cudaStream_t stream) {
  Rng r(std::uint64_t(M) * 1000003u + N + K);
  const std::vector<__nv_bfloat16> a = random_rows(r, M, K), w = random_rows(r, N, K);
  DeviceArray<__nv_bfloat16> da(a), dw(w);
  DeviceArray<uint8_t> qa(std::size_t(M) * K), sa(kbd::mx_sf_bytes(M, K)), qw(std::size_t(N) * K),
      sw(kbd::mx_sf_bytes(N, K));
  DeviceArray<__nv_bfloat16> c(std::size_t(M) * N);
  kbd::mx_quantize(da.get(), M, K, {qa.get(), sa.get()}, stream);
  kbd::mx_quantize(dw.get(), N, K, {qw.get(), sw.get()}, stream);
  kbd::MxGemm gemm;
  gemm.run(qa.get(), sa.get(), qw.get(), sw.get(), c.get(), N, M, N, K, stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  bool close_a = false, close_w = false;
  const std::vector<double> ra = decode(qa, sa, a, M, K, close_a), rw = decode(qw, sw, w, N, K, close_w);
  char what[128];
  std::snprintf(what, sizeof what, "M %d N %d K %d: MX copies within E4M3 rounding", M, N, K);
  std::printf("%s %s\n", close_a && close_w ? "PASS" : "FAIL", what);
  if (!(close_a && close_w)) ++g_failures;
  std::vector<double> ref(std::size_t(M) * N), mag(std::size_t(M) * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0, t = 0;
      for (int k = 0; k < K; ++k) {
        const double p = ra[std::size_t(i) * K + k] * rw[std::size_t(j) * K + k];
        s += p;
        t += std::fabs(p);
      }
      ref[std::size_t(i) * N + j] = s;
      mag[std::size_t(i) * N + j] = t;
    }
  const std::vector<__nv_bfloat16> hc = c.download();
  // each element within two BF16 ulps (2^-7 relative: the block-scaled kernels round their output to within one
  // ulp, not to nearest) plus FP32 accumulation (well below 1e-5 of sum |terms|); misplaced scales err by about
  // the element itself
  double worst = 0;
  for (std::size_t i = 0; i < hc.size(); ++i) {
    const double err = std::fabs(double(bf16_to_float(hc[i])) - ref[i]);
    worst = std::max(worst, err / (std::fabs(ref[i]) * 0.0078125 + 1e-5 * mag[i] + 1e-30));
  }
  std::snprintf(what, sizeof what, "M %d N %d K %d: MxGemm vs FP64 of the decoded operands", M, N, K);
  std::printf("%s %s: worst error / bound %.3f\n", worst <= 1.0 ? "PASS" : "FAIL", what, worst);
  if (!(worst <= 1.0)) ++g_failures;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  run(40, 200, 2304, stream);
  run(7, 64, 3072, stream);
  run(136, 96, 9216, stream);
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
