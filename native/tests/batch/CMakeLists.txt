# Batched decoding (model/batch.h, model/batch_drafter.h) and its kernels.
# The MXFP8 copies and the block-scaled GEMM of the batched drafter against FP64 (no model).
ingine_add_test(NAME batch_mx_gemm SOURCES test_mx_gemm.cpp LIBS ingine_kernels_batch CUDA::cudart GPU ORACLE
                TIMEOUT 300)
# Batched-step expert kernels against their prefill counterparts.
ingine_add_test(NAME batch_moe SOURCES test_moe.cu LIBS ingine_kernels_batch ingine_kernels_prefill GPU TIMEOUT 600)
# The head with the token choice fused against sample over its logits (greedy, sampled, ties).
ingine_add_test(NAME batch_head_sample SOURCES test_head_sample.cu LIBS ingine_kernels_batch GPU TIMEOUT 300)

if(TARGET ingine_model)
  # BatchDrafter against the Drafter with the real mtp.safetensors and a synthetic 4096-token target (~4 GB).
  ingine_add_test(NAME batch_drafter SOURCES test_batch_drafter.cpp LIBS ingine_model CUDA::cudart GPU MODEL
                  TIMEOUT 600)
  # Real-layer timings (old vs new expert GEMMs, T = 64..512); exits 77 (skip) unless INGINE_BENCH=1.
  ingine_add_test(NAME batch_moe_bench SOURCES bench_moe.cu LIBS ingine_model GPU MODEL LABELS bench TIMEOUT 900)
endif()
