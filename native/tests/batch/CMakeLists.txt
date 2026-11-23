# Batched-step kernels against their prefill counterparts.
ingine_add_test(NAME batch_moe SOURCES test_moe.cu LIBS ingine_kernels_batch ingine_kernels_prefill GPU TIMEOUT 600)

# Real-layer timings (old vs new expert GEMMs, T = 64..512); exits 77 (skip) unless INGINE_BENCH=1.
if(TARGET ingine_model)
  ingine_add_test(NAME batch_moe_bench SOURCES bench_moe.cu LIBS ingine_model GPU MODEL LABELS bench TIMEOUT 900)
endif()
