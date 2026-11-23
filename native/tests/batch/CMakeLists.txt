# Batched decoding (model/batch.h, model/batch_drafter.h).
# The MXFP8 copies and the block-scaled GEMM of the batched drafter against FP64 (no model).
ingine_add_test(NAME batch_mx_gemm SOURCES test_mx_gemm.cpp LIBS ingine_kernels_batch CUDA::cudart GPU ORACLE
                TIMEOUT 300)
# BatchDrafter against the Drafter with the real mtp.safetensors and a synthetic 4096-token target (~4 GB).
if(TARGET ingine_model)
  ingine_add_test(NAME batch_drafter SOURCES test_batch_drafter.cpp LIBS ingine_model CUDA::cudart GPU MODEL
                  TIMEOUT 600)
endif()
