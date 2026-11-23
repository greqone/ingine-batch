add_library(ingine_kernels_batch STATIC
  batch.cu
  draft_attention.cu
  draft_markov.cu
  draft_rows.cu
  mx_gemm.cpp
)
ingine_cuda_target(ingine_kernels_batch)
ingine_internal_includes(ingine_kernels_batch)
target_include_directories(ingine_kernels_batch PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_batch PUBLIC ingine_core CUDA::cudart CUDA::cublasLt)
