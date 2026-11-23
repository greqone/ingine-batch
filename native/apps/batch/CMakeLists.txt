# ingine-batch-bench: many requests at once (model/batch.h), aggregate decode throughput. Needs the model.
if(TARGET ingine_model)
  add_executable(ingine-batch-bench main.cpp)
  ingine_internal_includes(ingine-batch-bench)
  target_compile_definitions(ingine-batch-bench PRIVATE INGINE_REPO_DIR="${PROJECT_SOURCE_DIR}/..")
  target_link_libraries(ingine-batch-bench PRIVATE ingine_model ingine_serve ingine_text ingine_utf8proc CUDA::cudart)
else()
  message(STATUS "ingine-batch-bench: ingine_model not in this tree; skipped")
endif()
