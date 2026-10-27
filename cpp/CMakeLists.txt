# TorchTwins C++ side: one library mirroring py/torchtwins, plus the CLI.
find_package(Torch QUIET)
if(NOT Torch_FOUND)
  message(FATAL_ERROR
    "LibTorch not found (looked under CMAKE_PREFIX_PATH, including "
    "${LIBTORCH_VENDOR_PATH}).\n"
    "  tools/fetch_libtorch.sh\n"
    "    downloads the CPU build pinned in tools/libtorch.version into vendor/libtorch\n"
    "  cmake -DLIBTORCH_VENDOR_PATH=/path/to/libtorch ...\n"
    "    points at an existing install instead")
endif()

# LibTorch ships libgloo but not gloo/*.h; ProcessGroupGloo.hpp needs them.
set(GLOO_HEADERS_PATH "${CMAKE_SOURCE_DIR}/vendor/gloo" CACHE PATH "Path to gloo sources (for headers)")
if(NOT EXISTS "${GLOO_HEADERS_PATH}/gloo/algorithm.h")
  include(FetchContent)
  FetchContent_Declare(
      gloo_headers
      GIT_REPOSITORY https://github.com/facebookincubator/gloo.git
      GIT_TAG main
      GIT_SHALLOW TRUE
  )
  FetchContent_GetProperties(gloo_headers)
  if(NOT gloo_headers_POPULATED)
    FetchContent_Populate(gloo_headers)
  endif()
  set(GLOO_HEADERS_PATH "${gloo_headers_SOURCE_DIR}")
endif()

set(TORCHTWINS_SOURCES
    torchtwins/report.cpp
    torchtwins/config.cpp
    torchtwins/tokenizer.cpp
    torchtwins/tensors.cpp
    torchtwins/autograd.cpp
    torchtwins/modules.cpp
    torchtwins/attention.cpp
    torchtwins/gpt.cpp
    torchtwins/data.cpp
    torchtwins/train.cpp
    torchtwins/finetune.cpp
    torchtwins/checkpoint.cpp
    torchtwins/crossload.cpp
    torchtwins/remap.cpp
    torchtwins/distributed.cpp
    torchtwins/tensor_parallel.cpp
)

add_library(torchtwins_lib STATIC ${TORCHTWINS_SOURCES})
# Headers are included as "torchtwins/<name>.h", same as the Python import path.
target_include_directories(torchtwins_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(torchtwins_lib SYSTEM PRIVATE ${GLOO_HEADERS_PATH})
target_include_directories(torchtwins_lib SYSTEM PRIVATE
    ${meta_pytorch_tokenizers_SOURCE_DIR}/third-party/json/single_include)
target_compile_definitions(torchtwins_lib PUBLIC TORCHTWINS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# PUBLIC so executables inherit regex_lookahead's whole-archive link options: it registers
# the PCRE2 fallback (needed for GPT-2's `(?!\S)`) from a static initializer.
target_link_libraries(torchtwins_lib PUBLIC ${TORCH_LIBRARIES} tokenizers::tokenizers)
target_compile_features(torchtwins_lib PUBLIC cxx_std_20)
target_compile_options(torchtwins_lib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(torchtwins torchtwins/main.cpp)
target_link_libraries(torchtwins PRIVATE torchtwins_lib)
set_target_properties(torchtwins PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(lesson tensors autograd modules attention gpt data train finetune checkpoint crossload remap distributed tensor_parallel)
  add_test(NAME ${lesson} COMMAND torchtwins ${lesson})
endforeach()
add_test(NAME unknown_lesson COMMAND torchtwins no_such_lesson)
set_tests_properties(unknown_lesson PROPERTIES WILL_FAIL TRUE)

add_executable(parity tools/parity.cpp)
target_link_libraries(parity PRIVATE torchtwins_lib)
set_target_properties(parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME parity_usage COMMAND parity)
set_tests_properties(parity_usage PROPERTIES WILL_FAIL TRUE)

add_executable(bench tools/bench.cpp)
target_link_libraries(bench PRIVATE torchtwins_lib)
set_target_properties(bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME bench_smoke COMMAND bench tiny 1 1 2)
