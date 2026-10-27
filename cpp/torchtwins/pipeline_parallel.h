#pragma once
// Lesson 13 - pipeline_parallel (twin of py/torchtwins/pipeline_parallel.py).

#include "torchtwins/config.h"

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torchtwins {

inline constexpr int64_t kPpNumStages = 2;
inline constexpr int64_t kPpNumMicrobatches = 4;
inline constexpr int64_t kPpBatch = 8;
inline constexpr int64_t kPpSeq = 2;
inline constexpr int64_t kPpFeatures = 4;

// @twin layer_forward
[[nodiscard]] torch::Tensor layer_forward(const torch::Tensor& x, const torch::Tensor& weight,
                                          const torch::Tensor& bias);

// @twin stage_forward
[[nodiscard]] torch::Tensor stage_forward(const torch::Tensor& x,
                                          const std::vector<torch::Tensor>& weights,
                                          const std::vector<torch::Tensor>& biases);

// @twin send_activation
[[nodiscard]] torch::Tensor send_activation(const torch::Tensor& activation);

// @twin recv_activation
[[nodiscard]] torch::Tensor recv_activation(const torch::Tensor& activation);

// @twin split_microbatches
[[nodiscard]] std::vector<torch::Tensor> split_microbatches(const torch::Tensor& x,
                                                            int64_t num_microbatches);

// @twin join_microbatches
[[nodiscard]] torch::Tensor join_microbatches(const std::vector<torch::Tensor>& pieces);

// @twin dense_forward
[[nodiscard]] torch::Tensor dense_forward(const torch::Tensor& x,
                                          const std::vector<torch::Tensor>& weights,
                                          const std::vector<torch::Tensor>& biases);

// @twin pipeline_forward
[[nodiscard]] torch::Tensor pipeline_forward(
    const torch::Tensor& x, const std::vector<std::vector<torch::Tensor>>& stage_weights,
    const std::vector<std::vector<torch::Tensor>>& stage_biases);

// @twin pipeline_microbatch_forward
[[nodiscard]] torch::Tensor pipeline_microbatch_forward(
    const torch::Tensor& x, const std::vector<std::vector<torch::Tensor>>& stage_weights,
    const std::vector<std::vector<torch::Tensor>>& stage_biases, int64_t num_microbatches);

// @twin schedule_1f1b
[[nodiscard]] std::vector<std::string> schedule_1f1b(int64_t num_stages, int64_t num_microbatches);

// @endtwin
void run_pipeline_parallel(const Preset& preset);

}  // namespace torchtwins
