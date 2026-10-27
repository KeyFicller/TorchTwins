#pragma once
// Lesson 11 - distributed (twin of py/torchtwins/distributed.py).

#include "torchtwins/config.h"

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace torchtwins {

// Tiny, deterministic demo: 4 samples / 2 ranks, few SGD steps. Not the GPT preset.
inline constexpr int64_t kDistWorldSize = 2;
inline constexpr int64_t kDistSteps = 5;
inline constexpr double kDistLr = 0.25;
inline constexpr int64_t kDistInFeatures = 2;

// @twin make_xy
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> make_xy();

// @twin new_model
[[nodiscard]] torch::nn::Linear new_model(uint64_t seed);

// @twin train_single
[[nodiscard]] std::pair<std::vector<double>, torch::Tensor> train_single(uint64_t seed);

// @twin shard
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> shard(const torch::Tensor& x,
                                                            const torch::Tensor& y, int64_t rank,
                                                            int64_t world_size);

// @endtwin
void run_distributed(const Preset& preset);

}  // namespace torchtwins
