#pragma once
// Lesson 12 - tensor_parallel (twin of py/torchtwins/tensor_parallel.py).

#include "torchtwins/config.h"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace torchtwins {

// Tiny, deterministic demo. Not the GPT preset. world_size must divide out/in/hidden.
inline constexpr int64_t kTpWorldSize = 2;
inline constexpr int64_t kTpBatch = 2;
inline constexpr int64_t kTpSeq = 3;
inline constexpr int64_t kTpInFeatures = 4;
inline constexpr int64_t kTpOutFeatures = 4;
inline constexpr int64_t kTpHidden = 8;

// @twin shard_size
[[nodiscard]] int64_t shard_size(int64_t dim, int64_t world_size);

// @twin column_weight_shard
[[nodiscard]] torch::Tensor column_weight_shard(const torch::Tensor& weight, int64_t rank,
                                                int64_t world_size);

// @twin row_weight_shard
[[nodiscard]] torch::Tensor row_weight_shard(const torch::Tensor& weight, int64_t rank,
                                             int64_t world_size);

// @twin bias_shard
[[nodiscard]] torch::Tensor bias_shard(const torch::Tensor& bias, int64_t rank, int64_t world_size);

// @twin input_feature_shard
[[nodiscard]] torch::Tensor input_feature_shard(const torch::Tensor& x, int64_t rank,
                                                int64_t world_size);

// @twin all_gather_cat
[[nodiscard]] torch::Tensor all_gather_cat(const std::vector<torch::Tensor>& pieces,
                                           int64_t dim = -1);

// @twin all_reduce_sum
[[nodiscard]] torch::Tensor all_reduce_sum(const std::vector<torch::Tensor>& pieces);

// @twin column_parallel_linear
[[nodiscard]] torch::Tensor column_parallel_linear(const torch::Tensor& x,
                                                   const torch::Tensor& weight,
                                                   const torch::Tensor& bias, int64_t world_size);

// @twin row_parallel_linear
[[nodiscard]] torch::Tensor row_parallel_linear(const torch::Tensor& x, const torch::Tensor& weight,
                                                const torch::Tensor& bias, int64_t world_size);

// @twin dense_mlp
[[nodiscard]] torch::Tensor dense_mlp(const torch::Tensor& x, const torch::Tensor& w1,
                                      const torch::Tensor& b1, const torch::Tensor& w2,
                                      const torch::Tensor& b2);

// @twin tp_mlp
[[nodiscard]] torch::Tensor tp_mlp(const torch::Tensor& x, const torch::Tensor& w1,
                                   const torch::Tensor& b1, const torch::Tensor& w2,
                                   const torch::Tensor& b2, int64_t world_size);

// @endtwin
void run_tensor_parallel(const Preset& preset);

}  // namespace torchtwins
