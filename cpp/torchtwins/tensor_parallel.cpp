// Lesson 12 - tensor_parallel: column/row-parallel Linear shard math (single-process).
#include "torchtwins/tensor_parallel.h"

#include "torchtwins/modules.h"
#include "torchtwins/report.h"

#include <format>
#include <stdexcept>
#include <vector>

namespace F = torch::nn::functional;

namespace torchtwins {

// @twin shard_size
int64_t shard_size(int64_t dim, int64_t world_size) {
  // @step check
  if (dim % world_size != 0) {
    throw std::runtime_error(std::format("dim {} not divisible by world_size {}", dim, world_size));
  }
  return dim / world_size;
}

// @twin column_weight_shard
torch::Tensor column_weight_shard(const torch::Tensor& weight, int64_t rank, int64_t world_size) {
  // @step slice
  // Linear weight is [out, in]; column-parallel shards along out (dim 0).
  const auto rows = shard_size(weight.size(0), world_size);
  const auto start = rank * rows;
  return weight.slice(/*dim=*/0, start, start + rows).contiguous();
}

// @twin row_weight_shard
torch::Tensor row_weight_shard(const torch::Tensor& weight, int64_t rank, int64_t world_size) {
  // @step slice
  // Row-parallel shards along in (dim 1) → [out, in/ws].
  const auto cols = shard_size(weight.size(1), world_size);
  const auto start = rank * cols;
  return weight.slice(/*dim=*/1, start, start + cols).contiguous();
}

// @twin bias_shard
torch::Tensor bias_shard(const torch::Tensor& bias, int64_t rank, int64_t world_size) {
  // @step slice
  const auto rows = shard_size(bias.size(0), world_size);
  const auto start = rank * rows;
  return bias.slice(/*dim=*/0, start, start + rows).contiguous();
}

// @twin input_feature_shard
torch::Tensor input_feature_shard(const torch::Tensor& x, int64_t rank, int64_t world_size) {
  // @step slice
  const auto cols = shard_size(x.size(-1), world_size);
  const auto start = rank * cols;
  return x.slice(/*dim=*/-1, start, start + cols).contiguous();
}

// @twin all_gather_cat
torch::Tensor all_gather_cat(const std::vector<torch::Tensor>& pieces, int64_t dim) {
  // @step gather
  return torch::cat(pieces, dim);
}

// @twin all_reduce_sum
torch::Tensor all_reduce_sum(const std::vector<torch::Tensor>& pieces) {
  // @step reduce
  return torch::stack(pieces, /*dim=*/0).sum(/*dim=*/0);
}

// @twin column_parallel_linear
torch::Tensor column_parallel_linear(const torch::Tensor& x, const torch::Tensor& weight,
                                     const torch::Tensor& bias, int64_t world_size) {
  // @step shards
  std::vector<torch::Tensor> pieces;
  pieces.reserve(static_cast<size_t>(world_size));
  for (int64_t rank = 0; rank < world_size; ++rank) {
    const auto w = column_weight_shard(weight, rank, world_size);
    const auto b = bias_shard(bias, rank, world_size);
    pieces.push_back(F::linear(x, w, b));
  }
  // @step gather
  return all_gather_cat(pieces, /*dim=*/-1);
}

// @twin row_parallel_linear
torch::Tensor row_parallel_linear(const torch::Tensor& x, const torch::Tensor& weight,
                                  const torch::Tensor& bias, int64_t world_size) {
  // @step shards
  std::vector<torch::Tensor> pieces;
  pieces.reserve(static_cast<size_t>(world_size));
  for (int64_t rank = 0; rank < world_size; ++rank) {
    const auto x_r = input_feature_shard(x, rank, world_size);
    const auto w = row_weight_shard(weight, rank, world_size);
    pieces.push_back(F::linear(x_r, w));
  }
  // @step reduce
  auto out = all_reduce_sum(pieces);
  // @step bias
  out = out + bias;
  return out;
}

// @twin dense_mlp
torch::Tensor dense_mlp(const torch::Tensor& x, const torch::Tensor& w1, const torch::Tensor& b1,
                        const torch::Tensor& w2, const torch::Tensor& b2) {
  // @step forward
  return F::linear(GELU()(F::linear(x, w1, b1)), w2, b2);
}

// @twin tp_mlp
torch::Tensor tp_mlp(const torch::Tensor& x, const torch::Tensor& w1, const torch::Tensor& b1,
                     const torch::Tensor& w2, const torch::Tensor& b2, int64_t world_size) {
  // @step up
  auto h = column_parallel_linear(x, w1, b1, world_size);
  // @step act
  h = GELU()(h);
  // @step down
  return row_parallel_linear(h, w2, b2, world_size);
}

namespace {

// @twin max_abs_diff
double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  // @step diff
  return (a - b).abs().max().item<double>();
}

}  // namespace

// @twin run_tensor_parallel
void run_tensor_parallel(const Preset& preset) {
  // @step banner
  report::section("tensor parallel");
  report::count("world size", kTpWorldSize);
  report::count("batch", kTpBatch);
  report::count("seq", kTpSeq);
  report::count("in features", kTpInFeatures);
  report::count("out features", kTpOutFeatures);
  report::count("hidden", kTpHidden);
  report::text("collectives", "simulated (cat / sum)");

  // @step init
  torch::manual_seed(preset.seed);
  const auto x = torch::randn({kTpBatch, kTpSeq, kTpInFeatures});
  const auto w = torch::randn({kTpOutFeatures, kTpInFeatures});
  const auto b = torch::randn({kTpOutFeatures});
  const auto w1 = torch::randn({kTpHidden, kTpInFeatures});
  const auto b1 = torch::randn({kTpHidden});
  const auto w2 = torch::randn({kTpOutFeatures, kTpHidden});
  const auto b2 = torch::randn({kTpOutFeatures});

  // @step dense
  report::section("dense linear");
  const auto dense = F::linear(x, w, b);
  report::show("output", dense);

  // @step column
  report::section("column parallel");
  report::count("shard out", shard_size(kTpOutFeatures, kTpWorldSize));
  report::text("all gather", "cat along last dim");
  const auto col = column_parallel_linear(x, w, b, kTpWorldSize);
  report::show("output", col);
  report::scalar("max abs diff vs dense", max_abs_diff(col, dense));

  // @step row
  report::section("row parallel");
  report::count("shard in", shard_size(kTpInFeatures, kTpWorldSize));
  report::text("all reduce", "sum partials");
  const auto row = row_parallel_linear(x, w, b, kTpWorldSize);
  report::show("output", row);
  report::scalar("max abs diff vs dense", max_abs_diff(row, dense));

  // @step mlp
  report::section("mlp (column then row)");
  report::count("hidden", kTpHidden);
  const auto dense_h = dense_mlp(x, w1, b1, w2, b2);
  const auto tp_h = tp_mlp(x, w1, b1, w2, b2, kTpWorldSize);
  report::show("dense mlp", dense_h);
  report::show("tp mlp", tp_h);
  report::scalar("max abs diff vs dense", max_abs_diff(tp_h, dense_h));
}

}  // namespace torchtwins
