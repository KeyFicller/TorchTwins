// Lesson 13 - pipeline_parallel: stage split + micro-batch handoff (single-process).
#include "torchtwins/pipeline_parallel.h"

#include "torchtwins/modules.h"
#include "torchtwins/report.h"

#include <format>
#include <stdexcept>
#include <vector>

namespace F = torch::nn::functional;

namespace torchtwins {

// @twin layer_forward
torch::Tensor layer_forward(const torch::Tensor& x, const torch::Tensor& weight,
                            const torch::Tensor& bias) {
  // @step linear
  return GELU()(F::linear(x, weight, bias));
}

// @twin stage_forward
torch::Tensor stage_forward(const torch::Tensor& x, const std::vector<torch::Tensor>& weights,
                            const std::vector<torch::Tensor>& biases) {
  // @step layers
  auto h = x;
  for (size_t i = 0; i < weights.size(); ++i) {
    h = layer_forward(h, weights[i], biases[i]);
  }
  return h;
}

// @twin send_activation
torch::Tensor send_activation(const torch::Tensor& activation) {
  // @step handoff
  return activation.contiguous();
}

// @twin recv_activation
torch::Tensor recv_activation(const torch::Tensor& activation) {
  // @step handoff
  return activation.contiguous();
}

// @twin split_microbatches
std::vector<torch::Tensor> split_microbatches(const torch::Tensor& x, int64_t num_microbatches) {
  // @step split
  if (x.size(0) % num_microbatches != 0) {
    throw std::runtime_error(
        std::format("batch {} not divisible by num_microbatches {}", x.size(0), num_microbatches));
  }
  const auto chunks = x.chunk(num_microbatches, /*dim=*/0);
  return {chunks.begin(), chunks.end()};
}

// @twin join_microbatches
torch::Tensor join_microbatches(const std::vector<torch::Tensor>& pieces) {
  // @step join
  return torch::cat(pieces, /*dim=*/0);
}

// @twin dense_forward
torch::Tensor dense_forward(const torch::Tensor& x, const std::vector<torch::Tensor>& weights,
                            const std::vector<torch::Tensor>& biases) {
  // @step forward
  return stage_forward(x, weights, biases);
}

// @twin pipeline_forward
torch::Tensor pipeline_forward(const torch::Tensor& x,
                               const std::vector<std::vector<torch::Tensor>>& stage_weights,
                               const std::vector<std::vector<torch::Tensor>>& stage_biases) {
  // @step stage0
  auto h = stage_forward(x, stage_weights[0], stage_biases[0]);
  // @step p2p
  h = recv_activation(send_activation(h));
  // @step stage1
  return stage_forward(h, stage_weights[1], stage_biases[1]);
}

// @twin pipeline_microbatch_forward
torch::Tensor pipeline_microbatch_forward(
    const torch::Tensor& x, const std::vector<std::vector<torch::Tensor>>& stage_weights,
    const std::vector<std::vector<torch::Tensor>>& stage_biases, int64_t num_microbatches) {
  // @step split
  const auto micros = split_microbatches(x, num_microbatches);
  // @step pipe
  std::vector<torch::Tensor> outs;
  outs.reserve(micros.size());
  for (const auto& micro : micros) {
    outs.push_back(pipeline_forward(micro, stage_weights, stage_biases));
  }
  // @step join
  return join_microbatches(outs);
}

// @twin schedule_1f1b
std::vector<std::string> schedule_1f1b(int64_t num_stages, int64_t num_microbatches) {
  // @step build
  std::vector<std::string> events;
  for (int64_t mb = 0; mb < num_stages - 1; ++mb) {
    for (int64_t stage = 0; stage < num_stages; ++stage) {
      events.push_back(std::format("F mb{} stage{}", mb, stage));
    }
  }
  for (int64_t mb = num_stages - 1; mb < num_microbatches; ++mb) {
    for (int64_t stage = 0; stage < num_stages; ++stage) {
      events.push_back(std::format("F mb{} stage{}", mb, stage));
    }
    const auto done = mb - (num_stages - 1);
    for (int64_t stage = num_stages - 1; stage >= 0; --stage) {
      events.push_back(std::format("B mb{} stage{}", done, stage));
    }
  }
  for (int64_t mb = num_microbatches - (num_stages - 1); mb < num_microbatches; ++mb) {
    for (int64_t stage = num_stages - 1; stage >= 0; --stage) {
      events.push_back(std::format("B mb{} stage{}", mb, stage));
    }
  }
  return events;
}

namespace {

// @twin max_abs_diff
double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  // @step diff
  return (a - b).abs().max().item<double>();
}

}  // namespace

// @twin run_pipeline_parallel
void run_pipeline_parallel(const Preset& preset) {
  // @step banner
  report::section("pipeline parallel");
  report::count("num stages", kPpNumStages);
  report::count("num microbatches", kPpNumMicrobatches);
  report::count("batch", kPpBatch);
  report::count("seq", kPpSeq);
  report::count("features", kPpFeatures);
  report::text("p2p", "simulated (identity send/recv)");
  report::text("schedule", "1F1B labels (forward numeric parity uses micro-batch pipe)");

  // @step init
  torch::manual_seed(preset.seed);
  const auto x = torch::randn({kPpBatch, kPpSeq, kPpFeatures});
  std::vector<torch::Tensor> weights;
  std::vector<torch::Tensor> biases;
  weights.reserve(4);
  biases.reserve(4);
  // Same RNG order as Python: all weights, then all biases.
  for (int i = 0; i < 4; ++i) {
    weights.push_back(torch::randn({kPpFeatures, kPpFeatures}));
  }
  for (int i = 0; i < 4; ++i) {
    biases.push_back(torch::randn({kPpFeatures}));
  }
  const std::vector<std::vector<torch::Tensor>> stage_weights = {
      {weights[0], weights[1]},
      {weights[2], weights[3]},
  };
  const std::vector<std::vector<torch::Tensor>> stage_biases = {
      {biases[0], biases[1]},
      {biases[2], biases[3]},
  };

  // @step dense
  report::section("dense forward");
  const auto dense = dense_forward(x, weights, biases);
  report::show("output", dense);

  // @step sequential
  report::section("pipeline sequential");
  report::text("handoff", "stage0 → send/recv → stage1");
  const auto seq = pipeline_forward(x, stage_weights, stage_biases);
  report::show("output", seq);
  report::scalar("max abs diff vs dense", max_abs_diff(seq, dense));

  // @step microbatch
  report::section("pipeline microbatch");
  report::count("micro batch size", kPpBatch / kPpNumMicrobatches);
  const auto micro =
      pipeline_microbatch_forward(x, stage_weights, stage_biases, kPpNumMicrobatches);
  report::show("output", micro);
  report::scalar("max abs diff vs dense", max_abs_diff(micro, dense));

  // @step schedule
  report::section("1f1b schedule");
  const auto events = schedule_1f1b(kPpNumStages, kPpNumMicrobatches);
  report::count("events", static_cast<int64_t>(events.size()));
  for (size_t i = 0; i < events.size(); ++i) {
    report::text(std::format("event {:02d}", i), events[i]);
  }
}

}  // namespace torchtwins
