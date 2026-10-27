// CLI - mirrors py/torchtwins/__main__.py.
//
//   torchtwins <lesson> [--preset tiny|small|gpt2] [--seed N] [--curve FILE]
#include "torchtwins/attention.h"
#include "torchtwins/autograd.h"
#include "torchtwins/checkpoint.h"
#include "torchtwins/config.h"
#include "torchtwins/crossload.h"
#include "torchtwins/data.h"
#include "torchtwins/distributed.h"
#include "torchtwins/finetune.h"
#include "torchtwins/gpt.h"
#include "torchtwins/modules.h"
#include "torchtwins/remap.h"
#include "torchtwins/tensor_parallel.h"
#include "torchtwins/tensors.h"
#include "torchtwins/tokenizer.h"
#include "torchtwins/train.h"

#include <array>
#include <exception>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct Lesson {
  std::string_view name;
  std::string_view description;
  std::function<void(const torchtwins::Preset&)> run;
};

const std::array<Lesson, 13> kLessons = {{
    {.name = "tensors",
     .description = "01 creation, shapes, indexing, broadcasting",
     .run = torchtwins::run_tensors},
    {.name = "autograd",
     .description = "02 gradients, no_grad, hand-written SGD",
     .run = torchtwins::run_autograd},
    {.name = "modules",
     .description = "03 GELU, LayerNorm, FeedForward",
     .run = torchtwins::run_modules},
    {.name = "attention",
     .description = "04 causal multi-head self-attention",
     .run = torchtwins::run_attention},
    {.name = "gpt",
     .description = "05 GPT model and greedy generation",
     .run = torchtwins::run_gpt},
    {.name = "data",
     .description = "06 tokens, train/val split, windows, batches",
     .run = torchtwins::run_data},
    {.name = "train",
     .description = "06 pre-training with AdamW, loss curves",
     .run = [](const torchtwins::Preset& preset) { torchtwins::run_train(preset); }},
    {.name = "finetune",
     .description = "07 spam classifier: new head, frozen body",
     .run = torchtwins::run_finetune},
    {.name = "checkpoint",
     .description = "08 save/load weights, restore logits",
     .run = torchtwins::run_checkpoint},
    {.name = "crossload",
     .description = "09 portable pickle bridge across runtimes",
     .run = torchtwins::run_crossload},
    {.name = "remap",
     .description = "10 HF/OpenAI GPT-2 key remap (Conv1D, c_attn)",
     .run = torchtwins::run_remap},
    {.name = "distributed",
     .description = "11 gloo DDP vs manual allreduce grads",
     .run = torchtwins::run_distributed},
    {.name = "tensor_parallel",
     .description = "12 column/row-parallel Linear shard math",
     .run = torchtwins::run_tensor_parallel},
}};

void tokenize_file(const std::string& path) {
  const std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto ids = torchtwins::Tokenizer().encode(buffer.str());
  std::cout << ids.size() << '\n';
  for (size_t i = 0; i < ids.size(); ++i) {
    std::cout << (i == 0 ? "" : " ") << ids[i];
  }
  std::cout << '\n';
}

int usage(int code) {
  std::cerr << "usage: torchtwins <lesson|list|tokenize FILE> [--preset NAME] [--seed N] [--curve "
               "FILE]\n";
  return code;
}

int run(int argc, char** argv) {
  std::optional<std::string> command;
  std::optional<std::string> path;
  std::string preset_name = "tiny";
  std::optional<uint64_t> seed;
  std::optional<std::string> curve;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--preset" && i + 1 < argc) {
      preset_name = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else if (arg == "--curve" && i + 1 < argc) {
      curve = argv[++i];
    } else if (arg.starts_with("-")) {
      return usage(2);
    } else if (!command) {
      command = std::string(arg);
    } else {
      path = std::string(arg);
    }
  }
  if (!command) {
    return usage(2);
  }
  if (*command == "list") {
    for (const auto& lesson : kLessons) {
      std::cout << std::format("{:<10} {}\n", lesson.name, lesson.description);
    }
    return 0;
  }
  if (*command == "tokenize") {
    if (!path) {
      return usage(2);
    }
    tokenize_file(*path);
    return 0;
  }
  auto preset = torchtwins::load_preset(preset_name);
  if (seed) {
    preset = preset.with_seed(*seed);
  }
  if (*command == "train") {
    torchtwins::run_train(preset, curve);
    return 0;
  }
  for (const auto& lesson : kLessons) {
    if (lesson.name == *command) {
      lesson.run(preset);
      return 0;
    }
  }
  return usage(2);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
