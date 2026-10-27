// Lesson 11 - distributed: gloo process group, DDP vs manual allreduce grads.
#include "torchtwins/distributed.h"

#include "torchtwins/report.h"

#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace torchtwins {

namespace {

constexpr const char* kEnvRank = "TORCHTWINS_DIST_RANK";
constexpr const char* kEnvWorld = "TORCHTWINS_DIST_WORLD_SIZE";
constexpr const char* kEnvStore = "TORCHTWINS_DIST_STORE";
constexpr const char* kEnvSeed = "TORCHTWINS_DIST_SEED";

std::shared_ptr<c10d::ProcessGroupGloo> make_pg(int rank, int world_size,
                                                const std::string& store_path) {
  auto store = c10::make_intrusive<c10d::FileStore>(store_path, world_size);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());
  return std::make_shared<c10d::ProcessGroupGloo>(store, rank, world_size, options);
}

void report_train(const std::vector<double>& losses, const torch::Tensor& weight,
                  const char* grad_sync, int64_t world_size, bool with_rank) {
  if (with_rank) {
    report::count("rank", 0);
    report::count("world size", world_size);
  }
  report::text("grad sync", grad_sync);
  for (size_t i = 0; i < losses.size(); ++i) {
    report::text(std::format("step {:04d}", i + 1), std::format("loss {:.4f}", losses[i]));
  }
  report::scalar("final loss", losses.back());
  report::show("weight", weight);
  std::cout.flush();
}

std::string self_exe() { return std::filesystem::canonical("/proc/self/exe").string(); }

class StdoutSilence {
 public:
  StdoutSilence() {
    fflush(stdout);
    saved_ = dup(STDOUT_FILENO);
    const int null_fd = open("/dev/null", O_WRONLY);
    if (null_fd >= 0) {
      dup2(null_fd, STDOUT_FILENO);
      close(null_fd);
    }
  }
  ~StdoutSilence() {
    fflush(stdout);
    if (saved_ >= 0) {
      dup2(saved_, STDOUT_FILENO);
      close(saved_);
    }
  }
  StdoutSilence(const StdoutSilence&) = delete;
  StdoutSilence& operator=(const StdoutSilence&) = delete;

 private:
  int saved_{-1};
};

std::string make_store_path() {
  return (std::filesystem::temp_directory_path() /
          std::format("torchtwins_dist_{}_{}", getpid(), getuid()))
      .string();
}

void spawn_ddp_workers(const Preset& preset, const std::string& store_path) {
  // Re-exec after torch work in the parent: plain fork() after LibTorch init can
  // deadlock on OpenMP / thread pools. Each child is a fresh process.
  const auto exe = self_exe();
  std::vector<pid_t> kids;
  kids.reserve(static_cast<size_t>(kDistWorldSize));
  for (int64_t rank = 0; rank < kDistWorldSize; ++rank) {
    const pid_t pid = fork();
    if (pid < 0) {
      throw std::runtime_error("fork failed for distributed worker");
    }
    if (pid == 0) {
      setenv(kEnvRank, std::to_string(rank).c_str(), 1);
      setenv(kEnvWorld, std::to_string(kDistWorldSize).c_str(), 1);
      setenv(kEnvStore, store_path.c_str(), 1);
      setenv(kEnvSeed, std::to_string(preset.seed).c_str(), 1);
      // Rank 0 keeps stdout for the transcript; other ranks discard stdout.
      // All ranks discard stderr so Gloo/tokenizers chatter does not interleave.
      if (rank != 0) {
        const int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
          dup2(null_fd, STDOUT_FILENO);
          close(null_fd);
        }
      }
      {
        const int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
          dup2(null_fd, STDERR_FILENO);
          close(null_fd);
        }
      }
      const std::string seed = std::to_string(preset.seed);
      execl(exe.c_str(), "torchtwins", "distributed", "--seed", seed.c_str(),
            static_cast<char*>(nullptr));
      _exit(127);
    }
    kids.push_back(pid);
  }
  int failures = 0;
  for (const pid_t kid : kids) {
    int status = 0;
    if (waitpid(kid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++failures;
    }
  }
  if (failures != 0) {
    throw std::runtime_error("distributed worker failed");
  }
}

}  // namespace

// @twin make_xy
std::pair<torch::Tensor, torch::Tensor> make_xy() {
  // @step data
  // Target is 2*x0 + 3*x1 so a bias-free Linear(2, 1) can fit it.
  const auto x = torch::tensor({{0.5, 1.0}, {1.0, 0.5}, {0.0, 1.5}, {1.5, 0.0}}, torch::kFloat32);
  const auto y = 2.0 * x.select(/*dim=*/1, /*index=*/0).unsqueeze(1) +
                 3.0 * x.select(/*dim=*/1, /*index=*/1).unsqueeze(1);
  return {x, y};
}

// @twin new_model
torch::nn::Linear new_model(uint64_t seed) {
  // @step init
  torch::manual_seed(seed);
  torch::nn::Linear model(torch::nn::LinearOptions(kDistInFeatures, 1).bias(false));
  {
    const torch::NoGradGuard guard;
    model->weight.zero_();
  }
  return model;
}

// @twin train_single
std::pair<std::vector<double>, torch::Tensor> train_single(uint64_t seed) {
  // @step setup
  auto model = new_model(seed);
  const auto [x, y] = make_xy();
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(kDistLr));
  std::vector<double> losses;
  // @step loop
  for (int64_t step = 0; step < kDistSteps; ++step) {
    optimizer.zero_grad();
    const auto loss = (model(x) - y).pow(2).mean();
    loss.backward();
    optimizer.step();
    losses.push_back(loss.item<double>());
  }
  return {losses, model->weight.detach().clone()};
}

// @twin shard
std::pair<torch::Tensor, torch::Tensor> shard(const torch::Tensor& x, const torch::Tensor& y,
                                              int64_t rank, int64_t world_size) {
  // @step slice
  // world_size must divide the sample count so every rank sees the same local batch size.
  const auto n = x.size(0) / world_size;
  const auto start = rank * n;
  const auto end = start + n;
  return {x.slice(/*dim=*/0, start, end).contiguous(), y.slice(/*dim=*/0, start, end).contiguous()};
}

// @twin allreduce_avg_
void allreduce_avg_(c10d::Backend& pg, torch::Tensor& tensor) {
  // @step avg
  // SUM allreduce then divide — matches DDP's gradient averaging and Python's helper.
  std::vector<at::Tensor> tensors = {tensor};
  c10d::AllreduceOptions opts;
  opts.reduceOp = c10d::ReduceOp(c10d::ReduceOp::SUM);
  pg.allreduce(tensors, opts)->wait();
  tensor /= static_cast<double>(pg.getSize());
}

// @twin train_ddp_rank
std::pair<std::vector<double>, torch::Tensor> train_ddp_rank(int rank, int world_size,
                                                             uint64_t seed,
                                                             const std::string& store_path) {
  // @step group
  // LibTorch has no DistributedDataParallel wrapper like Python; we open a gloo
  // ProcessGroup and allreduce grads by hand after backward (same math as DDP).
  // Gloo prints a connection banner to stdout; silence it so the twin transcript
  // stays byte-stable with Python.
  std::shared_ptr<c10d::ProcessGroupGloo> pg;
  {
    const StdoutSilence silence;
    pg = make_pg(rank, world_size, store_path);
  }
  // @step wrap
  // "wrap" on C++ = keep the raw module; sync is explicit in the loop below.
  auto model = new_model(seed);
  const auto [x, y] = make_xy();
  const auto [local_x, local_y] = shard(x, y, rank, world_size);
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(kDistLr));
  std::vector<double> losses;
  // @step loop
  for (int64_t step = 0; step < kDistSteps; ++step) {
    optimizer.zero_grad();
    const auto loss = (model(local_x) - local_y).pow(2).mean();
    auto loss_report = loss.detach().clone();
    allreduce_avg_(*pg, loss_report);
    loss.backward();
    for (auto& param : model->parameters()) {
      auto& grad = param.mutable_grad();
      if (!grad.defined()) {
        continue;
      }
      allreduce_avg_(*pg, grad);
    }
    optimizer.step();
    losses.push_back(loss_report.item<double>());
  }
  return {losses, model->weight.detach().clone()};
}

namespace {

void run_ddp_worker() {
  const int rank = std::atoi(std::getenv(kEnvRank));
  const int world_size = std::atoi(std::getenv(kEnvWorld));
  const std::string store_path = std::getenv(kEnvStore);
  const uint64_t seed = std::strtoull(std::getenv(kEnvSeed), nullptr, 10);
  const auto [losses, weight] = train_ddp_rank(rank, world_size, seed, store_path);
  if (rank == 0) {
    report_train(losses, weight, "allreduce", world_size, /*with_rank=*/true);
  }
}

}  // namespace

// @twin run_distributed
void run_distributed(const Preset& preset) {
  if (std::getenv(kEnvRank) != nullptr) {
    run_ddp_worker();
    return;
  }

  // @step banner
  report::section("process group");
  report::text("backend", "gloo");
  report::count("world size", kDistWorldSize);
  report::count("steps", kDistSteps);
  std::cout.flush();

  // @step single
  report::section("single process");
  const auto [single_losses, single_weight] = train_single(preset.seed);
  report_train(single_losses, single_weight, "none", kDistWorldSize, /*with_rank=*/false);
  std::cout.flush();

  // @step ddp
  report::section("distributed (ddp)");
  std::cout.flush();  // section header before workers write their transcript lines
  const auto store_path = make_store_path();
  std::filesystem::remove(store_path);
  try {
    spawn_ddp_workers(preset, store_path);
  } catch (...) {
    std::filesystem::remove(store_path);
    throw;
  }
  std::filesystem::remove(store_path);
}

}  // namespace torchtwins
