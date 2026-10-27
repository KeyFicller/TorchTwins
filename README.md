# TorchTwins：一个 GPT，写两遍

用 **PyTorch（Python）** 和 **LibTorch（C++）** 各实现一遍同一个小型 GPT。两边文件名、类名、函数名一一对应，
每课打印的输出逐行相同，权重和 logits 数值一致。可以把它当作 PyTorch ↔ LibTorch 的对照教材。

**👉 并排对照站点：<https://keyficller.github.io/TorchTwins/>**

站点由 `tools/build_site.py` 从源码自动生成（GitHub Pages 每次部署前也会跑一遍，不再提交生成好的 HTML）：
左边 Python、右边 C++，按 `@twin` 代码块对齐，块内再按 `@step` 语义锚点切成一步一行，
API 差异高亮并附解释；结果页的表格和图表全部读自 `docs/results/*.json`。
首页结构图是预渲染的 SVG（`docs/diagrams/`），不依赖 jsDelivr / Mermaid CDN。
本地预览：`pip install pygments && python tools/build_site.py`，再打开 `docs/site/index.html`。

## 课程

| 课 | 命令 | 内容 | 关键 API 差异 |
|----|------|------|---------------|
| 01 | `tensors` | 创建、形状、view/reshape、广播、matmul、索引 | `x[:, 1:]` ↔ `x.index({Slice(), Slice(1, None)})`，`(2, 3)` ↔ `{2, 3}` |
| 02 | `autograd` | requires_grad、backward、手写 SGD、no_grad | `with torch.no_grad()` ↔ `torch::NoGradGuard` |
| 03 | `modules` | GELU、LayerNorm、FeedForward | `nn.Module` ↔ `torch::nn::Module` + `TORCH_MODULE`，`register_module` |
| 04 | `attention` | 因果多头注意力，mask 作为 buffer | `register_buffer`、`masked_fill`、`transpose` |
| 05 | `gpt` | TransformerBlock、GPTModel、贪心生成 | `nn.Sequential` ↔ `->forward()`，`ModuleList` |
| 06 | `data` / `train` | GPT-2 BPE、滑动窗口、确定性 batch、AdamW 训练 | `DataLoader` ↔ 两边相同的手写 batch 迭代 |
| 07 | `finetune` | 垃圾短信分类：换分类头、冻结参数、准确率 | `requires_grad_(False)`，替换子模块 |
| 08 | `checkpoint` | 保存 / 装回权重，logits 复原 | `torch.save(state_dict)` ↔ `torch::save(model)`（文件格式不互通） |
| 09 | `crossload` | 便携 `(names, tensors)` pickle，跨运行时可互通 | `torch.save((names, tensors))` ↔ `torch::pickle_save` / `pickle_load` |
| 10 | `remap` | HF/OpenAI GPT-2 键名重映射（合成张量） | Conv1D `.t()` vs reshape、`c_attn` 拆分、绑定 `lm_head`↔`wte`（untied/tied 前向对照 + omit 回退）、OpenAI 无前缀、缺键/多余键严格拒绝、形状/dtype 不符预检拒绝 |
| 11 | `distributed` | gloo 双进程：单进程基线 vs DDP/手写 allreduce，loss 与权重对齐 | `DistributedDataParallel` ↔ `ProcessGroupGloo` + 手写 `allreduce` |
| 12 | `tensor_parallel` | 单进程演示列/行并行 Linear：分片 matmul + all-gather / all-reduce，对齐 dense | `F.linear` + `cat`/`sum` ↔ `F::linear` + `torch::cat`/`sum`（模拟集合通信） |

另有三个基础设施"双胞胎"：`config`（共享 `configs/presets.json`）、`tokenizer`（离线 GPT-2 BPE）、`report`（统一输出格式）。

## 快速开始

### Python（PyTorch）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m torchtwins list          # 列出所有课程
python -m torchtwins tensors       # 运行第 01 课
python -m torchtwins train --preset tiny
```

### C++（LibTorch ≥ 2.10）

```bash
# 下载与 CI 相同版本的 CPU LibTorch 到 vendor/libtorch（版本钉在 tools/libtorch.version）
tools/fetch_libtorch.sh
# 或指向已有安装：cmake -DLIBTORCH_VENDOR_PATH=/path/to/libtorch ...

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
./build/bin/torchtwins tensors     # 和 python -m torchtwins tensors 输出逐行相同
./build/bin/torchtwins distributed  # 11：gloo 双进程，与 python -m torchtwins distributed 逐行相同
./build/bin/torchtwins tensor_parallel  # 12：列/行并行分片数学，与 python -m torchtwins tensor_parallel 逐行相同
```

第 11 课链接 `ProcessGroupGloo`，需要 gloo 头文件。CMake 会优先用 `vendor/gloo/`（可自行 clone），否则 `FetchContent` 拉取；LibTorch 已自带 `libgloo`，不必再编 gloo 本体。

### 跑一遍全部门禁

```bash
tools/lint.sh build                    # ruff、mypy --strict、@twin/@step 检查、clang-format、clang-tidy
ctest --test-dir build                 # C++ 冒烟测试
python -m pytest tests                 # 两边课程输出 diff、数值 parity、训练曲线 parity
python tools/parity.py                 # 权重初始化 + 前向 logits 对比
python tools/train_parity.py           # 逐步 loss 对比
python tools/build_site.py            # 生成对照站点（Pages 部署时也会生成）
```

## 实测结果

数据来自 `docs/results/`（PyTorch 2.14.1，LibTorch 2.10.0，Linux x86_64，CPU，4 线程）。

**数值一致性**（容差 1e-4）

| 模型 | 同种子初始化：权重最大误差 | 同权重前向：logits 最大误差 |
|------|---------------------------|-----------------------------|
| tiny | 0 | 0 |
| small | 0 | 7.2e-07 |
| gpt2（124M 结构） | 0 | 1.7e-06 |

**训练一致性**：tiny 配置、seed 123，两边都训练 288 步；逐步训练 loss 最大误差 1.9e-06，验证 loss 最大误差 4.8e-07。

**速度**（预热 5 次，取 30 次平均，单位 ms）

| 模型 | batch × 长度 | 前向 Python | 前向 C++ | 训练一步 Python | 训练一步 C++ |
|------|-------------|------------|----------|----------------|-------------|
| tiny | 4 × 32 | 1.53 | 1.34 | 27.5 | 24.8 |
| small | 2 × 256 | 27.9 | 33.0 | 160.6 | 163.6 |

两边调用的是同一套 ATen 算子，速度差距不大；两边版本不同，又是在共享机器上测的，这些数字只说明"差不多"，不说明谁更快。

## 目录结构

```
configs/presets.json     共享配置（tiny / small / gpt2）
assets/                  语料、垃圾短信数据集、gpt2.tiktoken
py/torchtwins/             Python 包，每课一个文件
cpp/torchtwins/            C++ 库，与 py/torchtwins 逐文件对应（.h + .cpp）
cpp/tools/               parity、bench 可执行文件
tools/                   parity / 训练 parity / benchmark / @twin 检查 / 站点生成 / lint / fetch_libtorch
tests/                   pytest（同时驱动 C++ 可执行文件）
docs/site/               生成的对照站点（GitHub Pages）
docs/results/            实测结果 JSON
```

## 已知差距

- 只在 Linux CPU 上验证过，macOS 和 CUDA 没测。
- 指令微调不在课程范围内。
- 第 08 课的 Module / `state_dict` 档案仍不互通；第 09 课的便携 `(names, tensors)` pickle 可以跨语言加载（`tools/parity.py` 用的就是这一套）。
- 第 10 课用**合成** HF/OpenAI 布局演示了 Conv1D 转置、`c_attn` 拆分、无 `transformer.` 前缀的 OpenAI 键名、绑定 `lm_head`→`wte`，以及装回时缺键/多余键的严格拒绝；尚未接入下载真实 `openai-community/gpt2` 权重（CI 体积/网络）。
