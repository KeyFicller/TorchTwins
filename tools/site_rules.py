"""PyTorch <-> LibTorch API differences highlighted on the comparison site.

Each rule matches one side's source line; the hover text explains the twin construct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiRule:
    key: str
    lang: str  # "py" or "cpp"
    pattern: re.Pattern[str]
    tip: str


def _rule(key: str, lang: str, pattern: str, tip: str) -> ApiRule:
    return ApiRule(key, lang, re.compile(pattern), tip)


RULES: tuple[ApiRule, ...] = (
    # --- modules ---
    _rule(
        "module",
        "py",
        r"\bnn\.Module\b",
        "Python 直接继承 nn.Module；C++ 写 XxxImpl : torch::nn::Module，再用 TORCH_MODULE 包一层",
    ),
    _rule(
        "module",
        "cpp",
        r"\btorch::nn::Module\b",
        "C++ 模块类约定以 Impl 结尾，真正使用的是 TORCH_MODULE 生成的句柄类",
    ),
    _rule(
        "TORCH_MODULE",
        "cpp",
        r"\bTORCH_MODULE\(\w+\)",
        "宏：生成持有 std::shared_ptr<XxxImpl> 的句柄类 Xxx，所以要用 model->method() 访问成员",
    ),
    _rule(
        "register_module",
        "cpp",
        r"\bregister_module\(",
        "C++ 没有 __setattr__ 魔法：子模块必须显式注册，参数才会出现在 parameters() 里；Python 只要 self.x = Module()",
    ),
    _rule(
        "register_parameter",
        "py",
        r"\bnn\.Parameter\(",
        "Python 把张量包成 nn.Parameter 再赋给 self 即自动注册",
    ),
    _rule(
        "register_parameter",
        "cpp",
        r"\bregister_parameter\(",
        "C++ 用 register_parameter(name, tensor) 显式注册可训练参数",
    ),
    _rule(
        "register_buffer",
        "py",
        r"\bregister_buffer\(",
        "buffer：随模型保存/移动设备，但不参与训练（这里是因果 mask）",
    ),
    _rule(
        "register_buffer",
        "cpp",
        r"\bregister_buffer\(",
        "与 Python 同名同义；返回注册后的张量，可直接用于成员初始化",
    ),
    _rule(
        "replace_module",
        "cpp",
        r"\breplace_module\(",
        "替换已注册的子模块；成员变量也要重新指向新模块，否则 forward 仍调用旧的",
    ),
    _rule(
        "sequential",
        "cpp",
        r"forward<torch::Tensor>",
        "C++ 的 Sequential 是类型擦除的，forward 需要显式写出返回类型",
    ),
    _rule(
        "arrow",
        "cpp",
        r"\b(model|blocks|optimizer_)->",
        "TORCH_MODULE 句柄本质是 shared_ptr：成员用 -> 访问，调用 forward 直接 model(x)",
    ),
    # --- autograd ---
    _rule("no_grad", "py", r"torch\.no_grad\(\)", "上下文管理器：with 块内不记录计算图"),
    _rule(
        "no_grad",
        "cpp",
        r"torch::NoGradGuard",
        "RAII 守卫：对象存活的作用域内不记录计算图，对应 Python 的 with torch.no_grad()",
    ),
    _rule(
        "grad",
        "cpp",
        r"\bmutable_grad\(\)",
        "grad() 返回 const 引用；就地清零梯度要用 mutable_grad()",
    ),
    _rule(
        "requires_grad",
        "cpp",
        r"torch::requires_grad\(\)",
        "C++ 用 TensorOptions 传 requires_grad，Python 是关键字参数 requires_grad=True",
    ),
    # --- tensors ---
    _rule(
        "index",
        "cpp",
        r"\.index(_put_)?\(\{",
        "C++ 没有切片语法：x[:, 1:] 写作 x.index({Slice(), Slice(1, None)})，赋值用 index_put_",
    ),
    _rule(
        "slice", "py", r"\[[^\]\n]*:[^\]\n]*\]", "Python 切片语法，C++ 侧对应 Slice(start, stop)"
    ),
    _rule("item", "cpp", r"\.item<\w+>\(\)", "C++ 的 item 是模板，需要写明要取出的标量类型"),
    _rule("dtype", "cpp", r"torch::k(Long|Float|Bool)\b", "dtype 常量：torch.long ↔ torch::kLong"),
    _rule(
        "shape",
        "cpp",
        r"torch::(zeros|ones|randn|full|randint)\(\{",
        "形状用花括号 IntArrayRef {2, 3}，Python 是元组 (2, 3)",
    ),
    # --- training ---
    _rule("optim", "py", r"torch\.optim\.AdamW\(", "超参数用关键字参数传入"),
    _rule(
        "optim",
        "cpp",
        r"torch::optim::AdamWOptions",
        "超参数用 Options 对象链式设置：AdamWOptions(lr).weight_decay(wd)",
    ),
    _rule(
        "generator",
        "py",
        r"torch\.Generator\(\)",
        "显式随机数生成器：两边用同一个 ATen CPU 算法，所以打乱顺序逐位相同",
    ),
    _rule(
        "generator",
        "cpp",
        r"at::make_generator<at::CPUGeneratorImpl>",
        "C++ 创建 CPU 生成器的方式；与 Python 的 torch.Generator().manual_seed(s) 等价",
    ),
    _rule(
        "functional",
        "cpp",
        r"\bF::\w+",
        "torch::nn::functional 的别名，对应 torch.nn.functional as F",
    ),
    _rule(
        "pickle",
        "cpp",
        r"torch::pickle_(save|load)",
        "IValue 级 pickle：第 09 课便携桥；可与 Python torch.save((names, tensors)) 互通；Module state_dict 仍不通用",
    ),
    # --- checkpoints ---
    _rule(
        "save",
        "py",
        r"torch\.save\(",
        "Python 通常存 state_dict（zip+pickle）；与 LibTorch 的 torch::save 档案不互通",
    ),
    _rule(
        "save",
        "cpp",
        r"torch::save\(",
        "LibTorch OutputArchive 序列化整个 Module；不是 Python 的 state_dict pickle",
    ),
    _rule(
        "load_state_dict",
        "py",
        r"\.load_state_dict\(",
        "按名字把张量拷进当前模块；C++ 侧由 torch::load 一次完成",
    ),
    _rule(
        "load",
        "cpp",
        r"torch::load\(",
        "InputArchive 按名字填回已注册的参数/buffer；对应 Python load_state_dict",
    ),
    _rule(
        "weights_only",
        "py",
        r"weights_only\s*=\s*True",
        "只反序列化张量，拒绝任意 pickle 对象；更安全的默认加载方式",
    ),
    _rule(
        "copy_",
        "py",
        r"\.copy_\(",
        "按名字把外来张量写进 Parameter；跨运行时装权重的基本动作（C++ 同名）",
    ),
    _rule(
        "copy_",
        "cpp",
        r"\.copy_\(",
        "按名字把外来张量写进已注册的 Parameter；跨运行时装权重的基本动作",
    ),
    _rule(
        "contiguous",
        "py",
        r"\.contiguous\(",
        "序列化前保证密排存储；C++ 同名",
    ),
    _rule(
        "contiguous",
        "cpp",
        r"\.contiguous\(",
        "序列化前保证密排存储，避免 strides 在另一侧被误解",
    ),
    _rule(
        "named_parameters",
        "py",
        r"named_parameters\(\)",
        "点分参数路径；两边顺序一致是 portable / parity 能对上的前提",
    ),
    _rule(
        "named_parameters",
        "cpp",
        r"named_parameters\(\)",
        "点分参数路径；两边顺序一致是 portable / parity 能对上的前提",
    ),
    _rule(
        "transpose_t",
        "py",
        r"\.t\(\)",
        "Conv1D↔Linear 换形：`(in,out).t()` → `(out,in)`；第 10 课 remap 的核心动作",
    ),
    _rule(
        "transpose_t",
        "cpp",
        r"\.t\(\)",
        "Conv1D↔Linear 换形：`(in,out).t()` → `(out,in)`；第 10 课 remap 的核心动作",
    ),
    _rule(
        "chunk",
        "py",
        r"\.chunk\(",
        "把融合的 `c_attn` 拆成 Q|K|V；C++ 同名，返回 vector",
    ),
    _rule(
        "chunk",
        "cpp",
        r"\.chunk\(",
        "把融合的 `c_attn` 拆成 Q|K|V；返回 std::vector<Tensor>",
    ),
    # --- distributed ---
    _rule(
        "ddp",
        "py",
        r"\bDistributedDataParallel\b",
        "Python 用 DDP 包装模块：backward 后自动 allreduce 梯度；C++ 课里没有同名类，需手写 ProcessGroupGloo::allreduce",
    ),
    _rule(
        "process_group",
        "py",
        r"init_process_group\(",
        "Python：dist.init_process_group(backend, init_method, rank, world_size)；本课用 gloo + file:// FileStore",
    ),
    _rule(
        "process_group",
        "cpp",
        r"\bProcessGroupGloo\b",
        "C++：c10d::ProcessGroupGloo(store, rank, size, options)；FileStore 做 rendezvous，createDefaultDevice() 绑 loopback",
    ),
    _rule(
        "allreduce",
        "cpp",
        r"\.allreduce\(",
        "手写梯度同步：allreduce(SUM) 再 /= world_size，等价于 DDP 的梯度平均",
    ),
    # --- tensor parallel ---
    _rule(
        "functional_linear",
        "py",
        r"\bF\.linear\(",
        "第 12 课：无状态线性；分片权重直接喂给 F.linear。C++ 是 F::linear（torch::nn::functional）",
    ),
    _rule(
        "functional_linear",
        "cpp",
        r"\bF::linear\(",
        "torch::nn::functional::linear；本课用命名空间别名 F，对应 Python torch.nn.functional as F",
    ),
    _rule(
        "stack_sum",
        "py",
        r"torch\.stack\(",
        "教学版 all-reduce：先 stack 再 sum；真分布式会走 ProcessGroup allreduce",
    ),
    _rule(
        "stack_sum",
        "cpp",
        r"torch::stack\(",
        "教学版 all-reduce：torch::stack 再 .sum(0)；真分布式会走 ProcessGroupGloo::allreduce",
    ),
)
