"""CLI - mirrors ``cpp/torchtwins/main.cpp``.

python -m torchtwins <lesson> [--preset tiny|small|gpt2] [--seed N] [--curve FILE]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from torchtwins.attention import run_attention
from torchtwins.autograd import run_autograd
from torchtwins.checkpoint import run_checkpoint
from torchtwins.config import Preset, load_preset
from torchtwins.crossload import run_crossload
from torchtwins.data import run_data
from torchtwins.distributed import run_distributed
from torchtwins.finetune import run_finetune
from torchtwins.gpt import run_gpt
from torchtwins.modules import run_modules
from torchtwins.remap import run_remap
from torchtwins.tensor_parallel import run_tensor_parallel
from torchtwins.tensors import run_tensors
from torchtwins.tokenizer import Tokenizer
from torchtwins.train import run_train

LESSONS: dict[str, tuple[str, Callable[[Preset], None]]] = {
    "tensors": ("01 creation, shapes, indexing, broadcasting", run_tensors),
    "autograd": ("02 gradients, no_grad, hand-written SGD", run_autograd),
    "modules": ("03 GELU, LayerNorm, FeedForward", run_modules),
    "attention": ("04 causal multi-head self-attention", run_attention),
    "gpt": ("05 GPT model and greedy generation", run_gpt),
    "data": ("06 tokens, train/val split, windows, batches", run_data),
    "train": ("06 pre-training with AdamW, loss curves", run_train),
    "finetune": ("07 spam classifier: new head, frozen body", run_finetune),
    "checkpoint": ("08 save/load weights, restore logits", run_checkpoint),
    "crossload": ("09 portable pickle bridge across runtimes", run_crossload),
    "remap": ("10 HF/OpenAI GPT-2 key remap (Conv1D, c_attn)", run_remap),
    "distributed": ("11 gloo DDP vs manual allreduce grads", run_distributed),
    "tensor_parallel": ("12 column/row-parallel Linear shard math", run_tensor_parallel),
}


def tokenize_file(path: Path) -> None:
    ids = Tokenizer().encode(path.read_text(encoding="utf-8"))
    print(len(ids))
    print(" ".join(str(i) for i in ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="torchtwins", description=__doc__)
    parser.add_argument("lesson", choices=[*LESSONS, "list", "tokenize"])
    parser.add_argument("path", nargs="?", type=Path, help="file for 'tokenize'")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--curve", type=Path, default=None, help="train: write loss curve JSON")
    args = parser.parse_args(argv)

    if args.lesson == "list":
        for name, (description, _) in LESSONS.items():
            print(f"{name:<10} {description}")
        return 0
    if args.lesson == "tokenize":
        if args.path is None:
            parser.error("tokenize needs a file path")
        tokenize_file(args.path)
        return 0

    preset = load_preset(args.preset)
    if args.seed is not None:
        preset = preset.with_seed(args.seed)
    if args.lesson == "train":
        run_train(preset, args.curve)
    else:
        LESSONS[args.lesson][1](preset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
