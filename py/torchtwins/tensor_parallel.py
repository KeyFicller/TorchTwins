"""Lesson 12 - tensor_parallel: column/row-parallel Linear shard math (single-process)."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812

from torchtwins import report
from torchtwins.config import Preset
from torchtwins.modules import GELU

# Tiny, deterministic demo. Not the GPT preset. world_size must divide out/in/hidden.
WORLD_SIZE = 2
BATCH = 2
SEQ = 3
IN_FEATURES = 4
OUT_FEATURES = 4
HIDDEN = 8


# @twin shard_size
def shard_size(dim: int, world_size: int) -> int:
    # @step check: dim must divide evenly across ranks
    if dim % world_size != 0:
        raise ValueError(f"dim {dim} not divisible by world_size {world_size}")
    return dim // world_size


# @twin column_weight_shard
def column_weight_shard(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    # @step slice: Linear weight is [out, in]; column-parallel shards along out (dim 0)
    # Contiguous rows per rank so local F.linear sees a dense [out/ws, in] shard.
    rows = shard_size(weight.size(0), world_size)
    start = rank * rows
    return weight[start : start + rows].contiguous()


# @twin row_weight_shard
def row_weight_shard(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    # @step slice: row-parallel shards along in (dim 1) → [out, in/ws]
    cols = shard_size(weight.size(1), world_size)
    start = rank * cols
    return weight[:, start : start + cols].contiguous()


# @twin bias_shard
def bias_shard(bias: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    # @step slice: column-parallel bias shards with the out dim
    rows = shard_size(bias.size(0), world_size)
    start = rank * rows
    return bias[start : start + rows].contiguous()


# @twin input_feature_shard
def input_feature_shard(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    # @step slice: row-parallel needs matching input feature shards on the last dim
    cols = shard_size(x.size(-1), world_size)
    start = rank * cols
    return x[..., start : start + cols].contiguous()


# @twin all_gather_cat
def all_gather_cat(pieces: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    # @step gather: concatenate per-rank outputs (simulates collective all-gather)
    return torch.cat(pieces, dim=dim)


# @twin all_reduce_sum
def all_reduce_sum(pieces: list[torch.Tensor]) -> torch.Tensor:
    # @step reduce: sum partial outputs (simulates collective all-reduce SUM)
    stacked = torch.stack(pieces, dim=0)
    return stacked.sum(dim=0)


# @twin column_parallel_linear
def column_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    # @step shards: each rank multiplies full x by its out-shard (+ bias shard)
    pieces: list[torch.Tensor] = []
    for rank in range(world_size):
        w = column_weight_shard(weight, rank, world_size)
        b = bias_shard(bias, rank, world_size) if bias is not None else None
        pieces.append(F.linear(x, w, b))
    # @step gather: all-gather along the feature dim restores the dense output
    return all_gather_cat(pieces, dim=-1)


# @twin row_parallel_linear
def row_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    # @step shards: shard x on the last dim; each rank uses matching in-shard of W
    pieces: list[torch.Tensor] = []
    for rank in range(world_size):
        x_r = input_feature_shard(x, rank, world_size)
        w = row_weight_shard(weight, rank, world_size)
        # Bias is applied once after the reduce, not per shard.
        pieces.append(F.linear(x_r, w, None))
    # @step reduce: all-reduce SUM reconstitutes the dense matmul
    out = all_reduce_sum(pieces)
    # @step bias: add full bias after the reduce (Megatron row-parallel convention)
    if bias is not None:
        out = out + bias
    return out


# @twin dense_mlp
def dense_mlp(
    x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor
) -> torch.Tensor:
    # @step forward: Linear → GELU → Linear (reference, no sharding)
    return F.linear(GELU()(F.linear(x, w1, b1)), w2, b2)


# @twin tp_mlp
def tp_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    # @step up: column-parallel expansion (Megatron MLP first GEMM)
    h = column_parallel_linear(x, w1, b1, world_size)
    # @step act: GELU on the gathered activation (same as dense after gather)
    h = GELU()(h)
    # @step down: row-parallel projection (Megatron MLP second GEMM)
    return row_parallel_linear(h, w2, b2, world_size)


# @twin max_abs_diff
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # @step diff: max |a-b| used to prove TP matches dense
    return float((a - b).abs().max().item())


# @twin run_tensor_parallel
def run_tensor_parallel(preset: Preset) -> None:
    # @step banner: world size and toy shapes for the demo
    report.section("tensor parallel")
    report.count("world size", WORLD_SIZE)
    report.count("batch", BATCH)
    report.count("seq", SEQ)
    report.count("in features", IN_FEATURES)
    report.count("out features", OUT_FEATURES)
    report.count("hidden", HIDDEN)
    report.text("collectives", "simulated (cat / sum)")

    # @step init: same seed → same dense weights on both runtimes
    torch.manual_seed(preset.seed)
    x = torch.randn(BATCH, SEQ, IN_FEATURES)
    w = torch.randn(OUT_FEATURES, IN_FEATURES)
    b = torch.randn(OUT_FEATURES)
    w1 = torch.randn(HIDDEN, IN_FEATURES)
    b1 = torch.randn(HIDDEN)
    w2 = torch.randn(OUT_FEATURES, HIDDEN)
    b2 = torch.randn(OUT_FEATURES)

    # @step dense: full Linear reference
    report.section("dense linear")
    dense = F.linear(x, w, b)
    report.show("output", dense)

    # @step column: column-parallel + all-gather must match dense
    report.section("column parallel")
    report.count("shard out", shard_size(OUT_FEATURES, WORLD_SIZE))
    report.text("all gather", "cat along last dim")
    col = column_parallel_linear(x, w, b, WORLD_SIZE)
    report.show("output", col)
    report.scalar("max abs diff vs dense", max_abs_diff(col, dense))

    # @step row: row-parallel + all-reduce must match dense
    report.section("row parallel")
    report.count("shard in", shard_size(IN_FEATURES, WORLD_SIZE))
    report.text("all reduce", "sum partials")
    row = row_parallel_linear(x, w, b, WORLD_SIZE)
    report.show("output", row)
    report.scalar("max abs diff vs dense", max_abs_diff(row, dense))

    # @step mlp: Megatron-style column → GELU → row must match dense MLP
    report.section("mlp (column then row)")
    report.count("hidden", HIDDEN)
    dense_h = dense_mlp(x, w1, b1, w2, b2)
    tp_h = tp_mlp(x, w1, b1, w2, b2, WORLD_SIZE)
    report.show("dense mlp", dense_h)
    report.show("tp mlp", tp_h)
    report.scalar("max abs diff vs dense", max_abs_diff(tp_h, dense_h))
