"""Lesson 11 - distributed: gloo process group, DDP vs manual allreduce grads."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from torchtwins import report
from torchtwins.config import Preset

# torch stubs omit multiprocessing.spawn; cast keeps mypy --strict happy.
_SpawnFn = Callable[..., None]
_mp_spawn_impl = cast(_SpawnFn, torch_mp.spawn)  # type: ignore[attr-defined]


def _mp_spawn(
    fn: Callable[..., None],
    *,
    args: tuple[Any, ...],
    nprocs: int,
    join: bool = True,
) -> None:
    _mp_spawn_impl(fn, args=args, nprocs=nprocs, join=join)


# Tiny, deterministic demo: 4 samples / 2 ranks, few SGD steps. Not the GPT preset.
WORLD_SIZE = 2
STEPS = 5
LR = 0.25
IN_FEATURES = 2


# @twin make_xy
def make_xy() -> tuple[torch.Tensor, torch.Tensor]:
    # @step data: fixed 4x2 features and linear targets
    # Target is 2*x0 + 3*x1 so a bias-free Linear(2, 1) can fit it.
    x = torch.tensor(
        [
            [0.5, 1.0],
            [1.0, 0.5],
            [0.0, 1.5],
            [1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    y = 2.0 * x[:, 0:1] + 3.0 * x[:, 1:2]
    return x, y


# @twin new_model
def new_model(seed: int) -> nn.Linear:
    # @step init: zero weights, same seed on every rank
    torch.manual_seed(seed)
    model = nn.Linear(IN_FEATURES, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


# @twin train_single
def train_single(seed: int) -> tuple[list[float], torch.Tensor]:
    # @step setup: full-batch baseline on one process
    model = new_model(seed)
    x, y = make_xy()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    losses: list[float] = []
    # @step loop: SGD steps on the full batch
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses, model.weight.detach().clone()


# @twin shard
def shard(
    x: torch.Tensor, y: torch.Tensor, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # @step slice: contiguous rows per rank
    # world_size must divide the sample count so every rank sees the same local batch size.
    n = x.size(0) // world_size
    start = rank * n
    end = start + n
    return x[start:end], y[start:end]


# @twin allreduce_avg_
def allreduce_avg_(tensor: torch.Tensor) -> torch.Tensor:
    # @step avg: SUM allreduce then divide (matches DDP grad averaging)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


# @twin train_ddp_rank
def train_ddp_rank(
    rank: int, world_size: int, seed: int, store_file: str
) -> tuple[list[float], torch.Tensor] | None:
    # @step group: init gloo via FileStore (no TCP port races in tests)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_file}",
        rank=rank,
        world_size=world_size,
    )
    # @step wrap: DistributedDataParallel allreduces grads after backward
    model = new_model(seed)
    ddp = DistributedDataParallel(model)
    x, y = make_xy()
    local_x, local_y = shard(x, y, rank, world_size)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=LR)
    losses: list[float] = []
    # @step loop: local forward, report global mean loss, DDP backward
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = ((ddp(local_x) - local_y) ** 2).mean()
        loss_report = loss.detach().clone()
        allreduce_avg_(loss_report)
        loss.backward()
        optimizer.step()
        losses.append(float(loss_report))
    weight = model.weight.detach().clone()
    dist.destroy_process_group()
    if rank == 0:
        return losses, weight
    return None


def _ddp_worker(rank: int, world_size: int, seed: int, store_file: str) -> None:
    """spawn entry: only rank 0 prints into the shared transcript."""
    result = train_ddp_rank(rank, world_size, seed, store_file)
    if result is None:
        return
    losses, weight = result
    report.count("rank", 0)
    report.count("world size", world_size)
    report.text("grad sync", "allreduce")
    for step, loss in enumerate(losses, start=1):
        report.text(f"step {step:04d}", f"loss {loss:.4f}")
    report.scalar("final loss", losses[-1])
    report.show("weight", weight)


# @twin run_distributed
def run_distributed(preset: Preset) -> None:
    # @step banner: backend and world size for the demo
    report.section("process group")
    report.text("backend", "gloo")
    report.count("world size", WORLD_SIZE)
    report.count("steps", STEPS)

    # @step single: one-process baseline (same data, same steps)
    report.section("single process")
    single_losses, single_weight = train_single(preset.seed)
    report.text("grad sync", "none")
    for step, loss in enumerate(single_losses, start=1):
        report.text(f"step {step:04d}", f"loss {loss:.4f}")
    report.scalar("final loss", single_losses[-1])
    report.show("weight", single_weight)

    # @step ddp: spawn WORLD_SIZE CPU ranks; rank 0 continues the transcript
    report.section("distributed (ddp)")
    fd, store_file = tempfile.mkstemp(prefix="torchtwins_dist_")
    os.close(fd)
    os.unlink(store_file)  # FileStore wants a path that does not exist yet
    try:
        _mp_spawn(
            _ddp_worker,
            args=(WORLD_SIZE, preset.seed, store_file),
            nprocs=WORLD_SIZE,
            join=True,
        )
    finally:
        Path(store_file).unlink(missing_ok=True)
