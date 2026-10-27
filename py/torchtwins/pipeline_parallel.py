"""Lesson 13 - pipeline_parallel: stage split + micro-batch handoff (single-process)."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812

from torchtwins import report
from torchtwins.config import Preset
from torchtwins.modules import GELU

# Tiny, deterministic demo. Not the GPT preset.
# 2 stages x 2 Linear(+GELU) blocks; batch splits into micro-batches.
NUM_STAGES = 2
NUM_MICROBATCHES = 4
BATCH = 8
SEQ = 2
FEATURES = 4


# @twin layer_forward
def layer_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # @step linear: Linear then GELU (one pipeline "block")
    out: torch.Tensor = GELU()(F.linear(x, weight, bias))
    return out


# @twin stage_forward
def stage_forward(
    x: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]
) -> torch.Tensor:
    # @step layers: run this stage's blocks in order (local compute, no peer yet)
    h = x
    for weight, bias in zip(weights, biases, strict=True):
        h = layer_forward(h, weight, bias)
    return h


# @twin send_activation
def send_activation(activation: torch.Tensor) -> torch.Tensor:
    # @step handoff: teaching stand-in for P2P send (identity on one process)
    return activation.contiguous()


# @twin recv_activation
def recv_activation(activation: torch.Tensor) -> torch.Tensor:
    # @step handoff: teaching stand-in for P2P recv (identity on one process)
    return activation.contiguous()


# @twin split_microbatches
def split_microbatches(x: torch.Tensor, num_microbatches: int) -> list[torch.Tensor]:
    # @step split: chunk batch dim into equal micro-batches (GPipe / 1F1B input)
    if x.size(0) % num_microbatches != 0:
        raise ValueError(f"batch {x.size(0)} not divisible by num_microbatches {num_microbatches}")
    return list(x.chunk(num_microbatches, dim=0))


# @twin join_microbatches
def join_microbatches(pieces: list[torch.Tensor]) -> torch.Tensor:
    # @step join: concatenate micro-batch outputs back to full batch order
    return torch.cat(pieces, dim=0)


# @twin dense_forward
def dense_forward(
    x: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]
) -> torch.Tensor:
    # @step forward: all layers on one device (reference, no pipeline)
    return stage_forward(x, weights, biases)


# @twin pipeline_forward
def pipeline_forward(
    x: torch.Tensor,
    stage_weights: list[list[torch.Tensor]],
    stage_biases: list[list[torch.Tensor]],
) -> torch.Tensor:
    # @step stage0: first stage owns early layers
    h = stage_forward(x, stage_weights[0], stage_biases[0])
    # @step p2p: send/recv boundary activation between stages
    h = recv_activation(send_activation(h))
    # @step stage1: second stage finishes the stack
    return stage_forward(h, stage_weights[1], stage_biases[1])


# @twin pipeline_microbatch_forward
def pipeline_microbatch_forward(
    x: torch.Tensor,
    stage_weights: list[list[torch.Tensor]],
    stage_biases: list[list[torch.Tensor]],
    num_microbatches: int,
) -> torch.Tensor:
    # @step split: cut the batch into micro-batches
    micros = split_microbatches(x, num_microbatches)
    # @step pipe: each micro-batch walks every stage (activation handoff between stages)
    outs: list[torch.Tensor] = []
    for micro in micros:
        outs.append(pipeline_forward(micro, stage_weights, stage_biases))
    # @step join: restore full-batch output order
    return join_microbatches(outs)


# @twin schedule_1f1b
def schedule_1f1b(num_stages: int, num_microbatches: int) -> list[str]:
    # @step build: warm-up forwards, steady 1F1B units, cool-down backwards
    events: list[str] = []
    # Warm-up: push the first (num_stages - 1) micro-batches through every stage.
    for mb in range(num_stages - 1):
        for stage in range(num_stages):
            events.append(f"F mb{mb} stage{stage}")
    # Steady: for each remaining micro-batch, forward the pipe then backward the oldest in-flight.
    for mb in range(num_stages - 1, num_microbatches):
        for stage in range(num_stages):
            events.append(f"F mb{mb} stage{stage}")
        done = mb - (num_stages - 1)
        for stage in range(num_stages - 1, -1, -1):
            events.append(f"B mb{done} stage{stage}")
    # Cool-down: drain backwards for micro-batches still unfinished after the last forward.
    for mb in range(num_microbatches - (num_stages - 1), num_microbatches):
        for stage in range(num_stages - 1, -1, -1):
            events.append(f"B mb{mb} stage{stage}")
    return events


# @twin max_abs_diff
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # @step diff: max |a-b| used to prove pipeline matches dense
    return float((a - b).abs().max().item())


# @twin run_pipeline_parallel
def run_pipeline_parallel(preset: Preset) -> None:
    # @step banner: stages, micro-batches, toy shapes
    report.section("pipeline parallel")
    report.count("num stages", NUM_STAGES)
    report.count("num microbatches", NUM_MICROBATCHES)
    report.count("batch", BATCH)
    report.count("seq", SEQ)
    report.count("features", FEATURES)
    report.text("p2p", "simulated (identity send/recv)")
    report.text("schedule", "1F1B labels (forward numeric parity uses micro-batch pipe)")

    # @step init: same seed → same layer weights on both runtimes
    torch.manual_seed(preset.seed)
    x = torch.randn(BATCH, SEQ, FEATURES)
    # 4 layers total → 2 per stage
    weights = [torch.randn(FEATURES, FEATURES) for _ in range(4)]
    biases = [torch.randn(FEATURES) for _ in range(4)]
    stage_weights = [weights[0:2], weights[2:4]]
    stage_biases = [biases[0:2], biases[2:4]]

    # @step dense: full stack reference
    report.section("dense forward")
    dense = dense_forward(x, weights, biases)
    report.show("output", dense)

    # @step sequential: one full-batch activation handoff across stages
    report.section("pipeline sequential")
    report.text("handoff", "stage0 → send/recv → stage1")
    seq = pipeline_forward(x, stage_weights, stage_biases)
    report.show("output", seq)
    report.scalar("max abs diff vs dense", max_abs_diff(seq, dense))

    # @step microbatch: GPipe-style forward over micro-batches, same math
    report.section("pipeline microbatch")
    report.count("micro batch size", BATCH // NUM_MICROBATCHES)
    micro = pipeline_microbatch_forward(x, stage_weights, stage_biases, NUM_MICROBATCHES)
    report.show("output", micro)
    report.scalar("max abs diff vs dense", max_abs_diff(micro, dense))

    # @step schedule: print 1F1B event list for the same shape (teaching transcript)
    report.section("1f1b schedule")
    events = schedule_1f1b(NUM_STAGES, NUM_MICROBATCHES)
    report.count("events", len(events))
    for i, event in enumerate(events):
        report.text(f"event {i:02d}", event)
