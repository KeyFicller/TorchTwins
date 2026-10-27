"""Gate: every lesson prints the same transcript in Python and C++."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from check_twins import check
from compare_outputs import compare_lines

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_CLI = REPO_ROOT / "build" / "bin" / "torchtwins"
LESSONS = [
    "tensors",
    "autograd",
    "modules",
    "attention",
    "gpt",
    "data",
    "train",
    "finetune",
    "checkpoint",
    "crossload",
    "remap",
    "distributed",
    "tensor_parallel",
]

needs_cpp = pytest.mark.skipif(not CPP_CLI.exists(), reason="C++ side not built")


def run_python(*args: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", "torchtwins", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def run_cpp(*args: str) -> str:
    return subprocess.run(
        [str(CPP_CLI), *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


@needs_cpp
@pytest.mark.parametrize("lesson", LESSONS)
@pytest.mark.parametrize("seed", ["123", "7"])
def test_lesson_transcripts_match(lesson: str, seed: str) -> None:
    py = run_python(lesson, "--seed", seed).splitlines()
    cpp = run_cpp(lesson, "--seed", seed).splitlines()
    assert py, "empty transcript"
    mismatches = compare_lines(py, cpp, atol=1e-4)
    assert not mismatches, "\n".join(f"{m.line}: {m.python} | {m.cpp}" for m in mismatches)


@needs_cpp
def test_tokenizers_agree_on_the_verdict() -> None:
    path = "assets/the-verdict.txt"
    assert run_python("tokenize", path) == run_cpp("tokenize", path)


def test_lesson_lists_match() -> None:
    py = run_python("list")
    assert all(lesson in py for lesson in LESSONS)
    if CPP_CLI.exists():
        assert py == run_cpp("list")


def test_twins_are_complete() -> None:
    assert check() == []
