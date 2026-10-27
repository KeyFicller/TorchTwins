"""Check that every Python module has a C++ twin with the same ``@twin`` blocks.

    python tools/check_twins.py

A twin pair is ``py/torchtwins/<name>.py`` <-> ``cpp/torchtwins/<name>.{h,cpp}``.
Each side marks comparable blocks with ``# @twin <block>`` / ``// @twin <block>``;
the set of block names must be identical and appear in the same order.
``// @endtwin`` ends a block early (e.g. to keep trailing declarations out of it).

Inside a block, ``# @step <id>: <note>`` / ``// @step <id>`` anchors split the
code into logical steps. Both sides must list the same step ids in the same order
(the C++ side reads the header first, then the source), every lesson/infra block
must have at least one step, and every Python anchor carries an English note.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_DIR = REPO_ROOT / "py" / "torchtwins"
CPP_DIR = REPO_ROOT / "cpp" / "torchtwins"
PY_MARKER = re.compile(r"^\s*# @twin (\w+)\s*$")
CPP_MARKER = re.compile(r"^\s*// @twin (\w+)\s*$")
END_MARKER = re.compile(r"^\s*(#|//) @endtwin\s*$")
PY_STEP = re.compile(r"^\s*# @step (\w+)(?::\s*(.*?))?\s*$")
CPP_STEP = re.compile(r"^\s*// @step (\w+)(?::\s*(.*?))?\s*$")
# Every twin pair with lesson-style @step anchors. Includes infra (report/config/tokenizer).
LESSONS = (
    "report",
    "config",
    "tokenizer",
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
)
# Entry points have no block-level twin: they differ by nature (argparse vs argv).
UNPAIRED = {"__init__", "__main__"}


@dataclass(frozen=True)
class Block:
    name: str
    lines: list[str]


def read_blocks(path: Path, marker: re.Pattern[str]) -> list[Block]:
    """Split a file into ``@twin`` blocks; a block runs until the next marker."""
    blocks: list[Block] = []
    current: tuple[str, list[str]] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = marker.match(line)
        if match or END_MARKER.match(line):
            if current:
                blocks.append(Block(current[0], current[1]))
            current = (match.group(1), []) if match else None
        elif current is not None:
            current[1].append(line)
    if current:
        blocks.append(Block(current[0], current[1]))
    return blocks


def twin_names() -> list[str]:
    return sorted(p.stem for p in PY_DIR.glob("*.py") if p.stem not in UNPAIRED)


def cpp_blocks(name: str) -> list[Block]:
    """Header blocks first (declarations), then source blocks; merged by name."""
    merged: dict[str, list[str]] = {}
    for suffix in (".h", ".cpp"):
        path = CPP_DIR / f"{name}{suffix}"
        if path.exists():
            for block in read_blocks(path, CPP_MARKER):
                merged.setdefault(block.name, []).extend(block.lines)
    return [Block(k, v) for k, v in merged.items()]


def order_in_source(name: str) -> list[str]:
    """Block order as it appears in the .cpp (falls back to .h for header-only twins)."""
    source = CPP_DIR / f"{name}.cpp"
    path = source if source.exists() else CPP_DIR / f"{name}.h"
    return [b.name for b in read_blocks(path, CPP_MARKER)]


def steps(lines: list[str], marker: re.Pattern[str]) -> list[tuple[str, str]]:
    """``(id, note)`` for every ``@step`` anchor in a block, in order."""
    return [(m.group(1), m.group(2) or "") for m in map(marker.match, lines) if m]


def check_steps(name: str, py_file: Path) -> list[str]:
    errors: list[str] = []
    cpp = {b.name: b.lines for b in cpp_blocks(name)}
    for block in read_blocks(py_file, PY_MARKER):
        where = f"{name}.{block.name}"
        py_steps = steps(block.lines, PY_STEP)
        cpp_ids = [s for s, _ in steps(cpp.get(block.name, []), CPP_STEP)]
        py_ids = [s for s, _ in py_steps]
        if name in LESSONS and not py_ids:
            errors.append(f"{where}: lesson block has no '# @step' anchors")
        if py_ids != cpp_ids:
            errors.append(f"{where}: step ids differ: py={py_ids} cpp={cpp_ids}")
        if len(set(py_ids)) != len(py_ids):
            errors.append(f"{where}: duplicate step ids {py_ids}")
        errors.extend(
            f"{where}.{sid}: Python anchor needs a note ('# @step {sid}: english note')"
            for sid, note in py_steps
            if not note
        )
    return errors


def check() -> list[str]:
    errors: list[str] = []
    for name in twin_names():
        if not (CPP_DIR / f"{name}.h").exists():
            errors.append(f"{name}: missing cpp/torchtwins/{name}.h")
            continue
        py = [b.name for b in read_blocks(PY_DIR / f"{name}.py", PY_MARKER)]
        cpp = [b.name for b in cpp_blocks(name)]
        if not py:
            errors.append(f"{name}: no '# @twin' blocks in Python")
        if set(py) != set(cpp):
            errors.append(
                f"{name}: only in Python {sorted(set(py) - set(cpp))}, "
                f"only in C++ {sorted(set(cpp) - set(py))}"
            )
        elif [n for n in py if n in order_in_source(name)] != order_in_source(name):
            errors.append(f"{name}: block order differs: py={py} cpp={order_in_source(name)}")
        else:
            errors.extend(check_steps(name, PY_DIR / f"{name}.py"))
    for path in CPP_DIR.glob("*.h"):
        if path.stem not in twin_names() and path.stem != "main":
            errors.append(f"{path.stem}: C++ file without Python twin")
    return errors


def main() -> int:
    errors = check()
    for error in errors:
        print(f"[twins] {error}")
    print(f"[twins] {len(twin_names())} twin pairs, {'OK' if not errors else 'FAILED'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
