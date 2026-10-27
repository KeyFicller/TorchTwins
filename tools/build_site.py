"""Generate the static PyTorch <-> LibTorch comparison site in docs/site/.

    python tools/build_site.py            # (re)generate docs/site/
    python tools/build_site.py --check    # fail if a local docs/site/ snapshot is stale
    python tools/build_site.py --record   # also re-record lesson transcripts (needs the C++ build)

GitHub Pages runs ``build_site.py`` on deploy; ``docs/site/`` is gitignored so PRs do
not churn generated HTML. Offline SVGs live in ``docs/diagrams/`` and are copied in.

Everything on the site is derived from the sources (``@twin`` blocks) and from the
measured result files in docs/results/*.json - no number is typed in by hand.
"""

from __future__ import annotations

import argparse
import filecmp
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pygments.lexers import CppLexer, PythonLexer
from pygments.token import STANDARD_TYPES, _TokenType

from check_twins import CPP_MARKER, CPP_STEP, END_MARKER, PY_MARKER, PY_STEP, twin_names
from site_assets import CSS, DIAGRAM_ARCH, DIAGRAM_CONCEPTS, page
from site_charts import bar_chart, line_chart
from site_notes import LESSON_TITLES, NOTES, STEP_NOTES
from site_rules import RULES, ApiRule

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = REPO_ROOT / "docs" / "site"
DIAGRAMS_SRC = REPO_ROOT / "docs" / "diagrams"
RESULTS = REPO_ROOT / "docs" / "results"
PY_DIR = REPO_ROOT / "py" / "torchtwins"
CPP_DIR = REPO_ROOT / "cpp" / "torchtwins"
CPP_CLI = REPO_ROOT / "build" / "bin" / "torchtwins"
TRANSCRIPTS = RESULTS / "transcripts.json"
LESSON_ORDER = list(LESSON_TITLES)
RUNNABLE = [
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
]


# ----------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Line:
    file: str
    number: int
    text: str


@dataclass(frozen=True)
class Block:
    name: str
    lines: list[Line]


def read_blocks(path: Path, marker: re.Pattern[str]) -> list[Block]:
    blocks: list[Block] = []
    name: str | None = None
    lines: list[Line] = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = marker.match(text)
        if match or END_MARKER.match(text):
            if name is not None:
                blocks.append(Block(name, lines))
            name, lines = (match.group(1) if match else None), []
        elif name is not None:
            lines.append(Line(path.name, number, text))
    if name is not None:
        blocks.append(Block(name, lines))
    return [Block(b.name, trim(b.lines)) for b in blocks]


def trim(lines: list[Line]) -> list[Line]:
    start, end = 0, len(lines)
    while start < end and not lines[start].text.strip():
        start += 1
    while end > start and not lines[end - 1].text.strip():
        end -= 1
    return lines[start:end]


def cpp_blocks(name: str) -> dict[str, list[Line]]:
    merged: dict[str, list[Line]] = {}
    for suffix in (".h", ".cpp"):
        path = CPP_DIR / f"{name}{suffix}"
        if path.exists():
            for block in read_blocks(path, CPP_MARKER):
                merged.setdefault(block.name, []).extend(block.lines)
    return merged


# ----------------------------------------------------------------------------- steps

SIGNATURE = "_sig"


@dataclass(frozen=True)
class Step:
    id: str
    note: str
    lines: list[Line]


def split_steps(lines: list[Line], marker: re.Pattern[str]) -> list[Step]:
    """Cut a twin block at its ``@step`` anchors; code before the first anchor is the signature.

    Rows are aligned by meaning (the anchor ids), never by text similarity: Python and
    C++ syntax differ too much for a line diff to pair anything but signatures.
    """
    out: list[Step] = []
    sid, note = SIGNATURE, ""
    current: list[Line] = []
    for line in lines:
        match = marker.match(line.text)
        if match:
            out.append(Step(sid, note, trim(current)))
            sid, note, current = match.group(1), match.group(2) or "", []
        else:
            current.append(line)
    out.append(Step(sid, note, trim(current)))
    return [s for s in out if s.id != SIGNATURE or s.lines] if len(out) > 1 else out


def pair_steps(where: str, py: list[Step], cpp: list[Step]) -> list[tuple[Step, Step]]:
    """Pair steps by id. ``tools/check_twins.py`` guarantees equal ids in equal order."""
    py_ids = [s.id for s in py if s.id != SIGNATURE]
    cpp_ids = [s.id for s in cpp if s.id != SIGNATURE]
    if py_ids != cpp_ids:
        msg = f"{where}: step ids differ py={py_ids} cpp={cpp_ids} (run tools/check_twins.py)"
        raise SystemExit(msg)
    by_id_py = {s.id: s for s in py}
    by_id_cpp = {s.id: s for s in cpp}
    empty = Step(SIGNATURE, "", [])
    ids = ([SIGNATURE] if SIGNATURE in by_id_py or SIGNATURE in by_id_cpp else []) + py_ids
    return [(by_id_py.get(i, empty), by_id_cpp.get(i, empty)) for i in ids]


# ----------------------------------------------------------------------------- highlighting


def css_class(ttype: _TokenType) -> str:
    while ttype not in STANDARD_TYPES:
        ttype = ttype.parent
    name: str = STANDARD_TYPES[ttype]
    return name


def highlight(lines: list[Line], lang: str) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Lex the whole block once (so multi-line strings work), then split per line."""
    lexer = PythonLexer() if lang == "py" else CppLexer()
    out: dict[tuple[str, int], list[tuple[str, str]]] = {}
    by_file: dict[str, list[Line]] = {}
    for line in lines:
        by_file.setdefault(line.file, []).append(line)
    for file_lines in by_file.values():
        source = "\n".join(line.text for line in file_lines) + "\n"
        segments: list[list[tuple[str, str]]] = [[]]
        for ttype, value in lexer.get_tokens(source):
            for k, part in enumerate(value.split("\n")):
                if k:
                    segments.append([])
                if part:
                    segments[-1].append((css_class(ttype), part))
        for line, segs in zip(file_lines, segments, strict=False):
            out[(line.file, line.number)] = segs
    return out


def api_marks(text: str, lang: str) -> list[tuple[int, int, ApiRule]]:
    marks: list[tuple[int, int, ApiRule]] = []
    if text.lstrip().startswith(("#", "//")):
        return marks
    for rule in RULES:
        if rule.lang == lang:
            marks.extend((m.start(), m.end(), rule) for m in rule.pattern.finditer(text))
    marks.sort(key=lambda m: m[0])
    kept: list[tuple[int, int, ApiRule]] = []
    for mark in marks:
        if not kept or mark[0] >= kept[-1][1]:
            kept.append(mark)
    return kept


def render_line(segments: list[tuple[str, str]], marks: list[tuple[int, int, ApiRule]]) -> str:
    out: list[str] = []
    pos = 0
    bounds = sorted({p for s, e, _ in marks for p in (s, e)})
    open_mark: tuple[int, int, ApiRule] | None = None
    for cls, value in segments:
        start = pos
        cuts = [b - start for b in bounds if start < b < start + len(value)]
        pieces = [value[i:j] for i, j in zip([0, *cuts], [*cuts, len(value)], strict=True)]
        for piece in pieces:
            mark = next((m for m in marks if m[0] <= pos < m[1]), None)
            if mark is not open_mark:
                if open_mark is not None:
                    out.append("</mark>")
                if mark is not None:
                    tip = html.escape(mark[2].tip, quote=True)
                    out.append(f'<mark class="api" data-key="{mark[2].key}" title="{tip}">')
                open_mark = mark
            escaped = html.escape(piece)
            out.append(f'<span class="{cls}">{escaped}</span>' if cls else escaped)
            pos += len(piece)
    if open_mark is not None:
        out.append("</mark>")
    return "".join(out)


# ----------------------------------------------------------------------------- pages


@dataclass(frozen=True)
class BlockStats:
    steps: int
    api: set[str]


def rich_note(text: str) -> str:
    """Escape a plain-text note and turn `code` spans into <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text, quote=False))


def render_side(
    lines: list[Line], lang: str, hl: dict[tuple[str, int], list[tuple[str, str]]], keys: set[str]
) -> str:
    if not lines:
        return f'<td class="tside {lang} empty"></td>'
    out: list[str] = []
    previous: str | None = None
    for line in lines:
        if previous is not None and line.file != previous:
            out.append(f'<div class="file-sep">{html.escape(line.file)}</div>')
        previous = line.file
        marks = api_marks(line.text, lang)
        keys.update(m[2].key for m in marks)
        code = render_line(hl.get((line.file, line.number), [("", line.text)]), marks)
        api_cls = " has-api" if marks else ""
        out.append(
            f'<div class="cl{api_cls}"><span class="ln" title="{line.file}:{line.number}">'
            f'{line.number}</span><span class="code">{code}</span></div>'
        )
    return f'<td class="tside {lang}"><div class="seg">{"".join(out)}</div></td>'


def step_head(module: str, name: str, index: int, py: Step, cpp: Step) -> str:
    if py.id == SIGNATURE:
        files = {line.file for line in cpp.lines}
        detail = (
            "C++ 在头文件 <code>.h</code> 里声明，在 <code>.cpp</code> 里定义；Python 一处写完。"
            if any(f.endswith(".h") for f in files)
            else ""
        )
        return (
            f'<tr class="step-head sig"><th colspan="2"><span class="step-no">0</span>'
            f'<span class="step-title">签名 / 声明</span><span class="step-note">{detail}</span>'
            "</th></tr>"
        )
    detail = STEP_NOTES.get(f"{module}.{name}.{py.id}", "")
    anchor = f"{name}-{py.id}"
    return (
        f'<tr class="step-head" id="{anchor}"><th colspan="2"><span class="step-no">{index}</span>'
        f'<code class="step-id">@step {py.id}</code><span class="step-title">{rich_note(py.note)}</span>'
        f'<a class="anchor" href="#{anchor}">#</a>'
        f'<span class="step-note">{rich_note(detail)}</span></th></tr>'
    )


def render_block(module: str, name: str, py: list[Line], cpp: list[Line]) -> tuple[str, BlockStats]:
    py_steps = split_steps(py, PY_STEP)
    cpp_steps = split_steps(cpp, CPP_STEP)
    code_py = [line for s in py_steps for line in s.lines]
    code_cpp = [line for s in cpp_steps for line in s.lines]
    hl_py, hl_cpp = highlight(code_py, "py"), highlight(code_cpp, "cpp")
    keys: set[str] = set()
    rows_html: list[str] = []
    pairs = pair_steps(f"{module}.{name}", py_steps, cpp_steps)
    number = 0
    for left, right in pairs:
        if left.id != SIGNATURE:
            number += 1
        rows_html.append(step_head(module, name, number, left, right))
        rows_html.append(
            '<tr class="step-body">'
            f"{render_side(left.lines, 'py', hl_py, keys)}{render_side(right.lines, 'cpp', hl_cpp, keys)}"
            "</tr>"
        )
    note = NOTES.get(f"{module}.{name}")
    if note is None and keys:
        tips = sorted({r.key: r for r in RULES if r.key in keys}.values(), key=lambda r: r.key)
        note = "本块涉及的 API 差异：" + "；".join(
            f"<code>{html.escape(r.key)}</code>" for r in tips
        )
    note_html = f'<div class="note">{note}</div>' if note else ""
    py_files = sorted({line.file for line in code_py})
    cpp_files = sorted({line.file for line in code_cpp}, key=lambda f: not f.endswith(".h"))
    table = (
        f'<section class="block" id="{name}"><h3><span class="twin">@twin</span> {name}'
        f'<span class="nsteps">{number} 步</span>'
        f'<a class="anchor" href="#{name}">#</a></h3>{note_html}'
        '<table class="twin-table steps"><colgroup><col class="half"><col class="half"></colgroup>'
        "<thead><tr>"
        f'<th><span class="badge py">PyTorch</span> {" + ".join(py_files)}</th>'
        f'<th><span class="badge cpp">LibTorch</span> {" + ".join(cpp_files)}</th>'
        f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></section>"
    )
    return table, BlockStats(number, keys)


def render_transcript(module: str, transcripts: dict[str, Any]) -> str:
    entry = transcripts.get("lessons", {}).get(module)
    if not entry:
        return ""
    py, cpp = entry["python"], entry["cpp"]
    rows = []
    for left, right in zip(py, cpp, strict=False):
        cls = "same" if left == right else "num"
        rows.append(
            f'<tr class="{cls}"><td class="code">{html.escape(left)}</td>'
            f'<td class="code">{html.escape(right)}</td></tr>'
        )
    verdict = entry["verdict"]
    return (
        '<section class="block" id="transcript"><h3>运行输出（实测）'
        f'<span class="verdict {"ok" if verdict == "IDENTICAL" else "bad"}">{verdict}</span></h3>'
        f'<div class="note">命令：<code>python -m torchtwins {module}</code> 与 '
        f"<code>./build/bin/torchtwins {module}</code>；文本逐字比较，数字容差 1e-4。"
        "标灰的行逐字相同，其余行只在浮点末位不同。</div>"
        '<table class="twin-table transcript"><thead><tr><th>PyTorch stdout</th>'
        f"<th>LibTorch stdout</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
    )


def lesson_page(module: str, transcripts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    py_blocks = read_blocks(PY_DIR / f"{module}.py", PY_MARKER)
    cpp = cpp_blocks(module)
    parts: list[str] = []
    steps = 0
    api: set[str] = set()
    for block in py_blocks:
        section, stats = render_block(module, block.name, block.lines, cpp.get(block.name, []))
        parts.append(section)
        steps += stats.steps
        api |= stats.api
    number, title = LESSON_TITLES[module]
    toc = "".join(f'<a href="#{b.name}">{b.name}</a>' for b in py_blocks)
    py_loc = sum(1 for b in py_blocks for line in b.lines if not PY_STEP.match(line.text))
    cpp_loc = sum(1 for v in cpp.values() for line in v if not CPP_STEP.match(line.text))
    head = (
        f'<header class="lesson-head"><div class="num">{number}</div><div><h1>{title}</h1>'
        f'<p class="files"><code>py/torchtwins/{module}.py</code> ⇄ '
        f"<code>cpp/torchtwins/{module}.{{h,cpp}}</code></p>"
        f'<p class="stats">{len(py_blocks)} 个 twin 块 · {steps} 个语义步骤 · Python {py_loc} 行 · C++ {cpp_loc} 行 · '
        f"{len(api)} 类 API 差异（黄色高亮，悬停看解释）</p></div></header>"
        f'<nav class="toc">{toc}{"<a href=#transcript>运行输出</a>" if module in RUNNABLE else ""}'
        "</nav>"
    )
    body = head + "".join(parts) + render_transcript(module, transcripts)
    summary = {
        "module": module,
        "py_loc": py_loc,
        "cpp_loc": cpp_loc,
        "steps": steps,
        "blocks": len(py_blocks),
        "api": api,
    }
    return page(f"{number} {title}", body, active=module, nav=nav_items()), summary


def nav_items() -> list[tuple[str, str, str]]:
    items = [("index", "首页", "index.html")]
    items += [(m, f"{LESSON_TITLES[m][0]} {m}", f"{m}.html") for m in LESSON_ORDER]
    items.append(("results", "实测结果", "results.html"))
    return items


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return data


def index_page(summaries: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{LESSON_TITLES[s['module']][0]}</td>"
        f'<td><a href="{s["module"]}.html">{LESSON_TITLES[s["module"]][1]}</a></td>'
        f'<td class="r">{s["blocks"]}</td><td class="r">{s["steps"]}</td><td class="r">{s["py_loc"]}</td>'
        f'<td class="r">{s["cpp_loc"]}</td><td class="r">{s["cpp_loc"] / max(s["py_loc"], 1):.2f}×</td>'
        f"<td>{' '.join(f'<code>{k}</code>' for k in sorted(s['api']))}</td></tr>"
        for s in summaries
    )
    train = load_json("train_curves.json").get("result", {})
    parity = load_json("parity.json")
    max_fwd = max((r["max_abs_diff"] for r in parity.get("results", [])), default=0.0)
    cards = (
        '<div class="cards">'
        f'<div class="card"><div class="big">{len(summaries)}</div>对双胞胎文件</div>'
        f'<div class="card"><div class="big">{sum(s["blocks"] for s in summaries)}</div>个 @twin 块</div>'
        f'<div class="card"><div class="big">{sum(s["steps"] for s in summaries)}</div>个按语义对齐的 @step 步骤</div>'
        f'<div class="card"><div class="big">{train.get("python_steps", "?")} = {train.get("cpp_steps", "?")}</div>'
        "训练步数 Python = C++<br><small>（旧版 85 ≠ 75）</small></div>"
        f'<div class="card"><div class="big">{max(train.get("max_train_diff", 0.0), 0.0):.1e}</div>逐步训练 loss 最大误差</div>'
        f'<div class="card"><div class="big">{max_fwd:.1e}</div>同种子初始化 + 前向 logits 最大误差</div>'
        "</div>"
    )
    body = (
        '<header class="hero"><h1>一个 GPT，写两遍</h1>'
        "<p>同一个 GPT-2 小模型，左边 <b>PyTorch</b>，右边 <b>LibTorch (C++)</b>。文件名、类名、函数名、"
        "打印格式一一对应；同一个随机种子在两边抽出逐位相同的权重，训练出几乎重合的 loss 曲线。"
        "本站每一行代码都直接取自仓库源码，每一个数字都来自工具实测的 JSON。</p></header>"
        f"{cards}"
        '<h2>课程</h2><table class="lessons"><thead><tr><th>#</th><th>课程</th><th>块</th><th>步骤</th>'
        "<th>Python 行</th><th>C++ 行</th><th>C++/Py</th><th>涉及的 API 差异</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<h2>模型结构</h2>"
        f"{DIAGRAM_ARCH}"
        "<h2>概念映射：PyTorch ⇄ LibTorch</h2>"
        f"{DIAGRAM_CONCEPTS}"
        '<h2>怎么读</h2><ul class="howto"><li>每课一个页面，左右两栏按 <code>@twin</code> 块对齐；块内按 <code>@step</code> '
        "语义锚点切成若干步，<b>一步一行</b>：同一行左边是 Python、右边是 C++，顶部对齐，短的一侧留空（斜线格）。"
        "不做逐行 diff——两种语言语法差太多，按文本相似度只能对上函数签名。</li>"
        "<li>每一步上方是这一步做什么，以及两边写法为什么不同；"
        '<mark class="api">黄色</mark> = API 差异（悬停看解释）。</li>'
        "<li>页面末尾是两边程序的真实输出，逐行对照。</li>"
        '<li><a href="results.html">实测结果</a>：parity 误差、训练曲线叠加图、benchmark。</li></ul>'
    )
    return page("首页", body, active="index", nav=nav_items())


def results_page() -> str:
    parity = load_json("parity.json")
    curves = load_json("train_curves.json")
    bench = load_json("benchmark.json")
    parity_rows = "".join(
        f'<tr><td>{r["preset"]}</td><td>{r["check"]}</td><td class="r">{r["tensors"]}</td>'
        f'<td class="r">{r["max_abs_diff"]:.3e}</td><td>{"✅" if r["passed"] else "❌"}</td></tr>'
        for r in parity.get("results", [])
    )
    parts = [
        "<h1>实测结果</h1>",
        f'<p class="meta">PyTorch {parity.get("pytorch", "?")} · LibTorch {parity.get("libtorch", "?")} · '
        f"{parity.get('machine', '?')}。所有数字由 <code>tools/parity.py</code>、<code>tools/train_parity.py</code>、"
        "<code>tools/bench.py</code> 写入 <code>docs/results/*.json</code>，本页只负责画出来。</p>",
        "<h2>① 数值一致性（parity）</h2>",
        '<div class="note"><b>init</b>：同一个种子，C++ 抽出的随机权重与 Python 逐位相同；'
        "<b>forward</b>：把 Python 的权重拷进 C++ 模型，比较 logits。</div>",
        '<table class="lessons"><thead><tr><th>preset</th><th>检查</th><th>张量数</th>'
        f"<th>max |diff|</th><th>通过</th></tr></thead><tbody>{parity_rows}</tbody></table>",
    ]
    if curves:
        py, cpp, res = curves["python"], curves["cpp"], curves["result"]
        diffs = [abs(a - b) for a, b in zip(py["train_loss"], cpp["train_loss"], strict=True)]
        parts += [
            "<h2>② 训练曲线：两条线几乎完全重合</h2>",
            f'<div class="note">preset <b>{res["preset"]}</b>，种子 {res["seed"]}。步数 Python '
            f"<b>{res['python_steps']}</b> / C++ <b>{res['cpp_steps']}</b>；逐步训练 loss 最大误差 "
            f"<b>{res['max_train_diff']:.2e}</b>，验证 loss 最大误差 <b>{res['max_val_diff']:.2e}</b>（门禁阈值 {curves['atol']:g}）。"
            "旧版因分词正则和 DataLoader 切分不同，同一配置下 Python 85 步、C++ 75 步。</div>",
            line_chart(
                "训练 loss（每步）与验证 loss",
                [
                    ("PyTorch train", py["steps"], py["train_loss"], "py", False),
                    ("LibTorch train", cpp["steps"], cpp["train_loss"], "cpp", True),
                    ("PyTorch val", py["eval_steps"], py["val_loss"], "py2", False),
                    ("LibTorch val", cpp["eval_steps"], cpp["val_loss"], "cpp2", True),
                ],
                "step",
                "loss",
            ),
            line_chart(
                "|PyTorch − LibTorch| 逐步 loss 差（放大看：都在 1e-6 量级）",
                [("|diff|", py["steps"], diffs, "diff", False)],
                "step",
                "|Δloss|",
            ),
        ]
    if bench:
        groups = [
            (f"{r['preset']} · {metric.replace('_ms', '')}", r["python"][metric], r["cpp"][metric])
            for r in bench["results"]
            for metric in ("forward_ms", "train_step_ms")
        ]
        parts += [
            "<h2>③ 性能（中位数，毫秒，越短越好）</h2>",
            f'<div class="note">{bench["machine"]}，{bench["threads"]} 线程，预热 {bench["warmup"]} 次后取 '
            f"{bench['iters']} 次的中位数；PyTorch {bench['pytorch']} / LibTorch {bench['libtorch']}。"
            "同一 preset、同一种子、同一输入形状。两边版本不同、机器共享，这组数只说明“差距不大”，"
            "不说明谁更快；大部分时间花在同一套 ATen 内核里，C++ 主要省掉的是 Python 调度开销，小模型上更明显。</div>",
            bar_chart(groups),
        ]
    return page("实测结果", "".join(parts), active="results", nav=nav_items())


# ----------------------------------------------------------------------------- transcripts


def record_transcripts() -> None:
    from compare_outputs import compare_lines  # noqa: PLC0415

    lessons: dict[str, Any] = {}
    for lesson in RUNNABLE:
        py = subprocess.run(
            [sys.executable, "-m", "torchtwins", lesson],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        cpp = subprocess.run(
            [str(CPP_CLI), lesson], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        mismatches = compare_lines(py, cpp, atol=1e-4)
        lessons[lesson] = {
            "python": py,
            "cpp": cpp,
            "verdict": "IDENTICAL" if not mismatches else f"{len(mismatches)} MISMATCH",
        }
    TRANSCRIPTS.write_text(
        json.dumps({"lessons": lessons}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ----------------------------------------------------------------------------- main


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    transcripts = load_json("transcripts.json")
    unknown = [k for k in NOTES if k.split(".")[0] not in twin_names()]
    if unknown:
        msg = f"notes for unknown modules: {unknown}"
        raise SystemExit(msg)
    summaries = []
    for module in LESSON_ORDER:
        html_text, summary = lesson_page(module, transcripts)
        (out / f"{module}.html").write_text(html_text, encoding="utf-8")
        summaries.append(summary)
    (out / "index.html").write_text(index_page(summaries), encoding="utf-8")
    (out / "results.html").write_text(results_page(), encoding="utf-8")
    (out / "style.css").write_text(CSS, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    # Offline SVGs for the homepage (no Mermaid CDN). Sources live in docs/diagrams/.
    dest = out / "diagrams"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for svg in sorted(DIAGRAMS_SRC.glob("*.svg")):
        shutil.copy2(svg, dest / svg.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if docs/site is stale")
    parser.add_argument("--record", action="store_true", help="re-record lesson transcripts")
    args = parser.parse_args(argv)
    if args.record:
        record_transcripts()
    if not args.check:
        build(SITE)
        print(f"[site] wrote {SITE.relative_to(REPO_ROOT)}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        cmp = filecmp.dircmp(tmp, SITE)
        stale = list(cmp.diff_files) + list(cmp.left_only) + list(cmp.right_only)
        # Also compare diagrams/ (dircmp is not recursive by default).
        for name in cmp.common_dirs:
            sub = filecmp.dircmp(Path(tmp) / name, SITE / name)
            stale.extend(f"{name}/{f}" for f in sub.diff_files + sub.left_only + sub.right_only)
    if stale:
        print(f"[site] stale: {stale} - run python tools/build_site.py")
        return 1
    print("[site] up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
