#!/usr/bin/env python3
"""tools/lint_tree.py - the tripwire: re-implementation signatures become findings, not war stories.

Layer 4 of the architecture elevation (campaign/ROADMAP-architecture.md): the laws leave the
markdown.  This tool greps the tree for the signatures of a second implementation or a bypassed
owner - the classes that produced real incidents here:

  gguf      GGUF magic bytes outside the format's owners.  Owners: include/strata/artifact/
            gguf_reader.hpp (C++ read), tools/gguf_reader.py (Python read), tools/gguf_writer.py
            (Python write), tools/vision/gguf_meta_test.cpp + src/ngram/ple_reader_test.cpp
            (tests of the format itself).  Everything else parsing or emitting the magic is a
            second implementation; the cvec alignment hazard (2026-11-04, d08c2d11) was one.
  kill      process-kill sweeps outside campaign/battery-lib.sh.  Exception:
            campaign/batteries/t22-custom-vs-stock.sh (dual-checkout, port-scoped by design).
            The *strata* substring sweep killed both engines at 27 sites before the lib.
  exit      std::exit in code linked into a served engine (the hosted trees).  A hosted CUDA or
            load failure must surface as a refused request or a refused boot, never a dead
            process with no reason.  Standalone reference/bench binaries are not hosted.
  retired   invocations of retired artifacts in living scripts (the prewarm scripts were
            absorbed into the server on 2026-11-04; 22 batteries silently no-oped them for
            hours before the fleet was migrated).

Every hit must either be fixed or carry an entry in lint_exceptions.json with a reason - an
exception without a reason is itself a finding.  History and evidence (results/, research/,
SVG-CUSTOM*.md, .docs/) are records, not living code: never scanned.

    python tools/lint_tree.py            # exit 0 clean, 1 findings (paths + rules)
    python tools/lint_tree.py --selftest # seeds a violation of every rule in /tmp, must catch all
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

GGUF_OWNERS = {
    "include/strata/artifact/gguf_reader.hpp",   # the C++ reader both engines fold onto
    "tools/gguf_reader.py",                      # the Python reader
    "tools/gguf_writer.py",                      # the Python writer
    "tools/vision/gguf_meta_test.cpp",           # tests of the format itself
    "src/ngram/ple_reader_test.cpp",
    "src/artifact/gguf_reader_test.cpp",
}
GGUF_SCAN = ["tools", "src", "serve", "include"]
GGUF_SIG = re.compile(r'b?"GGUF"')

KILL_OWNERS = {"campaign/battery-lib.sh"}
KILL_EXCEPTIONS = {"campaign/batteries/t22-custom-vs-stock.sh"}   # dual-checkout: port-scoped by design
KILL_SCAN = ["campaign"]
KILL_SIG = re.compile(r"pkill\b|nvidia-smi\s+--query-compute-apps")

# the trees linked into a served engine (strata + strata-dsv41); reference and bench executables
# are their own processes and may exit as CLI tools do
HOSTED_TREES = {
    "src/qwen35/q36_engine.cu", "src/program", "src/core", "src/kernels", "src/prefill",
    "src/platform", "src/spec", "src/ngram", "src/artifact", "src/concurrent", "src/dsv41",
    "include",
}
EXIT_SIG = re.compile(r"\bstd::exit\s*\(|(?<!std::)\bexit\s*\(\s*[0-9-]")
EXIT_ALLOWED = re.compile(r"^\s*//|/\*|\*")                       # a comment mentioning exit( is prose

RETIRED = {"prewarm2": re.compile(r"bash\s+\S*prewarm2\.sh"),     # living scripts invoking the retired
           "prewarm-ds4": re.compile(r"bash\s+\S*prewarm-ds4\.sh")}   # warm-in scripts (no-ops since cc05319f)
RETIRED_SCAN = ["campaign", "tools", "packaging"]

RECORDS = re.compile(r"campaign/(results|research)/|SVG-CUSTOM|\.docs/|docs/|bench/results/")


def lint() -> list[tuple[str, str, str]]:
    """(path, rule, line) for every hit not owned or excepted.  The exit rule is a ratchet: the
    ledger carries a per-file BASELINE (the size of the known migration worklist when the tripwire
    was armed); a file at or under its baseline is the worklist, a new file or an appended site is
    a finding.  The worklist only ever shrinks."""
    counts: dict[str, dict[str, int]] = {}
    for path, rule, line in walk():
        counts.setdefault(rule, {}).setdefault(str(path.relative_to(ROOT)), 0)
        counts[rule][str(path.relative_to(ROOT))] += 1
    out: list[tuple[str, str, str]] = []
    led = load_ledger()
    for rule, files_ in counts.items():
        for rel, n in files_.items():
            base = led.get(rule, {}).get(rel)
            cap = base.get("baseline", 0) if isinstance(base, dict) else base
            if base is None or n > int(cap):
                out.append((rel, rule, f"{n} site(s) (baseline {cap}): the new one(s) regress the ratchet"))
    return out


def load_ledger() -> dict:
    f = ROOT / "tools/lint_exceptions.json"
    if not f.is_file():
        return {}
    led = json.loads(f.read_text())
    for rule, entries in led.items():
        for path, e in entries.items():
            why = e.get("reason") if isinstance(e, dict) else e
            if not isinstance(why, str) or not why.strip():
                raise SystemExit(f"lint_exceptions.json: {rule}/{path} has no reason - an exception "
                                 "without a reason is itself a finding")
            if isinstance(e, dict) and "baseline" in e:
                e["baseline"] = int(e["baseline"])
    return led


def files(roots: list[str]):
    for root in roots:
        base = ROOT / root
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".sh", ".cpp", ".cu", ".hpp", ".cuh", ".txt"):
                continue
            if RECORDS.search(str(p.relative_to(ROOT))):
                continue
            yield p


def walk():
    for p in files(GGUF_SCAN):
        rel = str(p.relative_to(ROOT))
        if rel in GGUF_OWNERS:
            continue
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if GGUF_SIG.search(line):
                    yield p, "gguf", f"{i}: {line.strip()[:100]}"
        except OSError:
            pass
    for p in files(KILL_SCAN):
        rel = str(p.relative_to(ROOT))
        if rel in KILL_OWNERS or rel in KILL_EXCEPTIONS:
            continue
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if KILL_SIG.search(line):
                    yield p, "kill", f"{i}: {line.strip()[:100]}"
        except OSError:
            pass
    for p in files(["src", "include"]):
        rel = str(p.relative_to(ROOT))
        if not any(rel == t or rel.startswith(t + "/") for t in HOSTED_TREES):
            continue
        if rel.endswith(("_test.cpp", "_parity.cpp", "_test.cu", "_bench.cpp", "_bench.cu",
                         "_parity.cu")):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if "int main(" in text or rel.startswith("src/dsv41/vendor/"):
            continue      # an executable's own lifecycle (a CLI refusal with its reason printed),
                          # or the declared vendor isolation
        for i, line in enumerate(text.splitlines(), 1):
            if EXIT_SIG.search(line) and not EXIT_ALLOWED.match(line):
                yield p, "exit", f"{i}: {line.strip()[:100]}"
    for p in files(RETIRED_SCAN):
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                for name, sig in RETIRED.items():
                    if sig.search(line):
                        yield p, f"retired:{name}", f"{i}: {line.strip()[:100]}"
        except OSError:
            pass


def selftest() -> int:
    """Seed one violation per rule in a scratch overlay; every rule must fire.  The lint's own
    negative test, the same discipline as the window-width wall's."""
    import shutil
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="lint-selftest-"))
    try:
        (tmp / "tools/evil_tool.py").parent.mkdir(parents=True)
        (tmp / "tools/evil_tool.py").write_text('if buf[:4] != b"GGUF": raise SystemExit(1)\n')
        (tmp / "campaign/evil_battery.sh").parent.mkdir(parents=True)
        (tmp / "campaign/evil_battery.sh").write_text("pkill -f strata\n")
        (tmp / "src/core/evil_hosted.cpp").parent.mkdir(parents=True)
        (tmp / "src/core/evil_hosted.cpp").write_text("if (bad) std::exit(1);\n")
        (tmp / "campaign/evil-warm.sh").write_text("bash campaign/prewarm2.sh || true\n")
        hits = set()
        for root in [tmp]:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(tmp))
                text = p.read_text()
                for i, line in enumerate(text.splitlines(), 1):
                    if GGUF_SIG.search(line) and rel not in GGUF_OWNERS:
                        hits.add("gguf")
                    if KILL_SIG.search(line) and not rel.endswith("battery-lib.sh"):
                        hits.add("kill")
                    if EXIT_SIG.search(line) and not EXIT_ALLOWED.match(line):
                        hits.add("exit")
                    for name, sig in RETIRED.items():
                        if sig.search(line):
                            hits.add(f"retired:{name}")
        need = {"gguf", "kill", "exit", "retired:prewarm2"}
        missing = need - hits
        print("selftest:", "ALL RULES FIRE" if not missing else f"MISSING {missing}")
        return 0 if not missing else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    findings = lint()
    for rel, rule, line in sorted(findings):
        print(f"[{rule}] {rel}  {line}")
    print(f"\n{len(findings)} finding(s)"
          + ("" if not findings else "  - fix them, or own them in tools/lint_exceptions.json with a reason"))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
