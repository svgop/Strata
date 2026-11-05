#!/usr/bin/env python3
"""tools/test_geometry_gate.py - the Layer-2 gate's unit proof: every mismatch class refuses, every
consistent arm passes, old engines degrade to a warning.  Run: .venv/bin/python tools/test_geometry_gate.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0]]                       # server.py's import must not see CLI args
ns = {"__name__": "server_test", "__file__": str(pathlib.Path(__file__).resolve().parents[1] / "serve" / "server.py")}
exec(compile(open(ns["__file__"]).read(), ns["__file__"], "exec"), ns)
gate = ns["armed_geometry_problem"]

CV_ARGS = ["--control-vector-scaled", "/vec.gguf:1.0", "--control-vector-layer-range", "1", "32"]
CASES = [
    # (name, reported, args, verdict)  verdict: None = serve, str-substr = the refusal must contain it
    ("match passes",            "project:1-32", CV_ARGS, None),
    ("steering asked, none armed", "0",         CV_ARGS, "armed none"),
    ("steering asked, silent engine", None,     CV_ARGS, "armed none"),
    ("engine armed its own range", "project:4-24", CV_ARGS, "4-24"),
    ("mode drift",              "add:1-32",       CV_ARGS, "add:1-32"),
    ("last-layer drift",        "project:1-47",   CV_ARGS, "1-47"),
    ("no range asked, engine default accepted", "project:1-47",
     ["--control-vector-scaled", "/vec.gguf:1.0"], None),
    ("no steering asked, none armed", "0",       [], None),
    ("garbled report",          "project:",       CV_ARGS, "cannot have armed"),
    # the unexpected arm: a WARNING on stdout, not a refusal
    ("no steering asked but armed", "project:1-32", [], None),
]

fails = 0
for name, reported, args, want in CASES:
    class E:                                    # the engine's INFO face
        info = {"cvec": reported} if reported is not None else {}
    why = gate(E(), {"args": list(args)})
    if want is None:
        ok = why is None
    else:
        ok = why is not None and want in why
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {why or 'serve'}")
    fails += 0 if ok else 1

# the two standing configs against their real armed shapes (the plumbing-level check)
for cfg_path, shape in [("config/custom-iq3_s-8081.json", "project:1-47"),
                        ("config/custom-dsv41-8081.json", "project:1-32")]:
    import json
    cfg = json.loads((pathlib.Path(__file__).resolve().parents[1] / cfg_path).read_text())
    class E2:
        info = {"cvec": shape}
    why = gate(E2(), cfg)
    ok = why is None
    print(f"{'PASS' if ok else 'FAIL'}  {cfg_path} against {shape}: {why or 'validated'}")
    fails += 0 if ok else 1

print("\nGEOMETRY GATE:", "ALL PASS" if fails == 0 else f"{fails} FAILURES")
raise SystemExit(1 if fails else 0)
