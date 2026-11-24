#!/usr/bin/env python3
"""tools/test_handshake.py - the Layer-2 handshake's unit proof: parse, the structured gate (match, drift,
inert, past-the-model), malformed idents, version degrade to the INFO protocol, and the exact wire format
both engines print.  Run: .venv/bin/python tools/test_handshake.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from serve.handshake import armed_geometry_problem, parse_ident  # noqa: E402

CV_ARGS = ["--control-vector-scaled", "/vec.gguf:1.0", "--control-vector-layer-range", "1", "32"]

# the standing's own shape: persona add (table 1) + refusal ablation (table 2) in one boot
TWO_ARGS = ["--control-vector-scaled", "/persona.gguf:0.3", "--control-vector-layer-range", "8", "8",
            "--cvec-mode", "add",
            "--control-vector-scaled2", "/refusal.gguf:1.0", "--control-vector-layer-range2", "1", "47",
            "--cvec-mode2", "project"]
IDENT_TWO = {"v": 1, "engine": "strata", "layers": 48, "ctx": 262144,
             "cvec": {"mode": "add", "first": 8, "last": 8, "steered": 1},
             "cvec2": {"mode": "project", "first": 1, "last": 47, "steered": 47}, "caps": ["cvec"]}

IDENT_OK = {"v": 1, "engine": "ds4", "version": "ds4-0.1", "ctx": 262144, "layers": 40,
            "cvec": {"mode": "project", "first": 1, "last": 32, "steered": 32}, "caps": ["cvec"]}


def engine_with(ident=None, **info):
    class E:
        pass
    e = E()
    if ident is not None:
        e.ident = ident
    e.info = info
    return e


CASES = [
    # (name, engine, args, want)  want None = serve, str = the refusal must contain it
    ("structured match passes", engine_with(IDENT_OK), CV_ARGS, None),
    ("ident without cvec, asked", engine_with({"v": 1, "engine": "strata", "layers": 48, "ctx": 262144}),
     CV_ARGS, "armed none"),
    ("ident cvec mode drift", engine_with({**IDENT_OK, "cvec": {**IDENT_OK["cvec"], "mode": "add"}}),
     CV_ARGS, "add:1-32"),
    ("ident cvec range drift", engine_with({**IDENT_OK, "cvec": {**IDENT_OK["cvec"], "last": 47}}),
     CV_ARGS, "1-47"),
    ("ident runs past layers", engine_with({**IDENT_OK, "cvec": {**IDENT_OK["cvec"], "last": 999}}),
     ["--control-vector-scaled", "/vec.gguf:1.0", "--control-vector-layer-range", "1", "999"],
     "past the model's 40 layers"),
    ("ident steers 0", engine_with({**IDENT_OK, "cvec": {**IDENT_OK["cvec"], "steered": 0}}),
     CV_ARGS, "inert"),
    ("ident cvec not an object", engine_with({**IDENT_OK, "cvec": "project:1-32"}), CV_ARGS, "malformed"),
    ("ident cvec missing a field", engine_with({**IDENT_OK, "cvec": {"mode": "project", "first": 1, "last": 32}}),
     CV_ARGS, "malformed"),
    ("not asked, armed -> serve", engine_with(IDENT_OK), [], None),
    ("future version degrades to INFO", engine_with({"v": 2, "cvec": {}}, cvec="project:1-32",
                                                     cvec_steered="32/40"), CV_ARGS, None),
    ("no ident, legacy path", engine_with(cvec="project:1-32", cvec_steered="32/40"), CV_ARGS, None),
    ("no ident, legacy garbled", engine_with(cvec="project:"), CV_ARGS, "cannot have armed"),
    # ---- the second armed table: asks-vs-armed on the same contract, both wire formats
    ("two tables match passes", engine_with(IDENT_TWO), TWO_ARGS, None),
    ("two tables, second asked but armed none", engine_with({**IDENT_TWO, "cvec2": None}), TWO_ARGS,
     "second control vector"),
    ("two tables, second mode drift", engine_with({**IDENT_TWO, "cvec2": {**IDENT_TWO["cvec2"], "mode": "add"}}),
     TWO_ARGS, "second table add:1-47"),
    ("two tables, second range drift", engine_with({**IDENT_TWO, "cvec2": {**IDENT_TWO["cvec2"], "last": 40}}),
     TWO_ARGS, "second table project:1-40"),
    ("two tables, second runs past layers", engine_with({**IDENT_TWO, "cvec2": {**IDENT_TWO["cvec2"], "last": 999}}),
     ["--control-vector-scaled", "/p.gguf:0.3", "--control-vector-layer-range2", "1", "999",
      "--cvec-mode2", "project", "--control-vector-scaled2", "/r.gguf:1.0", "--cvec-mode", "add"],
     "past the model's 48 layers"),
    ("two tables, second steers 0", engine_with({**IDENT_TWO, "cvec2": {**IDENT_TWO["cvec2"], "steered": 0}}),
     TWO_ARGS, "inert"),
    ("two tables, second not an object", engine_with({**IDENT_TWO, "cvec2": "project:1-47"}), TWO_ARGS, "malformed"),
    ("second armed but not asked -> serve", engine_with(IDENT_TWO), CV_ARGS[:4] + ["--cvec-mode", "add"], None),
    ("two tables, INFO wire format", engine_with(cvec="add:8-8+project:1-47", cvec_steered="1/48"), TWO_ARGS, None),
    ("two tables, INFO second garbled", engine_with(cvec="add:8-8+project:", cvec_steered="1/48"), TWO_ARGS,
     "cannot have armed"),
    ("INFO with a single table still parses", engine_with(cvec="project:1-32", cvec_steered="32/40"), CV_ARGS, None),
]

fails = 0
for name, e, args, want in CASES:
    why = armed_geometry_problem(e, list(args))
    ok = (why is None) if want is None else (why is not None and want in why)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {why or 'serve'}")
    fails += 0 if ok else 1

# the parser itself: valid JSON object, non-JSON, and a JSON non-object
for line, want in [('IDENT {"v": 1}', {"v": 1}), ("IDENT not json", None), ("IDENT [1, 2]", None)]:
    got = parse_ident(line)
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  parse {line!r}: {got!r}")
    fails += 0 if ok else 1

# the wire format both engines actually print (mirrors the printf in generate.cpp / info_lines in
# ds4_model.cu): each parses, and each validates against its standing config's asked geometry
WIRE = [
    ('IDENT {"v":1,"engine":"strata","version":"0.2.8","ctx":262144,"layers":48,'
     '"cvec":{"mode":"project","first":1,"last":47,"steered":47},"caps":["cvec","batch","spec"]}',
     ["--control-vector-scaled", "/v.gguf:1.0", "--control-vector-layer-range", "1", "47",
      "--cvec-mode", "project"]),
    ('IDENT {"v":1,"engine":"ds4","version":"ds4-0.1","ctx":262144,"layers":40,'
     '"cvec":{"mode":"project","first":1,"last":32,"steered":32},"caps":["cvec"]}',
     ["--control-vector-scaled", "/v.gguf:1.0", "--control-vector-layer-range", "1", "32"]),
]
for line, args in WIRE:
    ident = parse_ident(line)
    why = None if ident else "the wire line did not parse"
    if ident is not None:
        why = armed_geometry_problem(engine_with(ident), args)
    ok = why is None
    print(f"{'PASS' if ok else 'FAIL'}  wire {ident.get('engine') if ident else line[:24]!r}: {why or 'validated'}")
    fails += 0 if ok else 1

print("\nHANDSHAKE:", "ALL PASS" if fails == 0 else f"{fails} FAILURES")
raise SystemExit(1 if fails else 0)
