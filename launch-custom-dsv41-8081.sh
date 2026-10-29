#!/bin/bash
# launch-custom-dsv41-8081.sh - T50/F-370: DeepSeek-V4.1-Flash on the CUSTOM tree's strata-dsv41 build.
# Alternate occupant of 8081 (same port law as the Qwen daily driver: swap by stopping one, starting the other).
# The engine is the upstream v0.2.7 dsv41 subsystem (src/dsv41, Project Maya ds4 port, MIT) built in this tree
# (build-fresh/strata-dsv41); the serving stack is OUR forked serve/server.py with the family port (eos, budget,
# slot-keep env, shared-start pin, model-card sampling). Text-first light: the vision tower is not passed
# (--vision off equivalent; serve/ds_pictures.py exists but is not wired).
# DS4_ATTN_SPLIT=1 ADOPTED (2026-10-29): the split-KV decode attention kernel (ds4_attn_split.cuh) -
# the FP4 rows' dequant inside the kernel, the unpack pre-pass gone.  3-leg battery vs the same-hour
# baseline (47.5/48.4/53.0): M1 51.9 (+9.4%), pair 51.9 (+7.2%), conc 57.0 (+7.5%) - every leg of every
# lane above every control leg; DS4_PROF attention stage 5.8-7.1 -> 5.6-5.8 ms/token.  Output class: the
# merge sums chunks in another order (last-place bits, 4e-7 vs a double reference - the F-224 class);
# the quality gate is owed (the T53 task-gate precedent; the change is far below --experts 5's trade).
# Revert: drop the line (the kernel stays default-off).
export DS4_ATTN_SPLIT=1

cd "$(dirname "$0")"
exec .venv/bin/python serve/server.py --engine strata --config "$PWD/custom-dsv41-8081.json" --port 8081 --host 0.0.0.0
