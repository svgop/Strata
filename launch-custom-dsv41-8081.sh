#!/bin/bash
# launch-custom-dsv41-8081.sh - T50/F-370: DeepSeek-V4.1-Flash on the CUSTOM tree's strata-dsv41 build.
# Alternate occupant of 8081 (same port law as the Qwen daily driver: swap by stopping one, starting the other).
# The engine is the upstream v0.2.7 dsv41 subsystem (src/dsv41, Project Maya ds4 port, MIT) built in this tree
# (build-fresh/strata-dsv41); the serving stack is OUR forked serve/server.py with the family port (eos, budget,
# slot-keep env, shared-start pin, model-card sampling). Text-first light: the vision tower is not passed
# (--vision off equivalent; serve/ds_pictures.py exists but is not wired).
cd "$(dirname "$0")"
exec .venv/bin/python serve/server.py --engine strata --config "$PWD/custom-dsv41-8081.json" --port 8081 --host 0.0.0.0
