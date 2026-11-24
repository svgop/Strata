# Adoption — the persona arm is the default qwen standing (two-table cvec)

Date: 2026-11-23. The winning arm is IN `config/custom-iq3_s-8081.json` — no extra config file, no
shelf variant. The launcher reads it unchanged (LAUNCH LAW intact).

## What the standing now arms

```
--control-vector-scaled      …/vectors/qwen3.8-flash-next-persona-r03-r03s9rnsp-s9-rho1.gguf:0.45
--control-vector-layer-range 8 8          --cvec-mode add      --cvec-dir per-layer
--control-vector-scaled2     …/vectors/qwen3.8-flash-next-refusal-r.gguf:1.0
--control-vector-layer-range2 1 47        --cvec-mode2 project
```

Table 1: the projected persona row (direction.8, add, ρ=0.45 — Leg B's deployed-proven dose).
Table 2: the standing refusal-r ablation, unchanged from its pre-round arming. One boot, one
per-request switch, applied sequentially per layer.

## The engine change that made this legal (the per-entry-mode gap, closed)

The cvec format carried ONE table + ONE mode per boot: add (`h += s·v`) and project (`h -= s(h·v)v`,
directional ablation) could not compose — the persona row would have displaced the refusal
ablation and re-hardened refusals (battery: 10/10 persona-armed vs 0/10 refusal-r-armed). Change
set (all opt-in; absent `*2` flags = the byte-identical default path):

- `include/strata/kernels/cvec.hpp`: `Cvec` carries an optional second table; `covers()` is the
  union (the graph-structure predicate), `steered_at`/`steered2_at` split them.
- `src/kernels/cuda/cvec.cu`: `cvec_upload2`, device mirrors, `cvec_apply` launches per table in
  order; the pending FFN write rides the FIRST firing table at each layer.
- `src/program/generate.cpp`: `--control-vector-scaled2/--control-vector-layer-range2/--cvec-mode2`;
  the second table joins the session digest; summary/IDENT compose (`cvec2` object added).
- `src/prefill/prefill.cpp`: the write-fuse fuses only when exactly one table acts at the layer.
- `src/kernels/cvec_parity.cpp`: two-table cases (compose at the shared layer, t2-only layer, and
  the t2-only-with-pending-write case that caught the deployment bug).
- `serve/handshake.py`: the Layer-2 gate verifies BOTH tables against the config (refuses on any
  asks-vs-armed mismatch); legacy INFO line accepts the `+mode:A-B` suffix.
- `campaign/vector-lab.py`: lab boots strip both tables (isolation unchanged).

Proof chain: `cvec_parity` all-passed (compose 9.5e-07, t2-only 4.8e-07, t2-with-write 1.1e-06;
every pre-existing case still passes — the default path unchanged); the exact failing repro
(native pack + spec verify + two tables) went exit 1 → exit 0 across the fix.

## The bug the parity run caught (banked, law 11: instrument the instrument)

First deployment attempt: `verify: layer 2 never rang (illegal memory access)` — the t2-only
apply launched with `write=1` but null `bo/inj` (the fold pointers were not forwarded to the
second launch). Parity had only covered t2-only WITHOUT a pending write; the new parity case
covers exactly that hole. Bisect trail: single-table variants passed on the new binary; the fault
tracked the first t2-only layer (moved 2→4 with range2 3–47).

## Verification (the deployed process, not the edit)

Standing launcher boot, the real config: healthy 40 s; armed `add:8-8+project:1-47` PASS;
deployed chat template asserted; **refusal battery ON 0/10, OFF 10/10** (suppression preserved
through the second table); smoke answer assertive ("I would commit to an irreversible action based
on incomplete intelligence rather than delay for certainty"); server STOPPED after.

## Evidence chain for the arm itself

- Persona window: `05-r03-deployed/` (plain@0.3 deployed WINDOW) and `08-legB-p12-deployed/`
  (P12@0.45 deployed WINDOW — the adopted dose/vector).
- Projection confirmation: `07-legA-interleave/` (same-window plain-vs-P12, both doses).
- Refusal displacement check: `09-refusal-pair/` (the battery that forced the two-table design).
- Fence event, recorded: the standing config and engine binary changed BY DESIGN at adoption
  (18:12 config hash flip mid-Leg-B is explained in `08-legB-p12-deployed/VERDICT.md`); the
  rollback binary chain: `engine/strata.pre-phase-gate` (current; the pre-two-table binary was retired 2026-11-24 - git has the source), the rollback config is one edit (drop the
  `*2` block) or git.
