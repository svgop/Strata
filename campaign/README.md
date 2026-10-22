# campaign/ layout (F-268 restructure, 2026-10-22)

One home per kind of artifact. Anything deleted from here lives in git history at
its old path — the journal (SVG-CUSTOM.md) is the index of what ran and why.

## Top level — LIVING instruments (paths are referenced by batteries; do not move)
- `probe-lane.py`     — the Proof-Law lane probe (M1/pair/conc/M3b/accept/ms-window)
- `probe-metrics.py`  — stability capture (TTFT/ITL/E2E p50-p99, clock/temp/power, golden hash) → results/metrics.jsonl
- `prewarm2.sh`       — the ADOPTED post-boot warm package (run after /health, before probing; F-266)
- `mmlu-pro-gate.py`, `mimicbench2.py` — quality gates (2x MMLU + mimic on adoptions)
- `bit-check.py`      — F-209 bit-identity check (temp-0 fixed prompts → results/f209-bit-*.json)
- `ctx-probe.py`, `depth-lane.py`, `depth-lane-real.py` — context/depth probes (post-retrain relevant)
- `mma-accum-ratelock.cu` — F-252 microbench (consumer-Blackwell fp32-accum half-rate; watch item)
- `t31-keeprefit.py`  — keep-rule simulator (agent deliverable, F-268)

## batteries/ — reusable harnesses
- `abab-run.sh`            — the generic ABAB interleave battery runner
- `t20-gates.sh`           — MMLU + mimic gate pair for any adoption
- `t22-custom-vs-stock.sh` — binary-swap fair-baseline A/B (custom vs stock)
- `t31-rebank.sh` + `t31-summary.py` — the quiet-window re-bank instrument (F-263 law; F-266 anchor)

## arms/ — parked future work (not dead, not standing)
- `launch-custom-iq3s-262k.sh` + `custom-iq3s-262k.json` — the 262k-context pair; refresh owed at GO time
  (paths inside predate this move — fix the .venv/config resolution when the switch is executed)

## research/ — dossiers, audits, specs (INDEX.md is the table of contents)
## results/ — battery logs, boot .outs, metrics.jsonl, spec-trace dataset, old-logs/ archives
## T14B-specs/ — retrain/re-quant execution specs

## Standing boot pair lives at REPO ROOT (do not move — LAUNCH LAW + F-249 .venv resolution):
`launch-custom-iq3s-8081.sh` + `custom-iq3_s-8081.json` + `refusal-r.gguf` (config-referenced asset).

## Policy (F-267/F-268)
- A battery's `.out` boot logs commit WITH their battery's `.log` summary; no drift.
- Run-once harnesses for closed questions get DELETED when the question closes (git history is the archive).
- The live engine log (`custom-iq3_s.log`, root) rotates gz into results/old-logs/ at cleanup passes.
