# campaign/ layout (F-268 restructure, 2026-10-22)

One home per kind of artifact. Anything deleted from here lives in git history at
its old path — the journal (SVG-CUSTOM.md) is the index of what ran and why.

## Top level — LIVING instruments (paths are referenced by batteries; do not move)
- `battery-lib.sh`    — the shared battery infrastructure (port_pid / cleanup_engine with the
  repo-path-scoped GPU kill / wait_health / wait_prewarm / final_serve / launcher_with_env — the
  standing launcher with env-arm overrides inserted after its exports, since the launcher's hard
  exports silently override env-passed arms, the F-284 class). Every battery and both
  ladders source it; the fleet's ~1,040 copy-pasted lines and the `*strata*` kill substring
  (which killed BOTH engines) lived in the per-script copies until 2026-11-04
- `probe-lane.py`     — the Proof-Law lane probe (M1/pair/conc/M3b/accept/ms-window)
- `probe-metrics.py`  — stability capture (TTFT/ITL/E2E p50-p99, clock/temp/power, golden hash) → results/metrics.jsonl
- `binary-ladder.sh`, `env-ladder.sh` — the promotion gauntlets (they wait for the server's own
  warm-in, the config's "prewarm" section, before their discarded leg)
- `gate-pair.sh`, `draft-temp-sweep.sh` — the cold-boot drivers (F-468/F-469): ONE gate per boot —
  NEVER two gates on one boot (the second gate re-asks identical prompts and the conversation
  cache serves run 1's parked KV back); one draft-temp arm per boot with the law-11 discard +
  measured legs and the M3b canary record.  Both boot via battery-lib's `launcher_with_env` and
  honor STRATA_BATTERY_DRYRUN (guards fire, nothing boots)
- `mmlu-pro-gate.py`, `mimicbench2.py` — quality gates (2x MMLU + mimic on adoptions)
- `bit-check.py`      — F-209 bit-identity check (temp-0 fixed prompts → results/f209-bit-*.json)
- `ctx-probe.py`, `depth-lane.py`, `depth-lane-real.py` — context/depth probes (post-retrain relevant)
- `mma-accum-ratelock.cu` — F-252 microbench (consumer-Blackwell fp32-accum half-rate; watch item)
- `t31-keeprefit.py`  — keep-rule simulator (agent deliverable, F-268)

## batteries/ — reusable harnesses
- `t20-gates.sh`           — MMLU + mimic gate pair for any adoption
- `t22-custom-vs-stock.sh` — binary-swap fair-baseline A/B (custom vs stock)
- `t31-rebank.sh` + `t31-summary.py` — the quiet-window re-bank instrument (F-263 law; F-266 anchor)
- `battery-lib.sh`         — shared engine up/down/witness helpers every standing battery sources
- the era runners (`t38-abab-b.sh`, `era-interleave.sh`) carry the ABAB interleave loop

## arms/ — parked future work (not dead, not standing)
- `launch-custom-iq3s-262k.sh` + `custom-iq3s-262k.json` — the 262k-context pair; refresh owed at GO time
  (paths inside predate this move — fix the .venv/config resolution when the switch is executed)

## research/ — dossiers, audits, specs (INDEX.md is the table of contents)
## results/ — battery logs, boot .outs, metrics.jsonl, spec-trace dataset, old-logs/ archives
## T14B-specs/ — retrain/re-quant execution specs

## Standing boot pair lives at REPO ROOT (do not move — LAUNCH LAW + F-249 .venv resolution):
`launch-custom-iq3s-8081.sh` (the launcher resolves `.venv` from its own dir).  The config it
starts is `config/custom-iq3_s-8081.json`; the steering vector lives in the estate at
`/home/svg/LLM/data/vectors/qwen3.8-flash-next-refusal-r.gguf`; the log at `logs/custom-iq3_s.log`.

## Policy (F-267/F-268)
- A battery's `.out` boot logs commit WITH their battery's `.log` summary; no drift.
- Run-once harnesses for closed questions get DELETED when the question closes (git history is the archive).
- The live engine log (`custom-iq3_s.log`, root) rotates gz into results/old-logs/ at cleanup passes.
