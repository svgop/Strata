# The persona-vector recipe — how r03 was built and how to rebuild it

Everything that made the 2026-11-23 window real, in one place. Two tools own the whole pipeline:
`campaign/vector-capture.py` (elicit → capture → reduce → build) and `campaign/vector-lab.py`
(config → boot → AB → judge → verdict). Both fail loud; neither silently no-ops.

## The 6-step recipe (exactly what produced the deployed WINDOW)

```sh
# 1. The prompt set: 216 register rows (3 traits × 2 poles × 12 question/seeds), one file, hash-pinned.
#    (Reused from 01-r02-dose/elicit.jsonl; `vector-lab.py elicit` regenerates one.)

# 2. Capture: teacher-forced ±arms through the legacy pack, every position's residual dumped.
.venv/bin/python campaign/vector-capture.py capture \
  --elicit campaign/results/vec-2026-11-23/01-r02-dose/elicit.jsonl \
  --out /home/svg/LLM/Custom-data/persona-vectors/capture/<run> --per-cell 12 --resume

# 3. Reduce + rank: d′/prominence/coherence report per trait.  A clean "no candidate" is a RESULT.
.venv/bin/python campaign/vector-capture.py report --out /home/svg/LLM/Custom-data/persona-vectors/capture/<run>

# 4. Build the row: dump slot k -> direction.<k-1>; row = rho * mean||h_slot|| * v̂,
#    refusal-orthogonalized; --project-router strips the router-visible component (see physics).
.venv/bin/python campaign/vector-capture.py build \
  --out /home/svg/LLM/Custom-data/persona-vectors/capture/<run> --slot 9 --trait assertive \
  --rho 1.0 --project-router 12 --out-dir /mnt/DATA/LLM/Models/GGUF/vectors

# 5. Dose ladder, base context (mechanism isolation): the bracket is the science.
STRATA_RAM_KEEP_GIB=48 .venv/bin/python campaign/vector-lab.py ladder \
  --vector <vectors dir>/qwen3.8-flash-next-persona-<...>.gguf --label <label> \
  --range 8 8 --mode add --doses 0.03,0.1,0.3,0.6 --out <bare-name>

# 6. The DECISION test: deployed template, one dose, the same ON/OFF paired run.
STRATA_RAM_KEEP_GIB=48 .venv/bin/python campaign/vector-lab.py ladder \
  --vector <file> --label <label> --range 8 8 --mode add --doses 0.3 --context deployed --out <bare-name>
```

Single boots (interleaves, knee probes): `vector-lab.py ab --label L --vector F --dose D --range 8 8
--mode add --out DIR` — derives, boots, runs the 36-pair AB, tears down, fences, in one command.

## The physics that the numbers depend on

- **Index rule (boot-proven)**: `direction.<l>` acts on the residual AFTER block l ⇒ dump slot k is
  steered by `direction.<k-1>` (k ∈ 2..48). **Slot 1 is NOT steerable** (no direction.-1); the build
  refuses slots < 2.
- **Norm calibration**: row norm = ρ × the HARMONIC mean of the per-stream ‖h_slot‖ (±-arms averaged).
  Every block RMS-normalises each stream independently before the gated mix, so the add's share of the
  block input follows the harmonic mean, not the arithmetic one; the arithmetic calibration
  under-priced the true dose 1.59× at slot 9 and cost 4 net MMLU questions at "0.45" (the 2026-11-24
  interleave). Provenance records both norms and the per-stream relative dose.
  The ladder's `FILE:SCALE` multiplies the stored row linearly — dose d means `d × row_norm` of
  absolute perturbation per stream.
- **The 4× read-out**: hyper-connections broadcast every block output to all 4 streams and the final
  read-out is their row-wise SUM ⇒ at equal α the logits see ~4× a dense-model paper's dose. Price
  brackets accordingly (r02's 0.35 damage was read-out ≈ 1.4 effective).
- **Router visibility** (RARE 2608.21236, verified locally): the MoE router reads semantic content
  off the residual; a router-visible steering component corrupts expert selection before the persona
  saturates. `--project-router 12` (block 12 = the max-visibility router at ⅓ depth) removes that
  component: norm kept ~0.73, post-projection visibility ≈ random. Projection removed the 0.6-dose
  factual casualty with zero persona loss (one run; confirm with a two-dose interleave before
  defaulting the construction).
- **Refusal orthogonalization** is built into `build`: the row is orthogonalized (2 passes) against
  the refusal axes that act at the row's index (v2-primary per-layer + Huihui r) — persona lever and
  abliteration lever stay independent.
- **Slot choice**: filters (prominence > 1.25, stream-pair cos ≥ 0.8, …) guard vector QUALITY, but
  the 2026-11-23 capture passed NONE (diffuse in depth, stream-decomposed) and the causal ladder
  still found a window at slot 9. Filters reject noise vectors; a causal dose ladder is the actual
  arbiter of "does this slot steer".

## The evaluation contract

- Every AB: 36 pairs (18 temperature-0 + 18 seeded), blind, swap-balanced, both orders must agree
  (`judge` returns None otherwise). 16-question factual suite + finish/length/degeneration
  telemetry ride along on every boot.
- **Instrument validation is in-run**: each AB judges one known-answer control pair (crisply
  assertive vs hedged) in both orders; `judge_control_ok: false` voids the session's reads.
- **Refusal posture battery** (`vector-lab.py refusal`): 10 classic refusal-baiting prompts, ON vs
  OFF in one armed boot, marker-counted. Exists because adopting a persona row into a standing
  config displaces whatever cvec block was there (one table + one mode per boot) — the battery
  proves the displacement is refusal-neutral instead of assuming it.
- **Length confound**: re-read criteria at |Δwords| ≤ 25 (and ≤ 12 when n allows) — the 2026-11-23
  serious effect held 14/0 at matched length; a trait effect that dies at matched length was
  cosmetics.
- **judgeA (position-bias) is context-dependent**: ~1.0 under the stock template, 0.6–0.8 under the
  deployed template. Stock numbers are mechanism evidence; the DEPLOYED ON/OFF test decides.
- Temperature-0 is NOT deterministic on this engine (`determinism_ok` false, identical_rate 0.0):
  inertness claims come from the factual suite and finish counters, never from identity.

## Hard laws (each paid for in compute)

- One cvec table + one mode per boot; per-request arm via `experimental_speed_projection`.
- `ladder/ab --out` takes a BARE name (lands under `campaign/results/vec-2026-11-23/`) or an
  explicit path with `/`; nothing nests silently.
- Never delete or clean anything under a live run's output; the boot owns those files until
  `down` returns.
- `build` refuses overwrites and out-of-range slots; `capture` aborts on any failed boot, row-count
  mismatch, or non-finite slot in the usable range.
- Raw dumps/npz live in `/home/svg/LLM/Custom-data/persona-vectors/` (never the repo); evidence
  JSON/MD under `campaign/results/`; standing config/launcher/engine binary are fences — a
  fences.txt before/after block rides on every ladder.

## The 2026-11-23 result set (all evidence in `campaign/results/vec-2026-11-23/`)

plain slot-9 @0.3 deployed: **WINDOW** (ass 12/4, ser 13/1, dir 12/3, facts 1.00, trait density
+40%) — see `05-r03-deployed/VERDICT.md`. Knee: 0.6 plain 0.94 facts; 0.6 projected 1.00 (`06`).
**ADOPTED into the standing** (`10-adoption/VERDICT.md`): the projected row @0.45 as table 1
(add, 8-8) + refusal-r as table 2 (project, 1-47) — a two-table engine extension
(`--control-vector-scaled2/--cvec-mode2/--control-vector-layer-range2`, per-entry mode composition;
paritied, gate-verified, deployed-verified). Vectors:
`…/vectors/qwen3.8-flash-next-persona-r03-r03s9rnsp-s9-rho1.gguf` (adopted, direction.8, norm
1.0261) and `…r03-r03s9-s9-rho1.gguf` (plain, the 0.3 fallback). Source capture `r03-run1`,
means.npz sha16 `55be45efa4d6cef2`. Rollback: `engine/strata.pre-persona-merge` + drop the `*2`
config block (or git).
