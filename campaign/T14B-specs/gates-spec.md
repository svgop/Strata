# T14B — FastMTP retrain: PARITY + QUALITY GATES spec (G0..G5)

Date: 2026-10-18. Owner lane: strata-custom-2. Companion specs: harvest spec, parity spec,
trainer-export spec (same directory). This file governs everything that happens AFTER `rt2`
(the retrained drafter directory) exists: the gate ladder that decides ADOPT / ITERATE / ABANDON.

Scope statement. The retrained MTP head REPLACES the drafter — different draft bits BY DESIGN
(T16 W2, SVG-CUSTOM-TODO.md:1069-1070: "Drafter swap = different bits BY DESIGN -> gates are
acceptance + 2x MMLU, not M3b"). So this ladder measures (a) acceptance at every depth,
(b) position-conditioned acceptance (the -40.5pt deep-position deficit is the retrain's target —
aggregate M3b alone is structurally blind to it), (c) quality (2x MMLU + the mimic 87.9 anchor),
(d) the F-230 decode-tax check if any prefill-path numerics moved, and (e) the per-depth
breakeven kill-rule arithmetic. It does NOT require draft bit-identity — that class of gate
(F-229's M3b canary) applies only where bit-identity is claimed (T16 #9, SVG-CUSTOM-TODO.md:1090).

Bindings honored: AGENTS.md "The performance campaign law" (:181-214, all 8 rules), the Proof
Law (F-108, SVG-CUSTOM.md:3027), F-225 (never compare across boot positions / session warm
trend), F-214 (battery boots must inherit the launcher env), F-205 (verify the arm env reached
the engine process), F-231 (deploy and battery never one command), T16 box law (batteries run
SERIAL; one variable per leg). No builds, no engine-code changes, no git operations in this spec.

---

## 0. Standing anchors and banked baselines (what every gate compares against)

| Quantity | Value | Source |
|---|---|---|
| M1 / pair / conc five-boot means | 198.5 / 197.6 / 312.4 t/s (banked at SM clock 2835; 3135 re-bank REQUIRED before absolute-milestone claims — T16 M0, SVG-CUSTOM-TODO.md:1056-1059) | SVG-CUSTOM.md:7998-8005 |
| M3b acceptance (same boots) | 74.5/76.0/76.0/76.0/72.2, mean 74.9% | SVG-CUSTOM.md:8005 |
| MMLU-Pro 2x gates | 95.7 / 94.3 (mean 95.0; gates run at floor arg 91.0, `mmlu-fbench-mmlu-{1,2}.json`) | SVG-CUSTOM.md:8008 |
| Mimic (mimicbench2, 40 tasks, weighted) | 87.9% (58/66 weighted; internalization 12/12, correction 8/8, voice 6/6, doc-structure 6/6, doctrine 6/8, arch-continue 6/8, blend 6/8, ui-deep 8/10) | SVG-CUSTOM.md:8009-8010; campaign/results/fbench-mimic.txt |
| Per-depth conditional marginals (min-p 0.5) | m ≈ [0.86, 0.85, 0.68, 0.75] at depths 2-5; offered ≈ 2.20/window; boot spread ±5pp | F-228, SVG-CUSTOM.md:7852-7862; T14C §1.2 |
| Marginal verify-row cost | c_row(M1) = 3.56 ms/row (range 2.5-4.6); c_row(conc) = 4.62 | T14C §1.3; F-233 (c_row=3.56 banked, SVG-CUSTOM.md:7992) |
| Deep-position acceptance (120k real corpus) | deep 53.1% (635/1195) vs shallow 93.6% = -40.5pt POSITIONAL; filler 68.0% at matched depth = -14.9pt CONTENT | F-236, SVG-CUSTOM.md:8113-8117; STEP 0 item 5, SVG-CUSTOM-TODO.md:1250-1255 |
| Quality floor | MMLU 95.0 / mimic 87.9, No-Regression Law absolute | T17 verdict, SVG-CUSTOM-TODO.md:1235-1236 |
| v1 gate (banked) | pos-2/3 >= +8pp each; pos-0 <= 1pp regression; e2e >= +8%; kill-rule depth extension only while marginal acceptance > 6-9% breakeven | T11.R3, SVG-CUSTOM-TODO.md:806-808; restated T16 W2 :1069-1070 and T14B P6 (campaign/research/T14B-fastmtp-training-infra.md:195-201) |

Adopt bar (campaign law): targeted metric >= +4% AND no priority metric regresses beyond -4%
(AGENTS.md rule 2, :192-193). Precedents: F-226 adopted at +4.35%, F-227 at +4.06%; F-231 held
below-bar at +1.81% and did NOT adopt.

---

## 1. Prerequisites and pre-flight artifacts

> **T27 AMENDMENT (2026-10-21) — THE PRODUCTION rt CONTRACT IS `mtp-en`, PINNED.**
> The standing launcher serves `--mtp /home/svg/LLM/data/mtp-en/rt`; its `draft_vocab.bin` is
> **162,100 B, sha256 369151522226a5ed…** (measured 2026-10-21). The full-vocab `mtp/rt` file is
> 425,196 B (b1e1d3a7a9e4bf86…). Every G0 copy step below must source the **mtp-en** file: copying
> the 425 KB full vocab binds the WRONG contract and the pre-flight byte-audit must fail loudly on
> a size/SHA mismatch, not silently pass. Also note `/home/svg/LLM/Custom-data` no longer exists
> (all 32 references in this spec set are stale paths) — resolve them to `/home/svg/LLM/data/`
> before any G0 run.

Inputs this spec assumes (produced by the sibling specs — none of them are gates here):
1. `/home/svg/LLM/Custom-data/mtp/rt2/` exists: trainer-exported `dense.bin` + `dense.txt`, with
   `experts.bin` and `draft_vocab.bin` copied BYTE-IDENTICAL from production `rt/` (the F-114
   validated shipping config — train dense only, stock experts; T14B §2.1/§3-P4,
   campaign/research/T14B-fastmtp-training-infra.md:29, 183-186). **Production = `mtp-en/rt`
   per the T27 amendment above.**
2. The rt2 value-audit is clean: every UNtrained tensor byte/quant-equal to stock rt
   (T14B P4 "value-audit", :185-186; the F-111..F-113 audit class).
3. The parity spec's STRATA_MTP_TAP gate passed (torch-vs-engine parity, cos > 0.9999 per
   stage; the F-80 law "G0 parity before ANY training step", T14B §2.4-1 and §3-P2, :84, 161-163).
4. Norms exported RAW and folded at rt build only (the F-109 double-fold law; mtp_rt.py writes
   Gemma "+1" into dense.bin F32 — tools/mtp_rt.py header, :10-11).

Pre-flight config artifacts (write these in a SEPARATE command from any battery run — the
F-231 deploy law, SVG-CUSTOM.md:7936-7938). Each is a copy of the standing config
`custom-iq3_s-8081.json` (single-variable law, AGENTS.md rule 5) with ONLY the deltas listed.
The `env` dict + per-arm `log` pattern is the existing stair50 convention
(campaign/stair50-iq3_s-8081.json: "log": ".../custom-iq3s-stair50.log", "env":
{"STRATA_SPEC_DEPTH": "1"}; server.py applies config env AFTER os.environ, t17-leg.sh:4-6).

| File | Deltas from custom-iq3_s-8081.json | Used by |
|---|---|---|
| campaign/t14b-g0-rt.json | `--mtp` -> `/home/svg/LLM/Custom-data/mtp/rt-g0`; `log` -> `custom-iq3s-t14b.log` | G0 |
| campaign/t14b-rt2-arm.json | `--mtp` -> `/home/svg/LLM/Custom-data/mtp/rt2`; `log` -> `custom-iq3s-t14b.log` | G2, G3, G4 |
| campaign/t14b-g1-ctl.json | `log` -> `custom-iq3s-t14b-g1ctl.log`; `"env": {"STRATA_SPEC_DEPTH": "1"}` | G1 |
| campaign/t14b-g1-arm.json | `--mtp` -> `.../rt2`; `log` -> `custom-iq3s-t14b-g1arm.log`; `"env": {"STRATA_SPEC_DEPTH": "1"}` | G1 |

Production `--mtp` today: `/home/svg/LLM/Custom-data/mtp/rt` (custom-iq3_s-8081.json:56-57).
Both launchers point at this same config (launch-custom-iq3s-8081.sh:42 and
launch-custom-iq3s-8081-conc.sh:47), so a config-level adoption propagates to both lanes.
`rt/` is NEVER written (T14B P5, :188-193). Keep `--mtp-q4 all --mtp-hnorm stream --spec 6
--spec-min-p 0.5` IDENTICAL in every config — train/serve arm parity is a G5 check item
(T14B P6, :199). No `--dump-*` flag may be armed on any serving boot (harvest-only).

Boot discipline for every leg below: `campaign/t17-leg.sh <config.json> <label>` — stops
whatever holds 8081, verifies VRAM drained (:21-30), evals the launcher export block so the
standing recipe is inherited (the F-214 lesson, :32-33; battery-config.sh:16-22), verifies the
arm env reached the engine process (F-205 law, :53-58), runs prewarm, boots, runs probe-lane,
banks `campaign/results/<label>.txt`. Serial only — dev may overlap, batteries never (T16).

---

## 2. The gate ladder

| Gate | Question | Kill power | Runtime | Evidence file(s) in campaign/results/ |
|---|---|---|---|---|
| G0 | Does the pipeline preserve a stock drafter end-to-end? (dry-run) | binary (pipeline bug) | ~1.0-1.5 h | t14b-g0-rt.txt, f209-bit-t14b-g0.json, t14b-g0-rtaudit.txt |
| G1 | Does the retrained head lift the per-position acceptance table without pos-0 regression? | binary (drafter dead on arrival) | ~1.5 h | t14b-g1-stair-{ctl,arm}.txt, depth-t14b-g1-real120k-{ctl,arm}.txt, f209-bit-t14b-g1-{ctl,arm}.json, t14b-g1-positions.md |
| G2 | Does e2e clear the adoption bar on the interleaved battery? | adoption verdict | ~2 h | t14b-g2-{ctl,arm}-r{1,2,3}.txt |
| G3 | Does quality hold? (2x MMLU-Pro, matched arm/ctl) | absolute (No-Regression Law) | ~45-60 min | mmlu-t14b-mmlu-{ctl,arm}-{1,2}.json |
| G4 | Does mimic hold at the 87.9 anchor? | absolute | ~40-50 min | mimicbench2-t14b-{ctl,arm}.json |
| G5 | F-230 tax check + per-depth kill-rule arithmetic | investigate-before-adopt | ~30 min desk | t14b-g5-tax.md |

Total: ~6.5-7 h of serial boots; split as session A = G0+G1, session B = G2+G3+G4+G5
(same-day ctl anchors; never compare across sessions — F-225, SVG-CUSTOM.md:7798-7802).

### G0 — Pipeline dry-run: stock round-trip through the NEW export chain

T11.R3 banked "dry-run on 9B sibling first (pipeline de-risk, 15-30 min on 5090)"
(SVG-CUSTOM-TODO.md:806). The infra audit resolved the prerequisite: NO sibling model
exists on this box (searched; T14B §5 table row "Dry-run on a 9B sibling", :263) and the
externally-validated production recipe de-risks the method side (T17A verdict,
SVG-CUSTOM-TODO.md:1174-1175). The sanctioned substitute is the free round-trip test: run the
STOCK tensors through the NEW export chain and require stock-class acceptance back — "cheap and
decisive" (T14B §5, :263; the F-114 test class). It exercises pack -> rt -> boot -> serve with
zero training, so any failure is a PIPELINE bug, not a drafter verdict.

Commands:
```
cd /home/svg/LLM/strata-custom-2
mkdir -p /tmp/t14b-g0
.venv/bin/python tools/mtp_pack.py --src /home/svg/LLM/Custom-data/mtp/tensors \
    --experts q2_0 --out /tmp/t14b-g0/mtp-roundtrip-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf /tmp/t14b-g0/mtp-roundtrip-q2_0.gguf \
    --out /home/svg/LLM/Custom-data/mtp/rt-g0
cp /home/svg/LLM/Custom-data/mtp/rt/draft_vocab.bin \
   /home/svg/LLM/Custom-data/mtp/rt/experts.bin /home/svg/LLM/Custom-data/mtp/rt-g0/
# (tool usage per tools/mtp_pack.py and tools/mtp_rt.py headers; ~0.9 GB + ~0.83 GB outputs)
bash campaign/t17-leg.sh custom-iq3_s-8081.json t14b-g0-ctl
.venv/bin/python campaign/bit-check.py t14b-g0-ctl
bash campaign/t17-leg.sh campaign/t14b-g0-rt.json t14b-g0-rt
.venv/bin/python campaign/bit-check.py t14b-g0
```

Pass/fail:
- rt-g0 value-audit vs rt: every tensor byte/quant-equal (the pack/rt path is deterministic;
  any drift = export-chain bug). Save the diff to campaign/results/t14b-g0-rtaudit.txt.
- Boot binds rt-g0: the drafter line prints rt-g0's MiB at bind (mtp.cpp:528, per T14B P5, :193).
- M3b FLAT vs the same-session ctl leg: |delta| <= ~0.5pp. Rationale: rt-g0 IS the stock
  drafter; the flattest same-bits canaries of the campaign read +0.13pp (F-229) and -0.04pp
  (F-233). A same-tensors round-trip that moves M3b has corrupted something — binary kill.
- bit-check.py (3 fixed prompts, temp 0, sha256, campaign/bit-check.py): expect 3/3 sha equal
  to the ctl leg — at temp 0 the exact-match verifier emits the target's own picks and wrong
  drafts are REJECTED, never emitted, so greedy text must be identical (T14C G0.1,
  campaign/research/T14C-fastmtp-acceptance-math-gates.md:176-180). CAVEAT (F-180 method
  finding, SVG-CUSTOM.md:5800-5804): greedy streams are history-dependent through the stateful
  expert tier — even a same-binary fresh boot can drift a token. On divergence: re-run
  bit-check on a SECOND fresh ctl boot first; only a reproducible arm-vs-ctl divergence with
  stable ctl is the engine-bug kill. Do not skip this triage step in either direction.

Runtime: ~1.0-1.5 h (export ~20 min, two boots x ~15-30 min, bit-checks ~3 min each).

### G1 — Per-position acceptance table (the drafter's own v1 gate)

The banked v1 gate is PER-POSITION, not aggregate (T11.R3, SVG-CUSTOM-TODO.md:806-807):
pos-2/3 >= +8pp each, pos-0 <= 1pp regression. The instrument is the engine's M3d per-depth
counter (`STRATA_SPEC_DEPTH=1` publishes accepted/tested by depth; probe-lane.py:146-149 prints
the lines; counters at generate.cpp:13006, print at :13431-13436 per T14C §1). M3d marginals
are CONDITIONAL: m_k = P(accept at k | offered at k, chain unbroken) — the exact quantity the
kill-rule consumes (T14C §1.1). Because boot spread moves the marginals ±5pp (stair50 vs
stair35, T14C §1.2), arm and ctl MUST be same-session boots.

Commands (session A, after G0):
```
bash campaign/t17-leg.sh campaign/t14b-g1-ctl.json t14b-g1-stair-ctl
.venv/bin/python campaign/depth-lane-real.py --port 8081 --label t14b-g1-real120k-ctl \
    --depth 120000 --decode-tok 1500 --log /home/svg/LLM/strata-custom-2/custom-iq3s-t14b-g1ctl.log
.venv/bin/python campaign/bit-check.py t14b-g1-ctl
bash campaign/t17-leg.sh campaign/t14b-g1-arm.json t14b-g1-stair-arm
.venv/bin/python campaign/depth-lane-real.py --port 8081 --label t14b-g1-real120k-arm \
    --depth 120000 --decode-tok 1500 --log /home/svg/LLM/strata-custom-2/custom-iq3s-t14b-g1arm.log
.venv/bin/python campaign/bit-check.py t14b-g1-arm
```
Then parse both stair probes into the position table and write
campaign/results/t14b-g1-positions.md: per depth k = 1..6, m_k = dp_ok/dp_tested (conditional),
plus offered/window and the probe's accept % — the columns G2's M3b read and G5's kill-rule
precompute both consume. The depth-lane-real runs give the POSITION-CONDITIONED read at ~120k
(real repo prose, 3.27 chars/token calibration, depth-lane-real.py:36; the instrument that
produced the 53.1% / 93.6% / 68.0% split, F-236).

Pass/fail:
- pos-0 (depth-1 conditional marginal): regression <= 1pp vs ctl same-session. Depth-1 is
  SACRED: at fixed shape each acceptance pp is +0.80% M1 (T14C §1.3 elasticity), so a retrain
  that trades k1 for k2 is net-negative before any depth gain (T14C G0.2, :186-188).
- pos-2 AND pos-3: +>= 8pp each vs ctl (T11.R3 v1 gate). Falsification economics: a head that
  misses this dies here for the cost of two boots — no 3v3 is spent (T14C §3 G0.2, :188-189;
  §7, :319).
- offered/window: REPORTED, not gated (it pre-computes G5; a retrained drafter with higher
  confidence also changes how many rows the min-p gate offers).
- Deep position (120k real-corpus acceptance from the depth files): arm >= ctl − 1pp (no
  regression required at v1). The retrain's whole point is this number (the -40.5pt deficit;
  position-oversampled training data was CONFIRMED by STEP 0, SVG-CUSTOM-TODO.md:1254-1255).
  Record the lift; a lift >= +5pp at matched depth is the retrain-target success signal and
  feeds the kill-rule table in G5. A DEEP REGRESSION with shallow lifts = the position-conditioned
  failure class -> ITERATE (decision D2) with position/content-augmented re-harvest, not abandon.
- bit-check arm vs ctl: same expectation and triage as G0 (temp-0 committed-stream identity;
  F-180 confounder check first).
- Boot-noise tie-breaker (F-225 law): if any margin sits within 3pp of its gate line, run one
  REVERSED-order pair (arm leg first, then ctl) before the verdict — F-228's +8.9% single-boot
  read was exactly the boot-noise the pairing exists to catch (SVG-CUSTOM.md:7860-7862).

What M3b means in this gate: M3b is the acceptance ACCOUNTING source, not a quality gate. A
deeper offered chain mechanically compresses the aggregate rate while tokens/window rise
(s35: M3b 75.2 -> 72.7 while C rose +10.5% — T14C G1, :193-194). True flags only:
M3b < ~70 (below the s35-arm floor) or a depth-1 marginal regression.

Runtime: ~1.5 h (two legs x ~35 min: boot ~12 min, probe ~8 min, depth-lane ~5 min, bit-check
~3 min).

### G2 — 3v3 A/B battery (the adoption instrument)

Interleaved arm/ctl boots, 3 rounds — the campaign's interleave discipline (F-225's lesson is
why single-boot reads do not adopt; F-233's fenced 3v3 is the model, SVG-CUSTOM.md:7981-7987).
Ctl = the standing config; arm = rt2 config; NOTHING else differs.

Commands (session B; configs already written in pre-flight — this loop is battery-only):
```
for r in 1 2 3; do
  bash campaign/t17-leg.sh custom-iq3_s-8081.json t14b-g2-ctl-r$r || exit 1
  bash campaign/t17-leg.sh campaign/t14b-rt2-arm.json t14b-g2-arm-r$r || exit 1
done
```
Verdict arithmetic (means over 3 rounds per lane; deltas arm-vs-ctl same-session):
- ADOPT bar: M1 >= +4%, arm ahead in all/most rounds; pair and 6-conc each >= -4%; M3b read
  jointly (mechanical compression expected per G1; flags: M3b < ~70 or depth-1 regression).
  Banked rationale: AGENTS.md rule 2 (:192-193); F-226 (+4.35) and F-227 (+4.06) adopted at
  this bar; F-231 (+1.81) rejected below it.
- The e2e >= +8% v1 line (T11.R3) is the CLEAR-WIN tier, distinct from the adopt bar: crossing
  it unlocks the depth-extension chapter (the kill-rule evaluation on the adopted base + the
  W2.5 min-p re-stair, SVG-CUSTOM-TODO.md:1072-1074) and the ~237-class milestone checkpoint
  (T16 ladder, :1093-1094). RECONCILIATION (T14C §2.1/§5-5): the retrain-alone central
  expectation at fixed shape is +5.7-6.4% M1 (scenario a), landing 223-229 — T11.R3's "~237"
  held the window flat while adding tokens and is re-priced, not overturned. Expect +4..+8% to
  be a NORMAL passing outcome; +8% is the exceptional one.
- The F-231 pattern veto: an M1-only win while pair sits twice at the 188-209 band floor must
  NOT adopt (T14C G2, :200-202; F-231's unresolved pair -3.61%, SVG-CUSTOM.md:7926-7929).
- Single-stream protection is absolute: M1 regression beyond noise kills the arm regardless of
  conc (AGENTS.md rule 4, :199-202).

Runtime: ~2 h (six legs x ~18-20 min).

### G3 — 2x MMLU-Pro (matched arm/ctl, same session)

The drafter is output-affecting whenever acceptance changes the resample path (T14B P6, :197-199
— "treat as token-stream-affecting"), so the MMLU gate is mandatory class. 2x per arm
unconditionally (T16 #9 gates law, SVG-CUSTOM-TODO.md:1090-1092; AGENTS.md rule 3 escalates at
2 if the first is borderline, :196-197).

Commands (run immediately after G2, same session):
```
bash campaign/t17-leg.sh custom-iq3_s-8081.json t14b-g3-ctl
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 t14b-mmlu-ctl-1
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 t14b-mmlu-ctl-2
bash campaign/t17-leg.sh campaign/t14b-rt2-arm.json t14b-g3-arm
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 t14b-mmlu-arm-1
.venv/bin/python campaign/mmlu-pro-gate.py 0.933 t14b-mmlu-arm-2
```
MUST use the repo venv python — system python3 lacks `datasets` (F-230 note,
SVG-CUSTOM.md:7911-7912: "fix before the next quality gate"). The floor argument 0.933 is the
AGENTS.md rule-3 canonical form (also T14B P6); the banked 95.0 anchor gates ran 91.0
(mmlu-fbench-mmlu-{1,2}.json) and floors 0.7/0.9/0.933/91 have all read 92.9-95.7 on this
build, so the floor arg is not the sensitive variable — arm/ctl MATCHING is. The script posts
the floor per-request (the matched-window form, mmlu-pro-gate.py:29-37) and prints the binomial
1-sigma (:59-61).

Pass/fail:
- Pooled arm (mean of 2) >= pooled ctl − 1 sigma_pooled (sigma ~= 5.2 pts per run, ~= 3.7
  pooled; T14C G3, :204-206).
- Watch/escalation: any single gate < 93.3 (AGENTS.md rule-3 escalation floor) or the 2-gate
  mean < 92 escalates to investigation before any adoption claim (F-180 watch rule,
  SVG-CUSTOM.md:5809-5811 — "re-gate MMLU at the next adoption round; a 3-gate mean < 92
  escalates"; this ladder runs the 2x form).
- Anchor context: 95.0 mean is the standing quality floor (T16/T17). A passing-but-below-anchor
  mean (e.g. 93.5) with a clean pooled delta vs same-session ctl is a PASS by the law, with the
  anchor restored expected at the five-boot re-anchor; record it either way.

Runtime: ~45-60 min (two boots + four gates at ~8-10 min each).

### G4 — Mimic (mimicbench2, the 87.9 anchor)

Commands (same boots as G3, before tearing each down):
```
# on the g3-ctl boot:
.venv/bin/python campaign/mimicbench2.py 8081 t14b-ctl
# on the g3-arm boot:
.venv/bin/python campaign/mimicbench2.py 8081 t14b-arm
```
Usage per the script header (`python mimicbench2.py [port] [label]`; 40 tasks, weighted scoring
— surface 1pt / structural 2pt / internalization 3pt; writes
campaign/results/mimicbench2-<label>.json and prints `MIMICBENCH2 [<label>]: 58/66 = 87.9%`
format; mimicbench2.py:35-47). NOTE: mimicbench2 is the anchor instrument (87.9 = 58/66
weighted); the older mimicbench v1 band 95.3-98.4 quoted in T14C G3 is a DIFFERENT instrument —
never mix the scales.

Pass/fail:
- Paired read: arm >= ctl − 3.0 weighted points AND arm >= 84.9 hard floor. Rationale: one w2
  task = 3.0/66 weighted = the smallest visible step; 84.9 = the 87.9 anchor minus one w2 task.
  The ±1-task tolerance is an INFERENCE (the campaign has banked no repeated-mimic variance;
  low confidence) — the paired same-session ctl read is the real control, and the anchor must
  be re-confirmed at the five-boot re-anchor after adoption (quality floor: mimic 87.9,
  SVG-CUSTOM-TODO.md:1092, 1235-1236).
- Mimic failure is ABSOLUTE (No-Regression Law, T17 verdict :1236): two failures = abandon (D6),
  not iterate-the-drafter — a mimic drop with clean MMLU means the harvest distribution damaged
  something MMLU cannot see; the T11.R5 lesson (3-bit damage lands in code/math first) applies.

Runtime: ~40-50 min (two runs x ~15-20 min on already-booted legs).

### G5 — The coupled-MTP tax check (F-230 law) + per-depth kill-rule arithmetic

Desk gate over G1/G2 evidence; fires boots only if the tax signature appears.

1. Config-diff assertion (single-variable law, AGENTS.md rule 5): the arm/ctl configs differ
   ONLY in `--mtp` (+ log path, + the stair env dict). Any other delta VOIDS the battery. No
   `--dump-*` on any serving boot. Train-serve parity: `--mtp-q4 all --mtp-hnorm stream`
   identical both sides (T14B P6, :199).
2. The law being checked (F-230, SVG-CUSTOM.md:7906-7909): "any optimization that changes
   prefill BITS pays ~-2.5% on every decode t/s metric through acceptance" — the coupled
   exact-match verifier flips borderline accept/reject on ~1.8e-4 KL of KV change; F-230
   measured -2.7% relative acceptance = -2.8% M1 with the arithmetic closing exactly. The
   retrained head replaces DRAFT-side weights only; target prefill bits are untouched BY
   CONSTRUCTION (rt2 carries dense+stock experts; the backbone pack is not in the swap). G5
   VERIFIES that construction rather than assuming it.
3. Signature read (from G1's position table + G2's M3b):
   - F-230 tax signature = UNIFORM acceptance dip across ALL depths including depth-1
     (theirs: M3b -2.1pp, arm max below ctl mean), with the decode loss explained by
     acceptance and nothing else.
   - Intended retrain signature = depth-1 flat + position-conditioned deep lifts.
   - A uniform dip -> INVESTIGATE before any verdict (decision D7): (a) direct-cast quant loss
     on fresh dense weights — falsifier: serve the SAME checkpoint bf16 vs q4 (T14A R2,
     campaign/research/T14A-fastmtp-method-recipe.md:156: if the cast costs > 2pp acceptance,
     add per-tensor AMAX re-calibration or a 1-epoch quantization-aware finetune at LR 5e-6;
     Q4 direct-cast is the validated shipping form — NEVER GPTQ the head, F-201-mtp-agent.md:9
     and SVG-CUSTOM.md:7079); (b) harvest contamination (a dump hook or numerics arm left
     armed); (c) an accidental config delta (back to step 1).
4. Arithmetic closure: delta-M1% ~= 0.80 x delta(per-proposal acceptance pp) at fixed shape
   (T14C §1.3). A decode deficit NOT explained by the acceptance delta is a non-acceptance
   regression (engine bug class) — investigate before adopt. F-230 is the closure precedent.
5. Kill-rule precompute (table below) from the arm's own G1 stair read: s_k (offer rates),
   m_k (conditional marginals), offered/window.

Evidence: write campaign/results/t14b-g5-tax.md containing the diff assertion, the signature
table, the closure arithmetic, and the per-depth kill-rule table. Runtime: ~30 min desk
(+2 boots ~35 min only if the bf16-vs-q4 falsifier fires).

---

## 3. The kill-rule: breakeven arithmetic per depth (the depth-extension chapter)

Banked phrasing (T11.R3, SVG-CUSTOM-TODO.md:807): "kill-rule depth extension only while
marginal acceptance > 6-9% breakeven." The 6-9% figure is the depth-economics literature form
(marginal depth pays above ~6-9% acceptance when draft rows are cheap; F-202 item 5,
SVG-CUSTOM.md:7076-7077 "marginal depth pays above 6-9% acceptance"; R35 "stopping rule >=
7-9%"). ON THIS BOX the form is wrong: a marginal verify row costs the FULL T-scaled graph,
c_row = 3.56 ms/row M1 (2.5-4.6) and 4.62 conc — 26-30% of the window, not the literature's
few percent (F-228's cost model correction, SVG-CUSTOM.md:7857-7860; T14C §1.3). The corrected,
implemented rule (T14C §1.4):

```
Depth extension pays iff  ΔC/Δoffered > c_row · C0/W0
  M1:   3.56 × 2.77/13.74  ≈ 0.72 expected tokens per extra OFFERED row
  conc: 4.62 × 5.69/22.32  ≈ 1.18 expected tokens per extra offered row
(one accepted token is worth W0/C0 ≈ 4.96 ms; a marginal row must return ≥ 0.72 of them
to cover its 3.56 ms.)
```

Per-depth baseline (F-228 stair at min-p 0.5; T14C §1.2 — boot spread ±5pp):

| depth k | offer rate s_k | conditional marginal m_k | verdict at c_row 3.56 (M1) | verdict at 4.62 (conc) |
|---|---|---|---|---|
| 2 | 0.63 | ~0.86 | pays (> 0.72) | dead (< 1.18) |
| 3 | 0.35 | ~0.85 | pays | dead |
| 4 | 0.17 | ~0.68 | DOES NOT PAY | dead |
| 5 | 0.08 | ~0.75 | borderline | dead |

Reading: the standing chain is trimmed close to correctly by the min-p gate; conc depth beyond
what staleness already sets is arithmetically dead at today's c_row (1.18 tokens/row is above
any measured marginal). An extension (min-p re-stair on the retrained drafter, W2.5; or
`--spec` beyond 6) is killed UNLESS the retrained head's own stair read shows m_k > 0.72 (M1)
at the added depth — and is only worth a 3v3 after the stair read clears it (never re-arm 0.35
on the old drafter: two measured negatives bracket 0.42, F-228 :7863, T14C G5 :226-228).

What a retrain must add for depth to pay (T14C §1.4, deep offered mass ~1.04 rows/window at
the 0.35 anchor):

| deep-marginal lift (d4/d5) | net M1 |
|---|---|
| +5pp | 0% (breakeven) |
| +10pp | +1.6% |
| +15pp | +3.3% |
| +20pp | +4.9% (reaches the adopt bar by depth ALONE only here) |

So the depth chapter is a CONDITIONAL second chapter: the acceptance-at-fixed-shape component
is what carries G2; depth pays only on top of a large deep lift or a cheaper row (the kernel
tranche taking c_row to <= 2.8 drops the threshold to ~0.56 — T14C §1.4). Retrain-alone
ceiling stays ~222-229 M1 (T14C §2.1); the 250 path = kernel tranche AND retrain.

---

## 4. Kill-rule decision table (adopt / iterate objective / abandon)

| # | Evidence state | Verdict | Action |
|---|---|---|---|
| D1 | G0 fails: value-audit drift, M3b not flat on a same-tensors round-trip, or reproducible bit-check divergence | PIPELINE BUG (not a drafter verdict) | fix export/parity chain; re-run G0. NOTHING else may run (the F-80 law: no gate downstream of a broken chain means anything). |
| D2 | G1 fails: m2 or m3 lift < +8pp, OR pos-0 regression > 1pp, OR deep-position regression at 120k | ITERATE the objective (max 2 iterations) | Objective-swap menu (banked, T17C retrain upgrades, SVG-CUSTOM-TODO.md:1161-1165): VAT (arXiv 2608.30135), LK losses (2602.23881, direct SUM min(p,q) — native to the 40,525-row sub-vocab head), AdaSPEC (2510.19779), warm-start discipline; Draft-OPD rejection replay (2605.29343); position/content-augmented re-harvest (F-236 data-spec); beta re-tune; if G5 found cast loss: AMAX recalib / fake-quant FT (T14A R2). Each iteration re-enters at G0 (value-audit) -> G1. Two failed iterations -> D6. |
| D3 | G1 pass AND G2 M1 >= +8% AND all lanes >= -4% AND G3 pass AND G4 pass | ADOPT — clear-win tier | Config swap (section 5) into both launchers; five-boot re-anchor (T16 law: five-boot checkpoint after every adoption, :1046-1048); F-entry + commit; THEN the depth chapter: one STRATA_SPEC_DEPTH=1 boot on the adopted base, kill-rule table above; W2.5 min-p re-stair only if the rule clears. |
| D4 | G1 pass AND G2 M1 in [+4%, +8%) AND all lanes >= -4% AND G3 pass AND G4 pass | ADOPT — bar tier | Same as D3. The +8% v1 line not met is EXPECTED (T14C re-pricing: central +5.7-6.4% at fixed shape). Depth chapter still allowed but only through the kill-rule (expected value +1.6-3.3% per the lift ladder — spend the boot only if the arm's m4/m5 read > 0.72). |
| D5 | G2 M1 < +4% (inside the band), or any lane < -4%, or the F-231 pair pattern (M1-only win, pair twice at the band floor) | NO ADOPTION | At most one D2 iterate if G1 showed untouched headroom; otherwise D6. Inside the band = not a win (AGENTS.md rule 2). |
| D6 | G3 fails twice (pooled arm < pooled ctl − 1σ, or 2-gate mean < 92), OR G4 < 84.9 twice, OR D2 exhausted | ABANDON | rt2 stays on disk labeled as an instrument; production config was NEVER touched (the swap only executes on D3/D4 — rollback is a no-op); F-entry records the verdict with the position table; the T17C objective-swap re-openers remain queued for a future wave. |
| D7 | G5 finds the F-230 uniform-dip signature | INVESTIGATE before any verdict | bf16-vs-q4 falsifier on the same checkpoint; harvest-contamination check; config re-diff. Fix and re-enter at G1. No adoption while the signature stands. |

Abandon is cheap by construction: the ladder is front-loaded so D2-class deaths cost ~3.5 h of
boots (G0+G1) and never reach the 3v3; a D6 after a full run costs ~7 h total and leaves the
production lane byte-untouched.

---

## 5. Rollback (and adoption) — the mtp/rt dir swap, one command

Safety is structural (T14B P5, :188-193): `--mtp` takes a DIRECTORY; production `rt/` is never
written; `rt2/` and `rt-g0/` are additive. Until D3/D4 executes, there is nothing to roll back —
the standing config is untouched and every arm boot was a config-json copy.

Adoption (one command; both launchers read this same config, launch-custom-iq3s-8081.sh:42 and
launch-custom-iq3s-8081-conc.sh:47):
```
cp custom-iq3_s-8081.json custom-iq3_s-8081.json.pre-t14b \
  && sed 's#/home/svg/LLM/Custom-data/mtp/rt"#/home/svg/LLM/Custom-data/mtp/rt2"#' \
       custom-iq3_s-8081.json.pre-t14b > custom-iq3_s-8081.json
```
Rollback (one command + the mandatory verify boot, AGENTS.md rule 7 "revert cleanly on failure
... VERIFY the restored state"):
```
cp custom-iq3_s-8081.json.pre-t14b custom-iq3_s-8081.json
bash campaign/t17-leg.sh custom-iq3_s-8081.json rollback-verify
```
Verification of the restored state: the boot log's drafter line binds `rt` (MiB printed at
bind, mtp.cpp:528), /health returns 200, and the probe reads M3b back in the standing band
(72.2-76.0). The `.pre-t14b` copy plus git (the config is tracked) double-covers the restore.
Optional cleanup after a settled adoption: delete `rt-g0/` (keep `rt/` forever — it is the
rollback).

Post-adoption obligations (T16): five-boot re-anchor at the current clock bin before any
absolute-milestone claim (the banked means are 2835-class; the 3135 re-bank is still owed,
SVG-CUSTOM-TODO.md:1056-1059); pair/conc means flat at every subsequent rung or the rung
reverts (:1093-1094); the next bits-changing arm re-runs the F-230 coupled-MTP battery + 2x
MMLU per the standing gates law (:1090-1092).

---

## 6. Contradictions encountered, and their resolutions

1. T11.R3's "e2e >= +8%" v1 gate vs T14C's re-priced +5.7-6.4% central expectation and +4% bar.
   Resolution: both lines kept with distinct roles — +4% adopts (campaign law), +8% marks the
   clear-win tier that unlocks the depth chapter. T14C §5-5 amends T11.R3's pricing (237 ->
   223-229 retrain-alone) without overturning the GO.
2. Kill-rule "6-9% marginal acceptance" (T11.R3, F-202 item 5, R35) vs T14C's 0.72/1.18
   tokens-per-offered-row. Resolution: the 6-9% form assumed cheap rows; our c_row (3.56-4.62
   ms, 26-30% of window) makes the true breakeven ~0.72 (M1). The spec implements the T14C
   form and retains "6-9% kill-rule" as the name.
3. "Gates are acceptance + quality, NOT M3b" (T16 W2) vs M3b as a same-bits canary (F-229/F-233)
   and as the acceptance-accounting source (T14C). Resolution: M3b is not a pass/fail quality
   gate anywhere in this ladder; it is (a) the accounting source for G1/G2, and (b) a flatness
   canary ONLY in G0 where the drafter is same-tensors by construction.
4. T14C G0.1's "greedy text MUST be byte-identical" vs F-180's "even the SAME binary produces
   different greedy streams run-to-run" (stateful expert tier). Resolution: bit-check equality
   is the contract and the expected result; on divergence, re-test with a fresh same-binary ctl
   boot FIRST (the F-180 confounder); only a reproducible arm-vs-ctl divergence with stable ctl
   is the kill. Recorded because both are banked laws.
5. MMLU floor argument: AGENTS.md rule 3 says 0.933, T14B P6 says 0.933, the banked 95.0 anchor
   gates ran 91.0, F-180-era gates 0.7, gdnc 0.9. Resolution: 0.933 for both arms (canonical,
   matched); floors 0.7-91 all read 92.9-95.7 on this build, so the arg is not the sensitive
   variable; arm/ctl same-session matching is.
6. mimic instrument scales: T14C G3 quotes the mimicbench v1 band 95.3-98.4; the T16/T17 anchor
   is mimicbench2's 87.9. Resolution: this ladder uses mimicbench2 (the anchor instrument)
   exclusively.
7. T11.R3's "9B-sibling dry-run" vs T14B's finding that no sibling exists on the box.
   Resolution: the stock-round-trip dry-run (G0) is the sanctioned substitute — it tests OUR
   chain end-to-end, which a sibling boot would not (T14B §5, :263).

## 7. Open questions

- The arm's post-retrain offered/window and deep marginals are THE empirical unknowns; G1's
  stair read answers both before any 3v3 is spent (T14C §6).
- c_row has no single-boot direct measurement (T14C §6) — if G5's closure arithmetic misses by
  more than ~1pp, one STRATA_DECODE_TIMING-class stair boot at depth 1 vs 3 vs 5 would pin it.
- Repeated-mimic variance is unbanked; G4's ±1-task tolerance is inference until the first
  paired arm/ctl run adds a second reading of the anchor.
- Whether the conc lane's stale-slot haircut (k=0.63, T14C §1.1) shifts under the retrained
  head — read it from G2's 6-conc telemetry rather than assuming.

## 8. Sources (all file:line verified this session, 2026-10-18)

Journal and ledger:
- SVG-CUSTOM.md:7852-7867 (F-228 stair/per-depth data, c_row mechanism, -1.75% verdict);
  :7869-7882 (F-229 canary +0.13pp, ungated mechanics); :7884-7912 (F-230 acceptance-tax law,
  1.8e-4 KL, -2.7%->-2.8% closure, venv caveat); :7914-7938 (F-231 pair flag, deploy law);
  :7963-7996 (F-233 3v3 protocol, c_row=3.56); :7998-8013 (standing scoreboard anchors);
  :8074-8102 (F-235 closeout); :8104-8136 (F-236 position/content split, -40.5/-14.9pt);
  :5762-5812 (F-180 MMLU watch rule, greedy statefulness method finding); :4807-4876 (F-159
  per-position vector origin); :6985-7048 (F-201 draft-temp, proposals-only class);
  :7076-7080 (F-202 item 5: depth economics 6-9%, NEVER GPTQ, Q4 direct-cast);
  :7788-7807 (F-225 warm-trend law); :3027-3037 (F-108 Proof Law).
- SVG-CUSTOM-TODO.md:793-809 (T11.R3 GO + v1 gates + kill-rule + beta/K + direct-cast);
  :1066-1071 (T16 W2 restate); :1090-1094 (T16 #9 gates law + milestone ladder);
  :1161-1165 (T17C retrain objective swaps); :1226-1236 (T17 verdict retrain v2, quality floor);
  :1238-1268 (STEP 0 RESULTS incl. depth split + parallel-8 + MTP-stays).
- AGENTS.md:181-214 (the performance campaign law, rules 1-8).

Research dossiers (campaign/research/):
- T14C-fastmtp-acceptance-math-gates.md (whole file: c_row §1.3, crossing rules §1.4, scenario
  table §2, the prior gate ladder §3, FastMTP mapping §2.1, contradictions §5).
- T14B-fastmtp-training-infra.md (rt2/rollback P5 :188-193, gates P6 :195-201, export chain
  §2.3/§3-P4, F-80/F-104/F-105/F-109/F-113/F-114 law list §2.4, sibling resolution §5 :263).
- T14A-fastmtp-method-recipe.md:156 (R2 cast-sensitivity falsifier).
- F-201-mtp-agent.md:9 (Q4 direct-cast 71-91% vs GPTQ degradation).

Instruments (campaign/, read in full):
- t17-leg.sh (leg driver: VRAM drain :21-30, F-214 env eval :32-33, F-205 env verify :53-58);
  battery-boot.sh; battery-config.sh (:16-22 env lesson); abab-run.sh (binary-swap 3v3 form);
- probe-lane.py (M1/M2/M3b/M3d extraction :92-149; log seek :60-64); mmlu-pro-gate.py (per-
  request floor :29-37, sigma :59-61, output :19, venv dataset requirement); mimicbench2.py
  (usage :39, weighting :35-37, output :47); bit-check.py (3 prompts, temp 0, sha256);
  depth-lane.py; depth-lane-real.py (repo-prose corpus, 3.27 chars/token :36);
- custom-iq3_s-8081.json (:12-21 spec 6 / min-p 0.5; :24-27 mtp-q4/hnorm; :56-57 --mtp rt);
  campaign/stair50-iq3_s-8081.json (env-dict + per-arm log pattern); launch-custom-iq3s-8081.sh
  (:42) and launch-custom-iq3s-8081-conc.sh (:47) both consume custom-iq3_s-8081.json;
- tools/mtp_pack.py and tools/mtp_rt.py headers (usage, formats, the Gemma +1 fold);
- /home/svg/LLM/Custom-data/mtp/{rt,tensors} layout (verified by listing, 2026-10-18);
- campaign/results/: mmlu-fbench-mmlu-{1,2}.json (floor 91.0, 95.7/94.3),
  mimicbench2-fbench-f233.json + fbench-mimic.txt (58/66 = 87.9), stair50/stair35 and
  s35arm/s35ctl files (F-228/T14C ground truth), depth-t17f-* (the 53.1/93.6/68.0 split).

Methods note: no new research was required (per assignment); every threshold above traces to a
banked F-entry, a campaign-law rule, or the T14C model, and every command line was checked
against the instrument's actual argparse/usage (ports, keys, log handling, output paths).
Unproductive directions avoided by reading first: battery JSONs are boot configs, not results
(T14C methods note); the probe "ms/window" meter is NOT used for c_row (10% disagreement,
T14C §5-2). The G0 "9B sibling" framing was corrected to the round-trip dry-run on T14B's
verified absence finding rather than re-searched.
