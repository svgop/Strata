# r02 verdict record — persona WAR vector r02 (multi-slot add, direction.18–26)

Vector: `/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-persona-war-r02-emb.gguf` — 9 unit rows
(direction.18–26), add mode, dose ρ × mean‖h_slot‖ per layer. Build: CAA-style teacher-forced
difference-in-means, refusal-orthogonalized. All tests blind, swap-balanced (both orders, ON vs OFF
in one boot), temperature-0 pairs + all-orders pairs, exact sign test, McNemar for the factual suite.

## Stock template (mechanism-isolation context; `01-r02-dose/`)

| dose | verdict | assertive | serious | direct | facts (off→on) | words | pos-bias A |
|---|---|---|---|---|---|---|---|
| 0.08 | MIXED | 5/6 p=0.726 | **11/1 p=0.003** | 4/5 p=0.746 | 1.00→1.00 | 167→163 | 1.00 |
| 0.35 | DAMAGE-FIRST | 0/18 p=1.0 | 4/11 p=0.982 | 1/15 p=1.0 | 1.00→0.688 | 163→18 | 1.00 |
| 1.0 | DAMAGE-FIRST | 0/16 p=1.0 | 2/14 p=1.0 | 0/16 p=1.0 | 1.00→0.00 | 165→5 | 1.00 |

Mechanism at 0.08: chatty openers drop on serious tasks (the "serious" score) but the trait does not
generalize — jokes still told when asked (assertive null). At 0.35 the capability gate fails FIRST
(facts 0.69, mean 18 words = premature EOS, arithmetic wrong); persona never saturates before
damage begins. The dose window is a needle: no dose achieved trait-moved + capability-intact.

## Deployed template (decision context, user's chat_template.jinja, max effort, thinking on; `03-r02-deployed/`)

| dose | verdict | assertive | serious | direct | facts | words | pos-bias A |
|---|---|---|---|---|---|---|---|
| 0.08 | MIXED (null) | 9/7 p=0.402 | 4/5 p=0.746 | 6/8 p=0.788 | 1.00→1.00 | 234→261 | 0.79 |

Verdict: **r02 does not earn its place in the deployed configuration.** Null across all three
criteria at the only survivable dose. Power caveat: 18 temperature-0 pairs; the consistency filter
is conservative; this rules out large effects, not small ones — but "large effect, small power" is
the only case r02 needed, and it lost.

## Judge calibration facts

- Position bias is severe and context-dependent: A-rate 1.00 (stock) vs 0.79 (deployed) — the
  deployed template largely cures first-position bias. Stock-template criteria scores are polluted
  by the 1.00 A-rate; the serious@0.08 p=0.003 (11/1) survives the mechanism read (openers
  verifiably change) but criterion-level stock numbers should be treated as position-bias suspect.
- Temperature-0 on this engine is not deterministic (determinism_ok false, identical_rate 0.0):
  identity-based inertness checks are meaningless here; all inertness claims come from the paired
  factual suite and finish/degeneration counters, both clean at 0.08.

## State of record

- Fences PASS across both result dirs (standing config, launcher, engine binary unchanged;
  learned profile untouched).
- Negative banked with mechanism: multi-slot wide add (18–26) at this construction has a
  damage-before-saturation dose-response. Re-open condition: a single-slot vector built from the
  r03 capture (personality-contrast residuals, slots chosen by causal prominence), or per-stream
  steering (requires the engine tweak), or SAE/route-level interventions.
- Next: r03 from the fresh capture (72 boots, `02-capture.log`) → base-context dose ladder →
  deployed-context ON/OFF decision test at the best dose. Three external-research lanes
  (steering methods, low-level stream/tensor engineering, alternatives+evaluation) running
  in parallel; their memos land in `campaign/research/`.
