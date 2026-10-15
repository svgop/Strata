# Contributing upstream — PR plan (banked F-209)

Target: Niko1221/Strata. Vehicle: fork `svgop/Strata`, branch off current upstream `main`
(0.1.44.1+). Mechanics: cherry-pick the engine hunks onto a fresh branch — never push the
campaign history (journal/tooling noise). One variable per PR. Evidence block per PR:
same-night ABAB means vs stock, MMLU/M3b gate results, bit-identical token-stream proof
wherever the output path is touched. Scrub the local API key from any tooling before sharing.

## Tier 1 — small, evidence-backed, highest acceptance odds
- **P1 hc wide-grid fix** (T1.4, implement first): 1-warp-per-row, 324 blocks vs 41,
  lane chunk order preserved = bitwise identical. ~30 lines, pure perf, no behavior change.
- **P2 tuned defaults** (each 1–5 lines, all measured vs stock same-night):
  - `STRATA_SPEC_DRAFT_TEMP=0.85` — F-201: M1 +13.4 / pair +9.8 / conc +7.5%
  - `ROUTE_RESIDENT 0.5→0.75` — F-200: +14.9/+8.5/+12.2%, 2x MMLU gate PASS (92.86/97.14)
  - `TYPICAL_DEVICE=1` — F-200: +9.0/+6.7/+6.4%
  - `VERIFY_ASYNC_LAUNCH=1` — F-200: +4.9/+7.9/+8.9%
  Bundle as one "tuned defaults" PR or split per flag; Niko's call.
- **P3 coupled slot drafts** — F-184: pair +11.5% for one env line.

## Tier 2 — medium code changes, kill-switches included
- **P4 commit parity double buffer** — F-196: pair +2.4 / conc +2.2%, acceptance identical;
  `STRATA_COMMIT_SYNC` kill-switch kept.
- **P5 R33 split draft** — F-206: conc +2.4% (219.1 = record), two-pass emit loop
  (draft_launch/draft_finish on MtpDrafter); mechanism predicts M1 flat, confirmed.

## Tier 3 — big, open an issue/RFC first
- **P6 Linux/CUDA port hardening** — F-172 build fixes + Linux lane work, F-67 overlap
  disarm on Linux, F-193 stack/link findings. Highest value to the project, largest review
  burden; upstream is Windows/Metal-forward. Lead with an issue showing the Linux matrix.
- **P7 doorbell host service thread** — the one-week rewrite pick (single service thread +
  batch G=1 serialization, verify.cpp:3080/:882). RFC issue, not a PR.

## Do NOT PR
Campaign tooling/journal/launchers; gpu-lock daemon (maybe `contrib/tools` later);
box-specific bins (2835); every env-gated negative (LFUSE, FUSE_WAIT, DF_PLE, HC_SPLIT=0,
DOORBELL_STORE, spec-8, PDL, GUMBEL, aux-cpus).

## Coordination
pre75-on-sm_120 is upstream's own open PR **#2075** — test it, comment with our sm_120
numbers, do not duplicate it with our own PR. Same courtesy for #646/#783/#2514/#2326.

## Sequencing
P1 lands for us first (it's already in the queue as T1.4) → PR immediately after adoption.
P2/P3 can be cut any time (defaults already live in our launcher). P4/P5 after their code
is re-verified on the current base. P6/P7 = issues now, code later.
