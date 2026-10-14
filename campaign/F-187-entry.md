
### F-187 — THE LAW WRITTEN, THE BUDGET MEASURED, TWO ARMS TESTED, THE PORT MAP REALITY-CHECKED (2026-10-14 ~14:00, goal round 16)

**THE LAW (operator order, now binding in AGENTS.md, commit 2af8735).** Every perf change: 3+3
same-hour probes, means only; adopt bar +4% targeted / no −4% regression on ANY priority metric;
SINGLE-STREAM PROTECTION explicit (concurrency-for-M1 trades = shelf variants, never adoptions);
sustained-reasoning gate for launch/overlap class; F-entry + commit + result files for wins AND
negatives; verified clean revert; Launch Law.

**THE MEASURED PHASE TABLE (diagnostic boot, VERIFY_PROFILE/DECODE_TIMING; replaces round-9's
byte-model estimates). Batch window (6 slots, 25.95 ms GPU):** GDN: VRAM hits 2.85, hc reads ≈ 4.2
total (hc0/hc1 down 1.14+1.14, up 0.71+0.75, norms 0.45), copy+combine 1.60, router+plan+bell 1.00,
shared+quant 0.89, rec 0.75, q8+qkv 1.17, head 0.84. QSA (12 layers): **attention 4.40 — the
single biggest measured line** (grows 1.6→4.4 with window width: THE 6-conc ceiling), q+q-idx 0.56,
append 0.32. Host: **CPU experts 4.94 ms/window** (the doorbell tail, live). Saved:
campaign/results/diag-phase-table.txt.

**QFUSE — REJECTED.** Arm 176.6/160.7/193.5/79.8 vs same-hour control 171.3/165.0/202.8/80.8:
M1 +3.1% (below bar), 6-conc −4.6% (past the edge) — the F-183 mirror trade. No boot line; arm
engagement unverified. Off.

**PDL — NEUTRAL, SUSTAINED-CLEAN, SHELVED.** Arm 174.0/163.3/206.6/82.7 vs control
171.3/165.0/202.8/80.8: all inside noise (M1 +1.6, conc +1.9), acceptance +1.9 pts; the MANDATORY
sustained gate PASSED (3000 tokens at 156.4 t/s, acceptance 71.9%, no collapse — the Windows −40%
was WDDM, confirmed). A stacking candidate when density work lands; not adopted alone.

**THE FLASHINFER PORT MAP (study agent; source at /tmp/flashinfer-src, main 2895577):**
- **GDN fused decode (v0.7.0)**: plain CUDA, Apache-2.0, geometry matches ours with one -D — but
  ONE TOKEN PER CALL: as-is it replaces 2 nodes with T launches. Port value = the STRUCTURE
  (persistent self-resetting grid barrier + pre-barrier state prefetch) transcribed into OUR
  multi-token kernel. Honest ceiling ~−36 nodes (~2%) + µs/layer. STRATA_GDN_FUSED plan written.
- **W4A16 fused MoE (v0.6.17)**: DISQUALIFIED near-term — no C++ artifact (CuTe-DSL Python only),
  FP4-packed weights only (no IQ3_S path), adoption = re-quant (+~30% expert VRAM) + DSL toolchain.
- **XQA ragged-Q (v0.6.16)**: INCOMPATIBLE — full-context paged attention cannot serve our
  top-2051 CELL-SELECTED sparse attention; page size 4 vs required 16-128; int8+FWHT-rotated KV
  unsupported. The ragged-length INTERFACE is the reusable idea, inside OUR kernel.

**QUEUE v4 (re-ordered by MEASURED value):**
- R18 [env]: ROUTE_RESIDENT margin sweep — targets the measured 4.94 ms CPU-expert doorbell tail
  (~15-18% of window). MMLU-gated.
- R19 [code]: QSA attention kernel — the 4.40 ms line and its width growth; ragged rows inside our
  chunk+merge structure (XQA's idea, our semantics).
- R20 [code]: hc-read fusion (4.2 ms measured across 6 launches/layer; QFUSE's direction, done as
  a proper norm+down+up merge) — and/or the hc Q8 byte cut (needs a re-quant with Q8_0 hc tensors,
  mechanically possible per the pack side).
- R21 [code]: DuetServe look-ahead launch + launch-completion events.
- R22 [code]: ECHO-style per-row verify budget on the slot path.
- R23 [code]: layered prefill (layer-group interleave).
- R24 [deferred]: GDN fused-structure transcription (~2%); FastMTP+KV-injection [training] stays.
Standing: merged 0.1.44 + BATCH_MTP=1 + batch 6; restored, verified clean (no arm/diag env).
