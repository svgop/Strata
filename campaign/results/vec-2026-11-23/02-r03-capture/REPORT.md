# r03 capture — result, audit, and the re-priced falsifier

Source: `capture --elicit campaign/results/vec-2026-11-23/01-r02-dose/elicit.jsonl --per-cell 12`
→ 72 boots (3 traits × 12 cells × ±2 arms), 58–85 prompt ids per boot, responses ≤ 200 tokens,
exit 0, no failed boots. Artifacts in the data home
(`/home/svg/LLM/Custom-data/persona-vectors/capture/r03-run1/`, means.npz sha16 `55be45efa4d6cef2`);
small evidence copied here (`report.json`, `runs.jsonl`). Elicit rows hash-pinned (`elicit.sha256`).

## Verdict: NO CANDIDATE — pre-registered filters, all three traits

Filters (pre-registered in `campaign/vector-capture.py` before the run): prominence > 1.25,
slot ≤ 28, promotion pct ≤ 0.75, energy ≤ 0.10, stream-pair cos ≥ 0.8, split-half reliability.
Best available per trait: assertive slot 1 (prom 1.146, d′ 30.2, cos_min 0.427; **slot 1 is not
steerable** — slot k maps to direction.k−1), effort slot 25 (1.091), resourceful slot 25 (1.150).
Nothing passed; nothing came close on coherence. No threshold relaxation, no noise vectors.

## Mechanism — measured here, corrected by the external audits

1. **The persona contrast is diffuse in depth.** prominence = d′_slot / median d′ across slots 1–44.
   Best 1.15 vs gate 1.25: no structural locus; d′ is high everywhere (17–30) because teacher-forced
   ±arms differ in response tokens at every position — generic separability, not a persona circuit.
   The peakedness filter exists to reject exactly this, and it did. (2611.01493 TopoSteer:
   separability-based and causal-based layer rankings can disagree — the causal ladder, not the
   d′ table, is the arbiter.)

2. **Stream decomposition, measured directly** (audit on means.npz, signed per-stream alignment of
   the shared mean direction): mean useful-energy fraction ≈ 0.50–0.59 at every probed slot — half
   the injected dose energy lands on the stream's own contrast axis. The report's low
   `stream_pair_cos_min` (≈0.0 in the 22–28 band) is driven by **stream 3 (index 2, the
   largest-norm stream, ‖h‖≈2.0–2.9)**: its contrast direction misaligns with the shared direction
   there (a ≈ 0.32–0.41) while streams 1, 2, 4 stay aligned (a ≈ 0.78–0.92). Stream 4 (index 3)
   carries the largest contrast share (≈0.40). **Slot 9 (direction.8) is the coherence pocket the
   pairwise filter hid**: the only slot where all four streams align ≥ 0.65
   (assertive a = [0.71, 0.65, 0.68, 0.92]).

3. **The dose axis was structurally mispriced** (audit 2409.19606 Hyper-Connections): the block
   output is broadcast to all 4 streams and the final read-out is the row-wise SUM of streams —
   at equal α the logits see ~**4×** the read-out dose a dense-model paper assumes, while the
   block-input read (A_m, ~convex weights) sees ~1×. r02's damage-first at dose 0.35 was
   read-out ≈ 1.4 effective. This re-opens the dose search at ¼ the bracket, no engine change
   needed (the ladder dose is the multiplier).

4. **Facts-fail-first has a named mechanism** (2608.21236 RARE): the router reads semantic content
   off the residual stream, so router-visible steering components corrupt expert selection before
   persona saturates. RARE's remedy — project the perturbation onto the router null space
   (loader-only, one GEMM/row) — recovered steering at 67.8% MMLU retention in their setting.

5. **Per-stream direction vectors are NOT the fix** (audit V2, mHC 2512.24880 / oHC 2609.02672):
   per-stream differential vectors are architecturally damped by the mixing matrix or chaotically
   remixed layer over layer. The shared direction is the right form; the dose and the router
   component are the levers.

## Consequences and re-open conditions

- r02's construction is explained and superseded: a 9-slot band add, unit rows, dose axis
  uncorrected for the 4× read-out — structurally incapable of a wide safe window.
- Re-open conditions, in order of cost: (T1) read-out-matched dosing — lab-only, zero engine
  change; (T2) router-null-space projection at build — loader/build change, free offline falsifier
  (‖R·v‖/‖v‖ per layer on the existing table); (T4) mid-band restriction / single-layer causal
  sweep; (T5) SteerMoE-style expert-mask channel on the existing expert-profile machinery — zero
  residual perturbation, lowest capability risk of all lanes (free falsifier: contrastive
  activation-frequency diff on 100 paired pairs).
- Non-vector lanes are now priced by the alternatives audit
  (`../../../research/2026-11-23-external-alternatives-audit.md`): prompt/template search scored by
  trait-vector projection (cheapest; Anthropic preliminary: prompting ≈ feature steering), screened
  LoRA (conditional GO, gated behind the cheaper tiers).

## The falsifier now running

Slot 9 (direction.8) single-row assertive vector, stock context (mechanism isolation; the deployed
ON/OFF test remains the ship decision), doses 0.03 / 0.1 / 0.3 / 0.6 — read-out-matched bracket:
covers ¼ of r02's surviving window through r02's 0.08 zone and past it, on the best-coherence
slot. Output: `../04-r03-slot9/` + `04-r03-slot9.log`. Success shape: trait criterion moving at
p < 0.05 with facts 1.00 and no truncation — evidence the window exists on a clean single slot.
If the whole bracket is null-or-damage, dense shared-add is closed on this model and T2/T5 (or the
non-vector lanes) take the queue.
