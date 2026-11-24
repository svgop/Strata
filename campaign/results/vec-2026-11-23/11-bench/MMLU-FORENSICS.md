# MMLU 90.0 — consolidated forensic verdict (18 angles, 6 investigators + lead)

Incident: the first MMLU-Pro 70q gate after adopting the persona arm read 63/70 = 90.0% (prior
gates on the identical fixed suite: 66/70 = 94.3, 65/70 = 92.9). Six read-only investigators, three
angles each (A: steering math — silent failure, angles covered by D/E/lead; B: graph capture —
retry silent failure, angles covered by E + lead's code read; C: decode/gate; D: residual algebra;
E: loader/handshake; F: statistics/provenance), plus the lead's own dive.

## The verdict

**No wiring defect. The persona add is real and slightly over-dosed; the gate's dominant loss is a
reasoning-budget truncation artifact the persona makes more likely; the 3-question delta is within
the instrument's noise either way.**

| angle | finding | verdict |
|---|---|---|
| D1 composition orthogonality | cos(persona add, refusal-r dir8) = −0.007; refusal-r is ONE global direction (47 rows bit-identical) and IS a build axis; 99.997% of the add survives | CLEAR |
| D2 refusal-r unchanged | config-diff: same file, scale, range, mode as pre-adoption table 1; byte-equivalent | CLEAR |
| E1 loader second-table path | type/width checks, project math, have2 bounds all match table 1; 47/47 rows unit-norm | CLEAR |
| E2/E3 summary string, parser | every consumer mode-agnostic; arg consumption exact; gate ran FULLY ARMED (no key sent, absent = on) | CLEAR |
| C2 flag/phase state | per-request flag applied at every serial start, no stale path; phase steering NOT armed (and unimplemented server-side in this tree) | CLEAR (latent slot-path defect noted) |
| B graph/pointer stability (lead) | capture strictly after both uploads; `upload2` rebinds pointers; t2-only launch gets real bo/inj (the fixed bug) | CLEAR |
| **C1 thinking force-close** | **3 of 7 fails (idx 30, 55, 59) force-closed at exactly 3001 tokens** — the server caps reasoning at 4000−1000 = 3000 then injects a wrap-up. 3 of 4 wrapped questions failed (Fisher p=0.0024); unwrapped: 4/66 fail. Without them: 66/70 = 94.3 | **SUSPECT (primary)** |
| **D3 dose vs the block's read** | each stream is RMS-normalised independently before mixing, so the add's share scales with the HARMONIC mean of stream norms (H=0.647), not the arithmetic mean (1.028) the build used: **effective block-input dose 0.462/0.647 = 0.713 = 1.59× nominal**, per stream 132%/96%/25%/32% of own norm; above the measured plain-row knee (0.6) | **SUSPECT (mechanism behind C1)** |
| F1/F2 statistics | exact P(X≤63 \| p=0.943)=0.10; 90.0 is the joint 2nd-lowest of 41 floor-0.933 gates (range 90.0–98.6, SD 2.9); 70q has 16–40% power for a true 3-pt drop (MDE 8.6–10.5 pts); needs ~534 q for a one-shot verdict | noise band, underpowered |
| F3 learned profile | this boot loaded the GOLD profile; learned file write-only in this config; F-271 cleared | CLEAR |
| C3 spec decode at T=0.1 | acceptance exact for the steered target, but typical-rescue re-accepts 7.8% of first rejections (drafter-directed); armed equally in priors; off at T=0 | UNCERTAIN (±2 q) |
| E1-latent prefill fuse | fused path passed table 1's `mode` with table 2's pointers → would ADD refusal-r on 46 layers under STRATA_CVEC_FUSE (AMD default) | **BUG — FIXED** (commit below) |

## Mechanism, in one paragraph

The persona row was norm-calibrated on the arithmetic mean of the four hyper-connection stream
norms. But every block RMS-normalises each stream on its own before the gated mix, so stream scale
is discarded before the read and the small streams count fully — the add's effective share of the
block input is set by the harmonic mean, making the true dose 1.59× nominal, on every token from
layer 9 to the final mixer. At that dose the persona lengthens reasoning (its measured effect on
everyday prompts is compression; on reasoning it is elaboration), and the server force-closes
thinking at 3000 generated tokens. Three of the four questions that got wrapped were then answered
wrong — clean wrong letters, no extraction failures. Strip the wrapped questions and the gate reads
94.3, identical to the pre-adoption run.

## Decisive experiment (one boot, interleaved, no rebuild)

Per-request arm via `experimental_speed_projection: false` — 70q ON vs 70q OFF in the same boot,
alternating, with `max_tokens ≥ 8192` (so the budget cap cannot fire), rows carrying content +
finish_reason + generated-token count. Prediction if the mechanism is right: ON ≈ OFF at the
raised budget; at the default budget ON wraps more and loses ~3. Second falsifier, if wanted:
re-gate at dose 0.3 (the banked window; effective 1.06× nominal instead of 1.59×).

## Fixes landed / queued

- LANDED: `src/prefill/prefill.cpp` — the fused cvec path now carries the fused table's own mode
  (`cv_mode`), not table 1's. Parity all-passed; binary redeployed (`48a3f62380d1803b`).
- QUEUED (design, not built): per-table enable (`cvec_set_enabled2`) so the persona can be
  phase-scoped to the answer while refusal-r stays constant — the "right token, right time" form.
  NOT the fix for this incident (the loss is truncation, not reasoning-phase contamination), but
  the correct long-run shape once the dose is re-calibrated.
- QUEUED: harmonic-mean dose calibration in `build_vector` (dose defined in the block's read
  domain, not the stream-mean domain).
- QUEUED: gate rows should store content/finish/tokens (the JSON alone could not split artifact
  from wrong; the lead recovered it from the engine log); `tools/test_handshake.py` has zero
  two-table cases; `web/app.js` renders the composed cvec string garbled (display only).
