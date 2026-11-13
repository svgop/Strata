# Frontier weight-space operations audit (2026-11-13) — RCO/GSQ/REAP/merging/trellis vs our packs

Batch 3 of the 18-agent frontier wave. The lane frame: byte-bound at 49.5-50 GB/s; the CPU pool at 84-97% of
RAM bandwidth (kernel tuning gives back 2-5%, never adds); every GiB of expert blobs = +512 VRAM slots =
+2.1pt hit. DECISIVE FACT: the IQ3_S pack is ALREADY per-tensor mixed-format and the engine consumes that
heterogeneity natively (native_experts.txt).

## Ranked findings
1. **RCO re-bake at EXPERT granularity with OUR traffic as the budget signal** — run IST-DASLab's own RCO
   (exact-budget assignment, Apache-2.0, 3-16x cheaper than evolutionary) per-expert instead of per-layer;
   chooser pass first over EXISTING formats. TOOLING ONLY (all formats have kernels). T17F prior art priced
   a chooser at −7-13% blob = +6-10% M1; compounding on hits AND miss lines. Hazards named (GEMQ calibration-
   domain collapse; router shift, <0.04%-params fix). 1-2 CPU-days/pass; one-boot falsifier (12-layer band,
   G2 canary + M3b A/B). Cheapest highest-EV item on the board.
2. **KILL-LINE: expert MERGING** — headline 80% memory is T5/MT0-era; LLM-scale evidence inverts it (REAP:
   HC-SMoE at 50% falls to 58.8-65.2% of baseline; merging costs 2-5pt even at 25%; mechanism = functional
   subspace collapse). 512 experts x high router variability = our worst case. FATAL to every floor. KILL.
3. **GSQ grid re-learning — split verdict** — chooser+imatrix (GO, = finding 1); grid re-learning inside
   i-quant formats (real but gated: their +6.25 Q2_K existence proof; ~900+ 5090-hours at our scale or
   per-layer-band over weeks; GPQA-class regression warns MMLU-Pro; phase 2 only if the chooser shows the
   FORMAT SET is binding, not the assignment).
4. **Graded shadow store (per-lane quant allocation generalized)** — the shadow already ships (+7.7 pair/
   +8.4 conc adopted); grade the shadow blobs by miss-frequency (iq_pack.py already supports per-layer
   mixed blob sizes) = tooling only, deepens the miss lane, shrinks the shadow's RAM (opens the mechanism
   for low-RAM boxes). MoPEQ INVERSION warning: high-frequency experts tolerated LOWER bits in their study —
   let the scorer decide, don't assume hot=precise. LOW risk (misses are ~3% of picks; the uniform-Q2_0
   gate already passed at 95.71 vs 93.3).
5. **Per-EXPERT sensitivity scorer from routing telemetry** — REAP-style gate-value x output-magnitude over
   our teacher-forced harness (frequency-only disproven for generative quality); tooling only; FIRST ACTION
   = measure the sensitivity SPREAD (flat spread kills the whole allocation program). De-risks 1/4/6.
6. **REAP-class partial prune 10-15%** — the only structural cut with surviving evidence (25% within 1pt
   upstream; engine path PROVEN by the Coder pack running 256/512); 50% dead; the #438 CJK damage is the
   boundary warning. Gated on 5 + full floor batteries. Irreversible at the artifact level.
7. **KILL-LINE: trellis/VQ own-bakes** — CONFLICT RESOLVED against the llamacpp-delta digest: WE ALREADY
   SERVE the trellis family (Gyro TQ1_0/TQ2_0, ported kernels, docs/GYRO.md) and MEASURED it: Gyro-M halves
   KL vs Q2_0 BUT decodes 17-21% SLOWER and scored 49/60 vs 57/60 on the reasoning set; Gyro-S −24% bytes
   at 13% worse KL. In-tree measurement > ik's PR claim (their TG gain is vs THEIR baseline kernels).
   KILL own-bakes; passive watch on agentionai decode kernels; re-price Gyro-M only as a low-RAM tier.
8. **KILL-LINE: low-rank+sparse / shared-basis factorization** (LoSparse/MoBE/Sub-MoE) — 1-2pt avg drops on
   tenths-of-a-point floor margins; SVD effective-rank limits concentrate damage in exactly our wide expert
   matrices; new-format kernel contract on kernels already at 84-97% bandwidth. KILL (re-price only if a
   shared-codebook GGUF ships in-tree with kernels).
9. **KILL-LINE: pack-to-pack / MoE-to-dense distillation** — 4B-token KD = ~250x our retrain scale;
   unquantifiable quality. KILL; KRAFTON result banked as reference.

## Sequencing
5 (scorer + spread measurement) -> 1 (RCO chooser re-bake) -> 4 (graded shadow) -> 6 (partial prune, gated)
-> 3b (grid re-learning, gated). Kill-lines 2/7/8/9 stand unless their named falsifiers change.

Sources: RCO arXiv:2605.00649 + repo; GSQ arXiv:2604.18556; REAP arXiv:2510.13999 + Cerebras blog;
MC-SMoE arXiv:2310.01334; HC-SMoE arXiv:2410.08589; GEMQ arXiv:2605.23078; MoPEQ arXiv:2509.02512;
QTIP arXiv:2406.11235; QuIP# arXiv:2405.04532; MoBE arXiv:2508.05257; LoSparse arXiv:2306.11222; Sub-MoE
AAAI'26; KRAFTON arXiv:2605.28207; EdgeMoE arXiv:2308.14352; Fiddler arXiv:2402.07033; Mixtral-offload
arXiv:2312.17238; QMoE MLSys'24; Not-All-Experts arXiv:2402.14800; MoE-compression survey arXiv:2406.02500;
in-repo SHADOW_STORE/GYRO/QUANTS/MODELS/DETAILS + T17F + train_cycle.
