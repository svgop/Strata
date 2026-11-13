# Frontier speculative-decoding audit (2026-11-13) — EAGLE-3/MTP/DSpark/block-verify vs our spec stack

Batch 1 of the 18-agent frontier wave. Baseline T17C (2026-10-17) respected; F-352/F-353 falsifications
honored — re-opens only via mechanisms the falsifiers never ran.

## Ranked findings
1. **Batch-lane adaptive verify-depth controller** — the lane that serves ALL traffic runs static D=1 with no
   cost model. Frontier: SmartSpec/TurboSpec goodput sizing; DSpark-in-SGLang (per-request confidence +
   additive cost table T(bs,K), ragged verify under graphs; dynamic trim wins as load grows; Nightjar +14.8%).
   LEGAL F-352 re-open: per-slot confidence-adaptive depth under a measured cost table ≠ the falsified fixed
   D=2. Inputs exist (sl.dprob survival, spec::CostModel). M port. Conc +0-8% band.
2. **Drafter objective swap on the queued FastMTP retrain: LK losses + VAT + AdaSPEC** — satisfies F-353's own
   next-attempt condition ("acceptance-weighted objective, not more steps"). LK: +3.5-8.2% tau, +5.6% on a
   DeepSeek-V3 MTP finetune; VAT: +11.4% acceptance length objective-only; AdaSPEC +15%. Acceptance 0.59 →
   0.65-0.70 = +5-15% EVERY lane; the keystone that legitimately re-opens depth.
3. **Block verification on the sampled lane** (arXiv:2403.10444, vLLM #46781 merged MRV2: +0.96pp AR,
   gains at later draft positions) — composes with our ADOPTED SPEC_PROB; cumulative prefix-ratio inside
   sample_tokens_spec. +1-2pp deep rows. M port, provably output-exact, greedy untouched.
4. **Draft-temp 0.85 is now wrong-shaped** — adopted under exact-match (F-201); under rejection sampling the
   acceptance sum maximizes at DT=1.0. One env boot. +0-1.5% sampled.
5. **Multi-depth/block drafter (GLM-5 shared-param multi-depth AL 2.76 vs 2.55; DeepSeek V4 DSpark +51-400%)**
   — the long-pole replacement; L port, VRAM-priced. Rank below 1-3 on per-point cost.
6. **Hybrid lookup lane armed but asleep** — suffix drafter + DraftPolicy + lookup-chain all in-tree OFF;
   SuffixDecoding 5.3x on agentic traffic; a lookup draft costs no drafter step = a FREE row in batch slots.
   Arming S; batch row M. +5-15% mimic-class.
7. Calibration/observability for the survival numbers controllers eat (DSpark ships sequential temperature
   scaling + ECE + cap-accept censored-ceiling estimator). S, enabling for 1/3/4.
8. Trees stay CLOSED (re-openers recorded: kernel tranche c_row<1.5ms or retrained drafter m<0.6).
9. P-EAGLE cross-slot draft batching — ~0 at D=1, real if 1/2/5 raise D.
10. MagicDec — different regime, no port.

## Already matched
De-serialized drafts (F-206 — the same class llama.cpp #31660 just banked +7-11%); adaptive serial controller
(ADAPTIVE_SPEC); vLLM-form logistic survival estimator (SPEC_LOGISTIC IRLS); distribution-exact sampled
verification ADOPTED (SPEC_PROB +5.8% M1); entropy-gated early stop (SPEC_ENTABORT); hybrid drafting surface
built; exact-match-under-batching vindicated (EqSpec shows vLLM/SGLang degrade where ours holds); window cap
= frontier draft lengths; SURVTOPK = the D-cut shape finding 1 promotes.

Sources: vLLM #46781; arXiv:2403.10444, 2405.19715, 2405.04304, 2503.01840, 2406.14066, 2408.11049,
2602.15763, 2411.04975, 2408.08696, 2510.22876; LMSYS DSpark-SGLang blog; llama.cpp #31660; AWS P-EAGLE;
vLLM #47602; local F-282/F-352/F-353/F-206/F-201/F-30b/F-216 + T17C.
