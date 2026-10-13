
### F-178 — THE RE-ANCHOR UNDER CURRENT CONDITIONS + THE ROUND-8 STUDY: a disarmed overlap found, the queue re-ranked (2026-10-14 ~01:00, goal round 8)

**THE RE-ANCHOR (fresh boot, standing recipe, midnight conditions).** Motivated by F-176/F-177's
drift observation (every arm 4-9% below the locked anchor). 3 full probes + 2 MMLU gates:

| metric | run1 | run2 | run3 | MEAN | locked anchor | delta |
|---|---|---|---|---|---|---|
| M1 single | 175.9 | 178.8 | 174.3 | **176.3** | 190.3 | −7.3% |
| M2 pair | 144.4 | 145.9 | 145.7 | **145.3** | 159.4 | −8.8% |
| M2 6-conc | 193.1 | 191.0 | 187.9 | **190.7** | 203.7 | −6.4% |
| M3b rate | 81.5 | 80.2 | 82.2 | **81.3%** | 80.5% | +0.8 pt |
| MMLU-Pro | 92.86±3.08 | 94.29±2.77 | — | **93.6** | 95.71 | −2.1 pts (≈1.5 q / 70; within noise) |

The drift is real, global (all metrics), and NOT GPU throttling (42 °C, zero active throttle
reasons, 0.22 s lifetime SW-power-cap). Candidate: the learned expert profile (re-saved 23:11 by
tonight's 18 probe runs) and/or ambient/system state. **Adoption claims now compare against
re-anchor v2 means; the locked anchor remains the bar reference.** Sources:
`campaign/results/reanchor2-{1,2,3}.txt`, `mmlu-reanchor2-{a,b}.json`.

**THE STUDY (leg A code subagent + leg B ecosystem subagent + operator-side paper reads).**

*LEG A — THE BLOCKER: the adopted F-67 overlap combo is DISARMED ON LINUX.* The launcher exports
`STRATA_ADAPT_OVERLAP=1` + `STRATA_ADAPT_OVERLAP_RESIDENT=1`, but the Linux port kept three arms the
Windows v2 battery dropped as vestigial: `STRATA_ADAPT_LAG=2` (launcher line 9),
`STRATA_DMA_BATCH=1` (line 12), `STRATA_VERIFY_DEVICE_PLAN=1` (line 26) — and the refusal chain at
`generate.cpp:8496-8516` kills the overlap on each (`dma_batch_mode()!=0` :8502, `adapt_lag()!=1`
:8503, adapt-gate refused under device-plan `verify.hpp:263-267`). The overlap never armed on this
box. Windows two-boot evidence: 6-conc 158/159 vs 145-152 baseline (launch-v2-battery,
T1e), i.e. **+4-9% concurrent was left on the table by a launcher bug**. Fix is env-only; verify
the boot line says the adaptive tier's copies land behind the window (not "off: not with
STRATA_DMA_BATCH").

*LEG A — the rest, ranked:* (2) **F-31 slot coupling** — concurrent batch slots run DRAFTLESS
(BATCH_MTP=0; drafts-only was worse, the rescue variant was quality-banned F-30b); coupled drafts are
quality-safe by construction (coupled_draft.hpp:5-6) and the wiring mostly exists — the open defect
is slot verify-row counter staging (verify.cpp:2576) not matching the coupled cells. The only path
to breaking the ~190 t/s concurrent ceiling; +25-50% class; needs code + rebuild.
(3) **R5b tau sweep — DIRECTION CORRECTED: higher tau is STRICTER** (criterion at generate.cpp:281-323:
accept iff q > tau·e^(−H); loosening means 0.15/0.10, not 0.25/0.30 as F-177 assumed). Also queued:
`STRATA_SPEC_RATIO=1` (Leviathan exact test, generate.cpp:325-338) — distribution-exact,
MMLU-safe-by-construction, A/B-pending since F-27. (4) L2 carve: correct site generate.cpp:7719-7727
over the GDN state span (112 MiB span, 60 MiB carve, hitRatio 0.536); 80/96 clamps to the ~75%
persisting ceiling ≈ 72 MiB; single-stream-only lever, +0-2%. (5) Controller verified DEAD
(zero references) and wiring-cheap, but honest verdict **0-3%** for our flat 77-82% acceptance
profile + per-layer fixed-cost wall — the min-p sweep already sits at the optimum it would learn.
(6) **PDL on Linux** (`STRATA_DF_PDL`, verify_kernels.cu:1380-1400): the F-107 retraction was
WDDM-attributed; no WDDM here; +3-8% possible; MANDATORY sustained-reasoning probe (2000-tok
generation) before adoption given the −40% Windows downside.

*LEG B — ecosystem (2025-26 scan).* Top transferable: **FastMTP recursive head re-tuning** (Red Hat
Speculators 0.6.0, Sep 2026): fine-tune native mtp weights recursively with position-decay loss
(beta 0.6 → weights [.51,.31,.18]); on Qwen3-Next-80B-A3B pos-2 acceptance 0.476→0.616 — the
train/serve recursion mismatch is the canonical cause of our 57% distilled plateau; ~443 s on
2×H200 for a 9B-class model. Then: EAGLE-3 fused features (arXiv 2503.01840), draft-vocab trim
(llama.cpp #25187: 621→91 µs head matmul, +1.4-3.1% e2e; 64k-row trim safer than 32k),
L2 evict_first hints (Blackwell-only), clock lock + power cap + **Xid 79 firmware check** (5090
GSP crash thread — verify driver 615.71.09 carries the fix before long runs). Calibration: depth 6
sits on the empirical plateau (arXiv 2607.17283 — speedup flat across K=2-6, decays past ~8);
draft-trees and NVFP4 deprioritized (trees: ggerganov's own position + MoE expert-union cost;
NVFP4: consumer Blackwell lacks the datacenter path and is DENSER than IQ3_S at our batch).
Operator-side reads: FastMTP recipe details (shared embed/lm_head, full vocab, target-generated
data) and the adaptive-γ literature (throughput_B(γ)=B(E[acc]+1)/C(B,γ); matches best fixed γ in
hindsight — but leg A's flat-profile evidence caps its value HERE; the dead Controller is its
natural home if ever wired).

**THE RE-RANKED QUEUE** (expected gain × probability / cost):
- **R9 — un-block the F-67 overlap** [env, free, NOW]: drop ADAPT_LAG/DMA_BATCH/VERIFY_DEVICE_PLAN,
  keep the OVERLAP arms, verify the boot line, 3× probe A/B vs re-anchor v2. Expect +2-6% conc,
  +1-3% M1. Same boot: read the L2 persisting ceiling for the carve-bump read-out.
- **R5b — tau 0.15 / 0.10 + SPEC_RATIO arm** [env]: MMLU gate EVERY arm (looser tau emits drafts
  the target finds more surprising; the batch-site rescue cost 3-4 MMLU points once).
- **R8 — clocks + power cap + Xid 79 check** [sudo, user step] + the iGPU display cable move:
  variance reduction first (it protects every future A/B; may explain part of the drift).
- **R10 — PDL Linux probe** [env]: sustained-reasoning gate mandatory.
- **R11 — F-31 slot coupling** [code+rebuild]: the concurrent ceiling breaker; instrument
  bt_slot_drafts/bt_slot_accepts first, diagnostic boot, fix the counter staging, full battery.
- **R12 — FastMTP recursive re-tune of the stock head** [training]: the structural acceptance lever;
  compose with EAGLE-3 features later; reuse the r17 training-chain tooling from the Windows arc.
- Open bar items unchanged: M5 mimic port, M6 depth + short-class matched anchors, NTFS master sync.
