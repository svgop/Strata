@echo off
rem savage-custom v2 launcher (strata-custom-2 = v0.1.43 base + the MAGMA layer, port 8082;
rem the v1 lane stays at 8081, stock at 8080).
rem
rem LAUNCH LAW (learned 2026-10-06): do NOT start this config through START-HERE.bat -
rem setup.py owns the "args" key and REWRITES it on every setup run (#629: "Engine options
rem added by hand to args are not merged"), which silently dropped the whole recipe.
rem Start serve/server.py directly instead (the upstream run-<model>.bat pattern).
rem
rem F-64 REBASE LAW (2026-10-11): v0.1.43's Windows default reads the file tier unbuffered,
rem bypassing the warm page cache - a big-RAM regression (52-58 vs 138-167 t/s serial);
rem STRATA_KEEP_MAPPING=1 restores parity (147-148). Irrelevant to the pinned complement in
rem practice, but every launcher built on .43+ carries it by law.
set STRATA_KEEP_MAPPING=1
rem --- standing arms (each measured; see SVG-CUSTOM F-05/F-12/F-15b/F-16) ---
rem STRATA_TYPICAL_ACCEPT=0.2 - Medusa typical acceptance (buun 1ffba5224 port; sampled
rem   traffic only; greedy stays stock exact-match). 0 = off arm. Overridable per launch.
if not defined STRATA_TYPICAL_ACCEPT set STRATA_TYPICAL_ACCEPT=0.2
rem STRATA_POOL_SPIN_US=100 - pool workers park after 100us instead of 20ms (#921).
set STRATA_POOL_SPIN_US=100
rem STRATA_KV_PREFETCH=1 - in-repo measured +5.3%/4.1% prefill at 32K/64K (docs/KV_PREFETCH.md)
set STRATA_KV_PREFETCH=1
rem F-67: ADAPT_LAG dropped - vestigial on .43 native code (T1d ablation: no loss)
rem F-67: DMA_BATCH dropped - vestigial on .43 native code (T1d ablation: no loss)
rem F-16 remediated prediction-credit (evidence-gated lane, proven at 103k: 128 t/s / 97.6%).
set STRATA_PREDICT_CREDIT=1.0
set STRATA_PREDICT_BUDGET=2
rem F-15b B6 victim protection (healthy at depth: 104-126 t/s @ 100-145k live agent traffic).
set STRATA_ADAPT_PINCAP=48
set STRATA_ADAPT_MIN_AGE=2
rem F-22: never-routed experts become evictable after 64 adapt rounds (needs the profile-save heat).
set STRATA_ADAPT_DORMANT=64
rem F-23 X-1: exchange rotation (needs the resident complement; +8.1% upstream anchor).
set STRATA_EXCHANGE_ROTATE=1
rem F-18 probe: B8 L2 set-aside over the GDN state (arm line must print at boot).
set STRATA_L2_SETASIDE_MIB=60
rem F-30c STANDING: 0 - SUSPENDED under the No-Regression Law. With the rescue off (quality
rem   law) drafts alone give pair 17.2-17.9s vs 16.5s draftless (-6%); coupled drafts engage on
rem   the serial path (83.9% acceptance) but NOT yet in slots (counter/cell staging - F-31 open).
rem   Flip to 1 only when the slot-side coupled counter work lands and gates clean.
set STRATA_BATCH_MTP=0
rem F-30b/F-31: the batch-site rescue is OFF (No-Regression Law: it was the only distribution-
rem   bending step in slots; armed, MMLU dropped 95.7 -> 91.4-92.9). Coupled drafts: the draft
rem   row samples with the target's chain + same Philox uniform (unbiased by construction,
rem   acceptance ~ sum min(p,q)) - works on the serial path, slot-side pending (F-31).
set STRATA_BATCH_RESCUE=0
set STRATA_SPEC_COUPLED=1
rem F-35 arm: batched PLE key/value projections for batch windows (verify.cpp; default OFF
rem   upstream). Batch-path fixed-cost lever.
set STRATA_PLE_BATCH=1
rem F-67: VERIFY_DEVICE_PLAN dropped - incompatible with the adopted overlap gate.
rem F-67 ADOPT: ADAPT_OVERLAP resident combo (T1e, two-boot confirmed: 6-conc 158/159
rem   vs 145-152 baseline, pair 11.37 = v1 class, depth prefill 6612-6719 night-best,
rem   mimic 100% campaign record, tokens identical by construction per its gate comment).
set STRATA_ADAPT_OVERLAP=1
set STRATA_ADAPT_OVERLAP_RESIDENT=1
rem F-68 ADOPT: STRATA_ROUTE_RESIDENT=0.5 - tail-rank (6-9) non-resident experts swapped
rem   for resident ones within 0.5 logits (output-bending, quality-gated): the doorbell-
rem   elimination lever. Serial 187 t/s (+28%), pair 9.17 (+24%), 6-conc 189 (+19%), depth
rem   7557/134.4; MMLU 95.7/95.7 (above the 94.3 v1 class), mimic 95.3/96.9 (in-band),
rem   compliance 12/12, zero empties. Windows: CPU experts 3.90->1.75 ms, hits 97%+.
set STRATA_ROUTE_RESIDENT=0.5
rem F-72 ADOPT: ROUTE_RESIDENT_MTP=1 - the drafter's router gets the same residency bias
rem   (L3-legal: lossiness in the drafter only; the verifier still decides). Draft
rem   acceptance 85% (vs ~70% standing), MMLU 95.7/94.3 in-class, speed neutral-in-drift,
rem   no regression - mechanism-adopted; sets up the R17/deeper-window acceptance axis.
set STRATA_ROUTE_RESIDENT_MTP=1
rem F-73 ADOPT: STRATA_DF_PDL=1 - upstream's programmatic dependent launch (pdl.hpp, sm_90+):
rem   kernels overlap their predecessor's epilogue+launch latency; bitwise by construction
rem   (weights-only prefetch before pdl_wait) with upstream's own parity harness. Two boots:
rem   depth decode 119.5/117.4 (window class 105-116), pair 9.85 first boot (best of the
rem   drifted window), MMLU 95.7/94.3 in-class. The F-71 'weeks-class build' was already
rem   shipped in .43 - this arms it.
rem F-107 RETRACT: DF_PDL=0 - it cost 40% on sustained reasoning generation (79.6 vs 136.8 t/s
rem   on the matched probe; the WDDM programmatic-submission cost the risk register named)
set STRATA_DF_PDL=0
rem ---------------------------------------------------------------------------
rem THE DEGRADATION HUNT (2026-10-12, temporary - remove once the stage is named).
rem F-119 measured the process degrading ~10x over ~30 minutes under traffic (10.3 t/s
rem aged vs 101.6 fresh) while the expert-cache hit rate stayed at 95-98% and the GPU sat
rem near idle. Every state counter was tested against the captured degraded boot and
rem EXONERATED: the GOOD 01:11 boot ended with 55,260 adaptive exchanges and 349
rem conversation evictions at 17.8 ms/window, while the DEGRADED 16:29 boot had only
rem 29,526 and 111 at 73.7 ms/window (Pearson r 0.82 was correlation with time, not cause);
rem KV index lookups per request were 3.5-5.9M on the good boot vs 0.4-1.5M on the bad one;
rem commit stayed at 103-105 GiB free; the GPU showed no throttle; total CPU load was 13%.
rem What is left is WHERE the time goes: GPU wait vs pool/plan vs host staging vs commit.
rem These two instruments print exactly that per window and per request, so the next
rem degradation episode - whoever boots the lane - captures its own diagnosis.
rem STRATA_SPLIT_TIMING: "stage N: ... per window: wait for the GPU X ms, pool + plan Y ms,
rem host staging Z ms, commit W ms" (generate.cpp:12630).
rem STRATA_DECODE_TIMING: the host-side per-request breakdown (generate.cpp:11367).
set STRATA_SPLIT_TIMING=1
set STRATA_DECODE_TIMING=1
cd /d "%~dp0"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "%~dp0strata-iq3_s-sfx.json" --port 8081
if errorlevel 1 pause
