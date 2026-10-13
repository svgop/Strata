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
rem STRATA_ADAPT_LAG=2 - kills the #463 lag-1 event sync stall in the adapt tier (F-12 wave).
set STRATA_ADAPT_LAG=2
rem STRATA_DMA_BATCH=1 - 84.9% fewer DMA submit cycles (F-12 wave).
set STRATA_DMA_BATCH=1
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
rem F-36 arm: device-side routing plan when all experts resident - skips the host plan
rem   publish round-trip (default OFF).
set STRATA_VERIFY_DEVICE_PLAN=1
set STRATA_VERIFY_PROFILE=1

cd /d "%~dp0"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "%~dp0strata-iq3_s.json" --port 8081
if errorlevel 1 pause
