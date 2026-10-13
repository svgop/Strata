@echo off
rem savage-custom launcher (strata-custom UD-Q4, port 8081; stock install stays at 8080).
rem
rem LAUNCH LAW (learned 2026-10-06): do NOT start this config through START-HERE.bat -
rem setup.py owns the "args" key and REWRITES it on every setup run (#629: "Engine options
rem added by hand to args are not merged"), which silently dropped the whole recipe.
rem Start serve/server.py directly instead (the upstream run-<model>.bat pattern).
rem
rem F-16 incident note: earlier "permanent" env edits silently no-op'd on LF/CRLF anchor
rem mismatches - this file is the explicit full-arms rewrite (2026-10-08, F-18).
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
rem F-18 probe: B8 L2 set-aside over the GDN state (arm line must print at boot).
set STRATA_L2_SETASIDE_MIB=60
cd /d "%~dp0"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "%~dp0strata-unsloth-ud-q4_k_xl.json" --port 8081
if errorlevel 1 pause
