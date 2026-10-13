@echo off
rem ============================================================================
rem DIAGNOSTIC ONLY - NOT AN ADOPTION CANDIDATE (2026-10-12, round 2 of the 2x goal)
rem
rem THE QUESTION: at margin 0.5 the per-window CPU-expert term grows with the window
rem depth T and eats the whole draft-yield gain (measured 2026-10-12 17:29 on boot
rem 17:17:30: T 2.7 -> CPU 4.7 ms/win at 159 t/s; T 4.0 -> CPU 7.1 ms at 184 t/s; a
rem 4.38-T window spent 13.56 of 29.12 ms inside the CPU expert path, 3.79 CPU experts
rem per layer-window, ~75 us each). If a stronger residency bias collapses that term,
rem the mechanism is proven and the L3-legal drafter-only version (a code change: a
rem separate margin for the MTP drafter, whose lossiness can never touch quality) is
rem the next build.
rem
rem WHY THIS CONFIG IS DIAGNOSTIC: margin 1.0 + ranks 4-9 is the F-69 frontier point
rem that measured 6-conc 205 / depth decode 145 - the fastest numbers this build has -
rem at a measured quality cost (MMLU 92.9 vs the 95.7 class). It is LOSSY. It is used
rem here to move the mechanism, not to be adopted. Quality is not claimed from this run.
rem ============================================================================
set STRATA_KEEP_MAPPING=1
if not defined STRATA_TYPICAL_ACCEPT set STRATA_TYPICAL_ACCEPT=0.2
set STRATA_POOL_SPIN_US=100
set STRATA_KV_PREFETCH=1
set STRATA_PREDICT_CREDIT=1.0
set STRATA_PREDICT_BUDGET=2
set STRATA_ADAPT_PINCAP=48
set STRATA_ADAPT_MIN_AGE=2
set STRATA_ADAPT_DORMANT=64
set STRATA_EXCHANGE_ROTATE=1
set STRATA_L2_SETASIDE_MIB=60
set STRATA_BATCH_MTP=0
set STRATA_BATCH_RESCUE=0
set STRATA_SPEC_COUPLED=1
set STRATA_PLE_BATCH=1
set STRATA_ADAPT_OVERLAP=1
set STRATA_ADAPT_OVERLAP_RESIDENT=1
rem --- THE DIAGNOSTIC KNOBS ---
set STRATA_ROUTE_RESIDENT=1.0
set STRATA_ROUTE_RESIDENT_RANKS=4-9
set STRATA_ROUTE_RESIDENT_MTP=1
set STRATA_DF_PDL=0
set STRATA_SPLIT_TIMING=1
set STRATA_DECODE_TIMING=1
cd /d "%~dp0"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "%~dp0strata-iq3_s.json" --port 8081
if errorlevel 1 pause
