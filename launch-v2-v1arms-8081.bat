@echo off
rem ============================================================================
rem ARM A/B CANDIDATE — "v1's arm recipe on the v2 binary" (campaign 2026-10-12)
rem
rem WHY THIS EXISTS (the archaeology finding, source-cited):
rem   The standing v2 recipe arms STRATA_ADAPT_OVERLAP=1 + _RESIDENT=1. That arm is
rem   REFUSED by the engine unless three other arms are OFF:
rem     src/program/generate.cpp:8422  "not with STRATA_DMA_BATCH"
rem     src/program/generate.cpp:8423  "not with STRATA_ADAPT_LAG / STRATA_ADAPT_NOWAIT"
rem     src/program/generate.cpp:8435  "the device-planned layers (STRATA_VERIFY_DEVICE_PLAN) are on"
rem   MAGMA v1 (launch-custom-8081.bat) runs all three of those ON:
rem     STRATA_ADAPT_LAG=2, STRATA_DMA_BATCH=1, STRATA_VERIFY_DEVICE_PLAN=1
rem   So the F-67 adoption did not "retire vestigial arms" — it traded v1's three arms
rem   for one v2 arm. The launcher comments at launch-v2-battery-8081.bat:23-24 call
rem   ADAPT_LAG and DMA_BATCH "vestigial"; the code says they are the price of overlap.
rem   Upstream ALSO measured this overlap arm NEGATIVE on Windows:
rem     src/program/generate.cpp:1425-1426 — "Windows CUDA: opt-in (RTX 5070, Q2_0:
rem     -2.6% story decode with it, IQ3_XXS neutral)".
rem   This launcher is therefore the exact A/B: v2's binary and config, v1's arms,
rem   no overlap, no ROUTE_RESIDENT (which v1 does not have either).
rem
rem THE A/B: this launcher vs launch-v2-battery-8081.bat, same window, same probe
rem (campaign\probe-lane.ps1). The Proof Law requires M1 and M2 at or above baseline
rem before this becomes the adopted recipe. Measure, then decide.
rem ============================================================================
rem F-64 REBASE LAW: v0.1.43's Windows default reads the file tier unbuffered, bypassing
rem the warm page cache; STRATA_KEEP_MAPPING=1 restores parity. v1 has no such code.
set STRATA_KEEP_MAPPING=1
rem --- the shared standing arms (identical in v1 and v2) ---
if not defined STRATA_TYPICAL_ACCEPT set STRATA_TYPICAL_ACCEPT=0.2
set STRATA_POOL_SPIN_US=100
set STRATA_KV_PREFETCH=1
rem --- v1's three arms that STRATA_ADAPT_OVERLAP=1 forces off ---
set STRATA_ADAPT_LAG=2
set STRATA_DMA_BATCH=1
set STRATA_VERIFY_DEVICE_PLAN=1
rem --- the rest of the standing set, identical to v1 ---
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
rem --- explicitly OFF: the four v2-era adoptions under test ---
set STRATA_ADAPT_OVERLAP=0
set STRATA_ADAPT_OVERLAP_RESIDENT=0
set STRATA_ROUTE_RESIDENT=0
set STRATA_ROUTE_RESIDENT_MTP=0
set STRATA_DF_PDL=0
cd /d "%~dp0"
".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "%~dp0strata-iq3_s.json" --port 8081
if errorlevel 1 pause
