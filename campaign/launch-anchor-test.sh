#!/bin/bash
# F-197 battery arm: the current engine at the ANCHOR recipe (parallel 4 via campaign/anchor-test.json,
# BATCH_MTP=0) - the same shape F-188's boot B used. Env battery = the standing launcher's.
cd /home/svg/LLM/strata-custom-2
export STRATA_TYPICAL_ACCEPT=0.2 STRATA_POOL_SPIN_US=100 STRATA_KV_PREFETCH=1 STRATA_ADAPT_LAG=2 STRATA_DMA_BATCH=1 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_PREDICT_CREDIT=1.0 STRATA_PREDICT_BUDGET=2 STRATA_ADAPT_PINCAP=48 STRATA_ADAPT_MIN_AGE=2 STRATA_ADAPT_DORMANT=64 STRATA_EXCHANGE_ROTATE=1 STRATA_L2_SETASIDE_MIB=60 STRATA_BATCH_MTP=0 STRATA_BATCH_RESCUE=0 STRATA_SPEC_COUPLED=1 STRATA_PLE_BATCH=1 STRATA_ROUTE_RESIDENT=0.5 STRATA_ROUTE_RESIDENT_MTP=1
exec .venv/bin/python serve/server.py --engine strata --config /home/svg/LLM/strata-custom-2/campaign/anchor-test.json --port 8081 --host 0.0.0.0
