#!/bin/bash
# savage-custom v2 launcher, Linux port (Qwen3.8 IQ3_S + r17 modded draft, port 8081).
# LAUNCH LAW preserved: starts serve/server.py directly (setup.py rewrites the
# "args" key on every setup run and would drop this recipe).
# --- standing arms (each measured; see SVG-CUSTOM F-05/F-12/F-15b/F-16/F-64) ---
export STRATA_KEEP_MAPPING=1        # F-64: keep mmap. No-op-likely on Linux (the unbuffered
                                    # file tier that forced this is a Windows path), harmless.
export STRATA_TYPICAL_ACCEPT=0.2    # Medusa typical acceptance (sampled traffic only)
export STRATA_POOL_SPIN_US=100      # pool workers park after 100us instead of 20ms
export STRATA_KV_PREFETCH=1         # +5.3%/4.1% prefill at 32K/64K
export STRATA_ADAPT_LAG=2           # kills the lag-1 event sync stall in the adapt tier
export STRATA_DMA_BATCH=1           # 84.9% fewer DMA submit cycles
export STRATA_PREDICT_CREDIT=1.0    # prediction credit (proven at 103k: 128 t/s / 97.6%)
export STRATA_PREDICT_BUDGET=2
export STRATA_ADAPT_PINCAP=48       # B6 victim protection
export STRATA_ADAPT_MIN_AGE=2
export STRATA_ADAPT_DORMANT=64      # never-routed experts evictable after 64 rounds
export STRATA_EXCHANGE_ROTATE=1     # +8.1% upstream anchor
export STRATA_L2_SETASIDE_MIB=60    # B8 L2 set-aside over the GDN state
export STRATA_BATCH_MTP=0           # suspended under the No-Regression Law
export STRATA_BATCH_RESCUE=0        # off (quality law)
export STRATA_SPEC_COUPLED=1        # coupled drafts on the serial path
export STRATA_PLE_BATCH=1           # batched PLE key/value projections
export STRATA_VERIFY_DEVICE_PLAN=1  # device-side routing plan when all experts resident
export STRATA_ADAPT_OVERLAP=1
export STRATA_ADAPT_OVERLAP_RESIDENT=1
# F-72 battery arms (missing from the first port; measured acceptance points when experts swap)
export STRATA_ROUTE_RESIDENT=0.5
export STRATA_ROUTE_RESIDENT_MTP=1

cd "$(dirname "$0")"
exec .venv/bin/python serve/server.py --engine strata --config "$PWD/custom-iq3_s-8081.json" --port 8081 --host 0.0.0.0
