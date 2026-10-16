#!/bin/bash
# savage-custom v2 launcher, Linux port (Qwen3.8 IQ3_S + r17 modded draft, port 8081).
# LAUNCH LAW preserved: starts serve/server.py directly (setup.py rewrites the
# "args" key on every setup run and would drop this recipe).
# --- standing arms (each measured; see SVG-CUSTOM F-05/F-12/F-15b/F-16/F-64) ---
# F-221 ADOPTED (2026-10-16): PR#540 chunk kernel + bounded merge on sm_120.
# ABAB 3v3: M1 +6.0 / pair +7.0 / conc +5.3; bit-identical (parity 1172/0 on this GPU).
export STRATA_ATTN_DAILY=1
export STRATA_GDN_CHUNKED=1  # F-224 ADOPTED 2026-10-16: prefill DeltaNet recurrence in WY chunks (kernels.cu; OUR gdn_rec_parity --bench: 1.68x at 32K, -115ms/chunk). Paired prefill probe 2x4: -141..168ms (~-4.2%) EVERY pair; decode lanes neutral (3v3 ABAB, all inside noise); M3b in-band; MMLU-Pro 2x gate 95.7/92.9. Different FP32 sum order (prefill only) - the gate covers it
export STRATA_FUSE_HITS=1  # F-226 ADOPTED 2026-10-17: the moe_hit_add->combine fold (was HARD-FALSE on CUDA, de05c182) - ZADD bitwise per native_moe.cu:46. 3v3 ABAB (fuarm/fuctl): M1 +4.35% (202.5 vs 194.1), pair +2.34%, conc +1.24%, zero lane regressions, M3b in-band (76.2 vs 75.6)
export STRATA_TYPICAL_ACCEPT=0.2    # Medusa typical acceptance (sampled traffic only; R5: worth +20.6pts accept / +8% M1)
export STRATA_POOL_SPIN_US=100      # pool workers park after 100us instead of 20ms
export STRATA_KV_PREFETCH=1         # +5.3%/4.1% prefill at 32K/64K
export STRATA_ADAPT_LAG=2           # kills the lag-1 event sync stall in the adapt tier
export STRATA_DMA_BATCH=1           # 84.9% fewer DMA submit cycles
export STRATA_VERIFY_DEVICE_PLAN=1  # device-side routing plan when all experts resident

export STRATA_PREDICT_CREDIT=1.0    # prediction credit (proven at 103k: 128 t/s / 97.6%)
export STRATA_PREDICT_BUDGET=2
export STRATA_ADAPT_PINCAP=48       # B6 victim protection
export STRATA_ADAPT_MIN_AGE=2
export STRATA_ADAPT_DORMANT=64      # never-routed experts evictable after 64 rounds
export STRATA_L2_SETASIDE_MIB=60    # B8 L2 set-aside over the GDN state
export STRATA_BATCH_MTP=1           # F-31 ADOPTED 2026-10-14: coupled slot drafts (60-67% accept, pair +11.5%, MMLU 97.14). Exact-match only - the F-30b rescue stays OFF.
export STRATA_BATCH_RESCUE=0        # off (quality law)
export STRATA_SPEC_COUPLED=1        # coupled drafts on the serial path
export STRATA_PLE_BATCH=1           # batched PLE key/value projections
export STRATA_TYPICAL_DEVICE=1      # F-200 ADOPTED 2026-10-15: device-side typical rescue (parity-proven port; +9.0/+6.7/+6.4 M1/pair/conc vs interpolated control)
export STRATA_VERIFY_ASYNC_LAUNCH=1 # F-200 ADOPTED 2026-10-15: v0.1.44 HostGraphLaunch (+4.9/+7.9/+8.9 - attacks the 69% GPU-reach wait)
export STRATA_SPEC_DRAFT_TEMP=0.85  # F-201 ADOPTED 2026-10-15: coupled drafter samples at 0.85x temp (proposals-only; exact-match verify emits the target's picks). Sweep 0.45-0.90 peaked here: M1 +13.4/pair +9.8/conc +7.5%

# (STRATA_ADAPT_OVERLAP / _RESIDENT removed F-187: the engine refuses them while DMA_BATCH=1 —
#  "is off: not with STRATA_DMA_BATCH" boot line; KEEP_MAPPING removed: dead branch on Linux,
#  expert_source.cpp:510-520. F-185 rejected the overlap trade anyway.)
# F-72 battery arms (missing from the first port; measured acceptance points when experts swap)
export STRATA_ROUTE_RESIDENT=0.75     # F-200 ADOPTED 2026-10-15: battery +14.9/+8.5/+12.2 M1/pair/conc vs interpolated control; routing-changing -> 2x MMLU gate PASSED (92.86/97.14, mean 95.00)
export STRATA_ROUTE_RESIDENT_MTP=1

cd "$(dirname "$0")"
exec .venv/bin/python serve/server.py --engine strata --config "$PWD/custom-iq3_s-8081.json" --port 8081 --host 0.0.0.0
