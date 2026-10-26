#!/bin/bash
# savage-custom v2 launcher, Linux port (Qwen3.8 IQ3_S + r17 modded draft, port 8081).
# LAUNCH LAW preserved: starts serve/server.py directly (setup.py rewrites the
# "args" key on every setup run and would drop this recipe).
# --- standing arms (each measured; see SVG-CUSTOM F-05/F-12/F-15b/F-16/F-64) ---
# F-221 ADOPTED (2026-10-16): PR#540 chunk kernel + bounded merge on sm_120.
# ABAB 3v3: M1 +6.0 / pair +7.0 / conc +5.3; bit-identical (parity 1172/0 on this GPU).
# T15C S3 / F-241 ADOPTED (2026-10-20): the STRATA_GR_NORM_FAST CUDA flip is now the CODE default
# (fused_gr.cu norm_fast_on(): the split norm's register kernel; bitwise per gr_parity byte-equality
# + the boot-time hc selftest every boot). POOLED 6v6 (s3norm + s3confirm): M1 +5.2% (per-battery
# +9.68% then +0.87% - the first battery's control window read low), pair +0.8%, conc +5.5%,
# M3b +2.3pp; no lane negative in any battery. ROLLBACK: export STRATA_GR_NORM_FAST=0
export STRATA_ATTN_DAILY=1
export STRATA_GDN_CHUNKED=1  # F-224 ADOPTED 2026-10-16: prefill DeltaNet recurrence in WY chunks (kernels.cu; OUR gdn_rec_parity --bench: 1.68x at 32K, -115ms/chunk). Paired prefill probe 2x4: -141..168ms (~-4.2%) EVERY pair; decode lanes neutral (3v3 ABAB, all inside noise); M3b in-band; MMLU-Pro 2x gate 95.7/92.9. Different FP32 sum order (prefill only) - the gate covers it
export STRATA_FUSE_HITS=1  # F-226 ADOPTED 2026-10-17: the moe_hit_add->combine fold (was HARD-FALSE on CUDA, de05c182) - ZADD bitwise per native_moe.cu:46. 3v3 ABAB (fuarm/fuctl): M1 +4.35% (202.5 vs 194.1), pair +2.34%, conc +1.24%, zero lane regressions, M3b in-band (76.2 vs 75.6)
export STRATA_POOL_DIET=1  # F-227 ADOPTED 2026-10-17: trailing repark waits skipped (run_phase + run; epoch-claim protocol makes them latency-shaping not correctness). 3v3 (dietarm/dietctl): M1 +4.06% (196.6 vs 188.9), pair +2.41%, conc -0.34% (noise-level round overlap), M3b in-band. =0 rolls back
# T28 share=1.0 arm REVERTED (F-263: no reproducible gain - single-window wins were noise; shelf: export STRATA_BATCH_DECODE_SHARE=1.0
export STRATA_COHORTS="${STRATA_COHORTS:-2}"  # F-233 ADOPTED 2026-10-17: two cohort verifiers on the batch lane (R2-S3) - B pool work runs under A GPU window (the F-223 escape). 3v3 fenced: conc +24.8% (306.8 vs 245.9, every arm round >304 = the 300 stretch goal CLEARED), M1 +1.2%, pair noise (byte-identical path), M3b -0.04pp flat. =1 off rolls back; env-overridable since F-284 (the hard export silently overrode the ternary arm boots - STRATA_COHORTS=3 now passes through; 3 needs batch>=9 AND the b9 lane fixed first)
export STRATA_TYPICAL_ACCEPT=0.2    # Medusa typical acceptance (sampled traffic only; R5: worth +20.6pts accept / +8% M1)
export STRATA_POOL_SPIN_US="${STRATA_POOL_SPIN_US:-100}"  # pool workers park after 100us instead of 20ms (overridable F-346: the wake-latency arm tests 2000)
export STRATA_KV_PREFETCH=1         # +5.3%/4.1% prefill at 32K/64K
export STRATA_ADAPT_LAG=2           # kills the lag-1 event sync stall in the adapt tier
export STRATA_DMA_BATCH=1           # 84.9% fewer DMA submit cycles
export STRATA_VERIFY_DEVICE_PLAN=1  # device-side routing plan when all experts resident
export STRATA_ELASTIC_SESSIONS=1
export STRATA_ELASTIC_PARK=1        # F-276 ADOPTED: demote parks via ConversationCache - returns restore 9-15ms vs 275-326ms re-reads
export STRATA_ELASTIC_WATERMARK=1   # F-276 ADOPTED: spare always holds one arena - promotes background 2.2-8ms, demote unmap 2-2.8ms
export STRATA_ELASTIC_TIMED_SCAN=1  # F-276 ADOPTED: demotes in true idle - the 128-310ms first-arrival bill removed
export STRATA_VMM_CHUNK_MIB=32      # F-276 ADOPTED: superchunks - 24 driver ops/arena vs 373 (29-51us/op GSP-measured)
export STRATA_SERVE_TELEMETRY=1     # F-276 ADOPTED: window p50/p99 + true per-token gaps + admission waits in /metrics  # F-273 ADOPTED (T33/T34): slot arenas VMM-elastic - S_min=2 mapped, promote ~12-17ms, demote feeds the cache; conc +4.0% 3/3 rounds, M1 no-regression, 9-wide with zero solo tax. OFF: unset
export STRATA_DM=1                  # F-277 ADOPTED (10-23): dm_wait_combine - the CPU-rows wait + pool rows' read + combine in ONE launch (verify.cpp:492 gate; boot line "STRATA_DM armed"). Pooled 8-boot (t32+t32b, two windows): M1 +2.5% (222.4 vs 217.1), pair +0.9, conc +1.3, M3b in-band; MMLU 95.7 x2; mimic PAIRED 3v3 dm 60/60/62 vs ctl 60/60 (the first 56-read was a tail draw - F-264 bit-identity held). OFF: unset
export STRATA_DF_PDL=0              # F-279 PIN: the T38 V2 code DEFAULTS STRATA_DF_PDL to 1 when unset (verify_kernels.cu:2084) - every boot since T38 silently ran PDL-armed (measured NEUTRAL by the T38 A-B-A-B deploy interleave, surface ~13 of ~1,860 nodes). Pinned 0 = the documented F-191/F-278 posture + pdl-accuracy.json protocol restored. Re-price arm is =1 (never =0)
export STRATA_SPEC_PROB="${STRATA_SPEC_PROB:-1}"  # F-282 ADOPTED (10-23 night): probabilistic draft acceptance (sampler.cu:891, the T17C-priced never-measured arm). T41 battery: M1 +5.8% (217.8 vs 205.8, 2/2 legs above EVERY ctl leg), conc +4.4%, M3b +2.0pp (mechanism-correlated); MMLU 4-run set 94.3/92.9/95.7/94.3 mean 94.3 (one sub-floor read, tiebreaker-clean); mimic 56-read = the night's 3-config bench low-mode (paired tests showed 60-62). OFF: STRATA_SPEC_PROB=0. Rider: five-boot pooled confirmation + mimic watch on the next battery

export STRATA_ADAPT_EARLY="${STRATA_ADAPT_EARLY:-1}"  # F-332 ADOPTED (10-25): the adaptive tier's decision scan (adapt_scan, generate.cpp) runs on a
                                  # thread UNDER the window run (snapshot of usage/pred/heat; decisions one round staler - EWMA noise); gate-set
                                  # + copy enqueues unchanged at round end (F-187 overlap lane stays closed). 3v3 same-hour: pair +5.4% (209.6 vs
                                  # 198.8), M1 +0.1 flat, conc +1.6 in-band, M3b +1.6pp; MECHANISM PROVEN: "adapt wait" 1.03 -> 0.28-0.62 ms/window.
                                  # =0 rolls back (inline scan, byte-identical legacy path). Rider: five-boot pooled confirmation (pair leg-1
                                  # sensitivity noted in F-332); engine rollback: engine/strata.pre-f332
# F-331 ARMS (10-25, both never priced on this lane; see SVG-CUSTOM F-331):
export STRATA_QSA_SPLIT_ATTN="${STRATA_QSA_SPLIT_ATTN:-1}"     # decode+drafter attention through the split int8-MMA kernel (verify.cpp:1973 + mtp.cpp:1088;
                                  # qsa_prompt_attn.cu:2271: each query's cells split over up to 20 blocks, LSE merge; int8-KV sm_80+;
                                  # fork Eddoursul fd23d91 idea; DETAILS vendor class +2.8/3.5% on other cards). =0 rolls back
export STRATA_PROMPT_ATTN_IMMA="${STRATA_PROMPT_ATTN_IMMA:-1}" # prefill attention, both products on INT8 tensor cores (qsa_prompt_attn.cu:1081 i8v3:
                                  # 3 int8 MMAs ~= 1 fp16 MMA where v2 spends 2; sm_120 measured 732 vs 233 rate; 24-bit split
                                  # finer than v2's fp16 hi+lo - precision IMPROVES). Prefill lane. =0 rolls back

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

cd /home/svg/LLM/strata-custom-2
exec .venv/bin/python serve/server.py --engine strata --config "$PWD/campaign/arms/f348-routing.json" --port 8081 --host 0.0.0.0
