### F-203 — TWO CLEAN REJECTIONS BY INTERLEAVE (doorbell value-store on CUDA; spec depth 8) + THE RUN-2 STALL'S fsync EXONERATED + THE STANDING STACK'S BEST-EVER WARM READINGS (2026-10-15 ~10:30, goal round 18 execution)

**THE LEGS.**
1. **STRATA_DOORBELL_STORE=1 (the #649 store-the-ring publish, un-gated from HIP for CUDA —
   F-203 code, default off): REJECTED, DECISIVELY.**  Two arms vs FOUR interleaved controls:
   dbstore 145.6/140.0 M1 (−21%), pair −20%, M3b **63.1/63.2 vs the controls' 75.2-76.9** — the
   acceptance state drops WITH the arm, and M3b dropping means OUTPUTS changed: the stored-value
   ring semantics desync the host service order subtly on CUDA (the HIP ring was a counter the
   CUDA wait loop does not expect as absolute per-group values).  A correctness-risk signature,
   not a perf trade.  The un-gate stays in the code DORMANT (default off) — DO NOT ARM on CUDA
   without re-deriving the value semantics per wait site.
2. **--spec 8 (depth 6→8 at DT 0.85): REJECTED.**  Two arms: 140.5/140.1 M1 (−22%), M3b 59-60.6.
   The marginal depth-7/8 rows cost ~1/3 more window time and bank ~nothing — consistent with
   the acceptance-theory prediction (under exact-match + sharpened draft, deep marginal
   acceptance collapses; measured 90%@3 → the tail is dead by 6).  Depth 6 STAYS.
3. **THE INTERLEAVE PATTERN VINDICATED AGAIN** (F-191's design): the first battery pass looked
   drift-confounded (warm ctl → two cold arms → warm ctl2); the ABAB re-run showed the cold
   state TRACKS THE ARMS.  New instrument law: **M3b is a state-canary — an arm that moves M3b
   by >±5pts is changing outputs, whatever its mechanism claims.**
4. **RUN-2 STALL: fsync EXONERATED** — strace -f on the engine through 3 full probes traced
   ZERO fsync/fdatasync (the profile save writes without sync, or via mmap).  The hunt's next
   suspect per the µarch brief: **DVFS clock transitions** (headless consumer parts sit at
   180/405MHz with no auto-boost; a governor ramp inside a window = exactly a ~100 t/s
   collapsed run).  Actionable next: verify the gpu-lock service holds clocks DURING probes
   (nvidia-smi dmon -s c alongside the next battery).

**THE POSITIVE RESULT — THE STANDING STACK'S BEST-EVER WARM READINGS.**  Four interleaved
control boots of the adopted recipe (F-200 stack + DT 0.85): **M1 179.7/181.4/176.5/182.4 ·
pair 175.0/171.0/173.3/174.2 · 6-conc 215.5/210.1/211.9/212.3 · M3b 75-77** — means
**179.9 / 173.4 / 212.5**.  vs stock's same-night window (147.1/149.1/187.7): **+22.3% M1 /
+16.3% pair / +13.2% conc** at matched windows — and the standing recipe now carries four
same-day adoptions (TYPICAL_DEVICE, VERIFY_ASYNC_LAUNCH, RR 0.75, DT 0.85) on top of the
F-31/batch-6/merge stack.  The +30% target is in reach of the R29/R30/R31 queue on warm windows.

Standing restored + verified.  Rollback: /tmp/strata-preF203-rollback (the deployed binary =
F-203 build, behaviorally identical to F-201's with the env off).
Sources: campaign/results/f203-*, /tmp/strace-fsync.log (empty — the exoneration itself).
