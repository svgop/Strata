### F-202 — THE NINE-BRIEF DEEP CORPUS COMPLETE (~90 findings): the slowness map is now measured, formulated, and sourced end-to-end; ROADMAP v3 (2026-10-15 ~08:00, goal round 18)

**THE EXECUTION TONIGHT (F-201): R25 ADOPTED — STRATA_SPEC_DRAFT_TEMP=0.85 (+13.4/+9.8/+7.5%
M1/pair/conc, sweep-peaked, proposals-only so no MMLU class).** R26 quantified, R27 assessed
(retune already banked). Standing = F-200 stack + DT 0.85.

**THE NINE BRIEFS** (investigative-researcher agents, 2-concurrent waves, full texts in
campaign/research/F-201-*-agent.md):
1. CUDA µarch — **5090 L2 is 96MB not 128** (carve = 62.5%, headroom to 72); node spacing is
   CONDITIONAL (1.5-6µs even on datacenter; near-zero for long kernels) — our 9.5µs is not a
   floor; per-graph-node L2 accessPolicyWindow; hot/cold weights 7/11µs; TMA multicast ABSENT on
   sm_120; bf16 small-ops ≈ +10% decode; device graph launch ~1ms/step; GB202 latencies worse
   than Ada; DVFS no-boost headless state = a drift suspect (verify lock holds during probes).
2. PCIe/doorbells — device-polls of host flags = uncached 1-2µs PCIe RTTs (est 1.8-3.6ms/window)
   → INVERT to posted writes (cuStreamBatchMemOp) + device-mem polls; threadfence_system = full
   L1+L2 invalidate + PCIe drain (ONE per handoff; 13 sites in elementwise.cu); DMA batch to
   2.8-10MB megablobs (12.7GB/s = per-op overhead floor; AM5 cap 15-25GB/s mixed); THP+
   cudaHostRegister the 40GB pool; WC one-way only; combined 4-7ms window + 3-3.6ms tail.
3. Zen5 host — cross-CCD ~200ns (placement: pollers CCD0/V-cache, workers CCD1@5.7GHz, per-CCD
   queues); PAUSE fine (65cy); LOCK/MFENCE 2.5× costlier (purge from flag paths); true 512-bit
   AVX-512 (shuffle-LUT dequant, 8 accumulator chains, NO gathers); hugepages: 10.5M→20K PTEs
   (IOTLB ~330ns/miss); DRAM ~60-70GB/s real; sched_yield useless; UMWAIT impossible on AMD.
4. KV — int4 KV + K-only Hadamard = the ONLY page-compatible int4 (SAW-INT4: +10.7-41.4% TPS);
   in-graph KV-gather overlap hides ~all of 4.40ms; UNIQUE page-stat scorer cached + radix top-k;
   page 8 > 4 on both axes; delta-only re-admission = 10-30×; cross-layer index sharing 2.17×;
   indexes ROT (incremental maintenance mandatory); L2 carveout measured WEAK (skip).
5. MTP — depth economics AL/(1+ρD): marginal depth pays above 6-9% acceptance, OURS IS 90% AT
   DEPTH 3; batch slot-chain steps + ONE deferred lm_head (draft head = 49-62% of draft time);
   FLA state snapshot µs-class; SpecDec++ depth gate +7-11%; head distillation on OWN traffic
   +14pp depth-3 (443s/2×H200); Q4 direct-cast validated (NEVER GPTQ the head); 4.6k vocab below
   every tested floor (re-rank from our corpus, watch OOV).
6. Acceptance THEORY — Yin optimality: every acceptance point above Metropolis costs exactly one
   TV point (P*-compensation required); Cactus δ-KL closed form replaces τ·e^−H (entropy in the
   criterion is structurally wrong — Prop 4); ERSD exponential races: chain CONTINUES THROUGH
   LOSING SLOTS, k≥3 arrivals pay on free verify rows; GLS shared-exponential endorsement for K
   slots (gains concentrate with temperature-mismatched drafts = our 0.85×); "below temp 0.8, k
   beyond 10 rarely helps"; Sequoia: wide-shallow dominates deep-narrow with free verify rows;
   DERIVED: under exact-match, acceptance = collision Σq_t·q_d (LINEAR → corner solution) — our
   0.85× is the theory-predicted optimum, and must NOT carry to any Metropolis path.
7. Scheduling — SHAPE-KEYED graphs (key on padded row count, NEVER slot layout) = the root-cause
   fix for all 176 re-captures; token-budgeted MIXED prefill+decode windows (vLLM V1 pattern);
   admission is DISPATCH-bound (90-95% overhead; our 54.8ms waves vs 13ms copy floor = 4×);
   Pensieve layer-pipelined admission; cost-ranked retention + pre-RoPE parking; conditional
   nodes prototype.
8. NVMe — RUN-2 STALL CONFIRMED as fsync-class: durable write on Gen5 consumer = ~500µs-1ms+
   PER CALL (the every-30-rounds profile save!); fixes: sidecar → other drive (zero code), async
   single-flight fdatasync, FADV_RANDOM/MADV_RANDOM on the expert tier, keep mmap+THP, no DIO.
   Diagnostic: strace fsync around round 30.
9. Formulations (identities EXECUTION-VALIDATED) — GDN one-pass fused identity (1.8e-15;
   readout decoupled from update → halves the serial chain, register-resident state per FLA);
   log-space deferred decay (9.1e-14; removes per-step exp; fold at λ<−30); hc down-mix
   ABSORBED into the adjacent GEMV (exact-static; n=4-static is lossy 0.010, n=2 safe); radix
   select for the indexer (EXACT, k-insensitive; sign-bit ordinal map — the −0.0 branch is a
   real bug trap); exponential-race sampler replaces sorts (EXACT; composition theorem for
   min-p/top-p/top-k filters); router selection needs ZERO exponentials (proof; keep full Z);
   NEGATIVE: no single-token WY win — one read+one write IS the floor.

**ROADMAP v3 (value-ordered, integrating all 15 briefs to date):**
- R28 GDN fused one-pass + log-space decay [exact, 1.5-2× step kernel]
- R29 doorbell inversion (posted writes + device-mem polls) + fence budget [1.8-3.6ms]
- R30 attention: row-fold + SpecSA sharing + L2 prefetch + int4-KV gate [4.40→1.5-2.5ms]
- R31 shape-keyed graphs [kills the re-capture class; prerequisite for scheduling wins]
- R32 partial device plan / slot indirection [4.90ms tail]
- R33 batched slot-chain drafting + deferred lm_head + depth gate [draft 6-9%→3-5%]
- R34 ERSD first-k arrivals + chain continuation [slot acceptance ceiling]
- R35 depth 6→8 at DT 0.85 + stopping rule ≥7-9% [one-boot config test]
- R36 mixed windows + event admission + delta re-admission [admission 4×; prefill stalls]
- R37 hc absorption into adjacent GEMV [exact-static; removes the starvation GEMV]
- R38 radix-select indexer + race sampler + router monotone selection [exact micro-wins]
- R39 run-2 stall fixes (sidecar isolation + async fdatasync) [smoothness = battery means]
- R40 DMA megablobs + THP pool + CCD re-pin [PCIe slice]
- R41 bf16 small-ops + device graph launch [µarch slice]
- R42 rescue δ-KL calibration [quality-gated; replaces τ heuristics with theory]
- Tier-3 (rebuild/training): head distillation on own traffic; Kitty KV; QTIP; layered prefill
  full; megakernel endgame.

Standing healthy on the full adopted stack (F-200 + DT 0.85); git 41 commits; rollbacks secured.
Sources: campaign/research/F-201-{cuda-uarch,pcie-doorbell,zen5-host,kv,mtp,acceptance-theory,
scheduling,nvme,formulations}-agent.md.
