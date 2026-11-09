---
type: digest
id: verify-arms-inventory-2026-11-09
created: 2026-11-09
status: evidence
topic: Every env-gated arm in the verifier/window path (verify.cpp, generate.cpp, mtp.cpp, verify.hpp) priced for the GSQ serial lane's 12.27 ms GPU-reach wait: what each arm changes at SERVE time, its documented default rationale, whether it is already armed in the lane launcher, the DEVICE_PLAN/doorbell interaction, and the ordered boot-test plan with the kill-line
scope: read-only study of /home/svg/LLM/strata-custom-2 (src/core/verify.cpp 4,721 ln, src/program/generate.cpp 18,533 ln, src/core/mtp.cpp 2,169 ln, include/strata/core/verify.hpp 681 ln) plus launch-custom-iq3s-8081.sh, config/custom-iq3_s-8081.json, campaign/battery-lib.sh + batteries/gsq-final.sh + results/sweep-diag-pcie90.txt. No builds, no boots run. Siblings: 2026-11-06-ud-slowdown-kernels.md (the kernel taxonomy this table hangs off), w2-ungated-draft-chain.md (DRAFT_UNGATED's study).
---

# Verify/window env arms, inventoried for boot pricing (2026-11-09)

Verdict: **the big window-path arms are already armed on the GSQ lane** (`launch-custom-iq3s-8081.sh` exports `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_DM=1`, `STRATA_VERIFY_ASYNC_LAUNCH=1`, `STRATA_FUSE_HITS=1`, `STRATA_PLE_BATCH=1`, `STRATA_QSA_SPLIT_ATTN=1`, `STRATA_ROUTE_RESIDENT=0.75`, `STRATA_SPEC_PROB=1`, `STRATA_ADAPTIVE_SPEC=1`). The remaining OFF arms that can still cut the per-layer doorbell chain are, in value order: **PLAN_PARTIAL** (extends the armed DEVICE_PLAN to miss groups), **DOORBELL_STORE** (drops one mapped RMW per served layer-group), **FUSE_WAIT=2** (one node fewer per miss group), **DRAFT_UNGATED** (draft-chain host bubbles), **GDN_SLOTS** (commit promote). The zero-doorbell family (WINDOW_GRAPH, LFUSE) is **inert on this lane** — every one of its gates reads `ar_on()`, and the released box runs the doorbell graph. No env can *force* all-resident: the gate is earned by the complement (verify.cpp:1113-1126 reads `hits.h_res`), which is the honest boundary of this whole inventory.

## 1. The lane, instrumented

From the fork's own capture, `campaign/results/sweep-diag-pcie90.txt:10` (the decode-timing line the arms must move):

```
extra5  T 3.26 | tok/win 3.03 | 21.27 ms/win = GPU wait 12.27 + host 5.10 (CPU 4.21) | => 142,5 t/s
```

- The 12.27 ms "GPU wait" is the host's spin on the per-layer doorbell (`while (*seq < want)`, verify.cpp:3149 serial `run()`, verify.cpp:4550/4581 pipelined `service()`) — i.e. the GPU's whole wall to reach each layer, which itself is mostly the per-layer wait chain `wait_flag_ge(m_flagA_)` → grouped hits → `wait_flag_ge(m_flagB_)` → PCIe grouped → `wait_flag_ge(m_flag_)` → rows copy → combine (verify.cpp:2240-2331).
- Host 5.10 (CPU 4.21) is the pool's per-window share (miss rows + planning). CPU expert misses scale ~1.75x/2.4x/3.05x per added row (verify.hpp:10-11), so *row-count* arms (SPEC_KEEP, ENTABORT, DRAFT_UNGATED, ADAPTIVE_SPEC — the last already armed) are host-work arms too.
- Config pins `--pcie-frac 0.35 --pcie-mode dma` (config/custom-iq3_s-8081.json:28-31), so flagB/PCIe staging is live, and `--pcie-frac 0` is the bit-identical A/B control for any partial-plan arm (verify.cpp:2098-2101 says exactly this).
- Serial single GPU ⇒ `always_publish_` is false on this lane (it is set only for `--pipeline-windows` and piped slot groups, generate.cpp:9195, 10511-10512), so DEVICE_PLAN is live and the AR gates decide the graph variant per window (`refresh_ar`, verify.cpp:2525-2539).

## 2. Task 1 — the exhaustive arm table (verifier/window path)

"Code" = built-in default; "L" = the GSQ launcher (`launch-custom-iq3s-8081.sh`) arms it (line in parens). SERVE column = what changes per window once graphs are captured (all these are capture-time constants; nothing re-decides mid-window except refresh_ar/adapt).

### 2a. Plan / doorbell / expert-wait chain (the GPU-reach path)

| env | code | L | what it changes at SERVE | documented rationale for the default |
|---|---|---|---|---|
| `STRATA_VERIFY_DEVICE_PLAN` | off | **on** (24) | E-6: for a layer-group whose routed experts are ALL resident (`d_res` check on device), `resident_plan` builds the plan on device into `plan_` and sets `skip_[grp]=ring`; the GPU skips `wait_flag_ge_planned(m_flagA_)`, skips the plan copy (`copy_i32_from_mapped_unless`), skips waitB, zero-fills the CPU-rows copy. Host still gets rung and still calls the pool (planning-only) — see §4. | "exact, but neutral on RIBPC 1-2 GPUs: off by default" (verify.cpp:1137-1138). Refused with set_always_publish (the device table may lag; verify.hpp:290-298). |
| `STRATA_PLAN_PARTIAL` | off | off | T1.1: with DEVICE_PLAN, a group WITH misses is planned over its resident entries on device (`skip_=kPlanPartialSkip`); resident grouped GEMM starts without the host plan round trip; only `flag_` (CPU rows) is waited. v1: pool routes every miss to CPU — no PCIe share in partial groups. | "the resident work no longer sits behind the host plan round-trip... --pcie-frac 0 keeps arms bit-identical for the A/B" (verify.cpp:2098-2101). |
| `STRATA_VERIFY_ALL_RESIDENT` | on (checks) | on (implicit) | =0 disables the AR residency check → a fully-resident stage would still run the doorbell graph. On this lane the complement is partial, so both settings take the doorbell graph; this is a control knob, not a lever. | AR is earned, never forced: 100% of [lb_,le_) resident in VRAM → "zero-doorbell graph" (verify.cpp:1105-1126, hpp:498-504). |
| `STRATA_DOORBELL_STORE` | off | off | F-203: `doorbell_publish_value` STORES the capture-known ring constant instead of the RMW `doorbell_publish(_res)` — drops the GPU's mapped-memory read per served layer-group (one PCIe RTT each). | "#649 A/B: the step's ring stored, not incremented over PCIe... the store drops the GPU's read of host memory (one PCIe RTT per served layer-group) that doorbell_publish's RMW pays" (verify.cpp:2109-2114, 228). |
| `STRATA_VERIFY_COHERENT` | off | n/a | HIP only (`#if STRATA_USE_HIP`, verify.cpp:225-227): handshake words in fine-grained coherent host memory. | #649 A/B for the gfx1030 verify timeouts. CUDA never reads it. |
| `STRATA_DM` | off | **on** (34) | dm_wait_combine: the CPU-rows wait + pool rows' read + combine as ONE launch instead of wait→copy→combine (two nodes fewer per group, no parts round trip). Gate requires `device_plan_`; inert without it ("STRATA DM requested but the device-plan route is off" boot line, verify.cpp:1165-1169). | T32 phase-1b basalt port; "the OFF arm's graph is exactly today's"; F-277 ADOPTED with pooled 8-boot M1 +2.5%. |
| `STRATA_DM_WAIT_STATS` | off | off | T26-1 instrument: stats variants of the wait kernels write flagA/flagB/CPU-rows wait time + per-layer slots to device words; a summary line every 512 windows. "changes no value the window computes" — but ADDS counter adds to every wait. | T26-1/T27-1 measurement of "where the MoE tail's per-group wait time goes on the host-plan path" (verify.cpp:415-422). The pricing instrument, not a cut. |
| `STRATA_FUSE_WAIT` | HIP: gfx12 only; CUDA: Turing only | off (this card) | CUDA: =1 folds plan-wait+plan-copy AND CPU-rows-wait+rows-copy; =2 only the plan pair; each is one kernel node fewer per group. Applies to the miss groups' `else` branch (verify.cpp:2248-2263) — i.e. exactly the groups DEVICE_PLAN skips planning for. | "A dependent kernel costs ~2.4 us in a gfx1201 graph whatever it does"; rows fold was -1.5% on RTX 2080 Ti, so it stays opt-in (verify.cpp:311-314, 350-357). |
| `STRATA_FUSE_HITS` | CUDA: off | **on** (15) | `moe_hit_add` folded into the combine (one launch fewer per group). | T12/F-226 ADOPTED: M1 +4.35%, ZADD bitwise (native_moe.cu:46). |
| `STRATA_FUSE_CONVAB` | off | n/a | HIP only: GDN conv + a/b projections in one launch. | "measured -0.3% on gfx1201" (verify.cpp:370). |
| `STRATA_PCIE_*` / `STRATA_SPLIT_*` / `STRATA_PIPED_CONCURRENT*` / `STRATA_COPY_HIP_PRIO` / `STRATA_PEER_HOT*` / `STRATA_LAYER_SPLIT_*` / `STRATA_MTP_STAGE0` | various | off | Multi-GPU split/PCIe-lane plumbing. Out of scope for the single-GSU serial lane (config has no `--pipeline-windows`, no split). | see docs/MULTI_GPU.md; the doorbell graph's host segment differs there (service() 4542-4633). |

### 2b. Per-layer GPU node-count / kernel-form folds (capture-time, bit-exact class)

| env | code | L | SERVE change | rationale |
|---|---|---|---|---|
| `STRATA_DEC_BATCH` | on | on (implicit) | router/top-k/gr-write/combine as multi-row launches; =0 reverts per-token. | standing shape since #783; =0 is the rollback leg. |
| `STRATA_HEAD_MIX_MULTI` | CUDA on / HIP off | on (implicit) | final mixer HC read for the window in one launch. | bitwise the same sums (verify.cpp:293-309). |
| `STRATA_ONE_TOKEN_COMMIT` | CUDA on / HIP off | on (implicit) | a T=1 window advances state inside its graph; no commit graph after it. Gated off by `always_publish_` and `g_qfuse()`. | eddoursul F7 (verify.cpp:290-294). Moot at T≥2. |
| `STRATA_QIL` | Turing only | off (this card) | fused quantize+interleave (`native_quantize_q8_1_il`) for n=2..4 projections: two launches fewer per GDN and per QSA layer. Fires per group at this lane's T≈2-3.9. | "Turing default (+0.5..+1% decode)"; sm_120 never tried it (verify.cpp:1485-1495). |
| `STRATA_FUSE_HEAD_GR` | off | off | the last layer's gr_write folded into the head's fused read (`apply=true`): one launch fewer per window, no intermediate write. | opt-in; needs head loaded + no cvec cover (verify.cpp:2184-2190). |
| `STRATA_DF_PLE` | off | off | the layer-1 PLE block's post-projection steps in one pass (`native_ple_postops_batch_snap`), writing the commit's history snapshots. Needs `ple_batch_kv` (PLE_BATCH armed makes it available). | "bitwise the per-token loop below" (verify.cpp:1384-1392). One layer per window. |
| `STRATA_DF_PDL` | code defaults 1 when unset | **pinned 0** (35) | programmatic dependent launch edges: quantizations/dense projections preload weights while the predecessor drains (~13 of ~1,860 nodes). | F-279 PIN: T38 measured NEUTRAL; =1 is the documented re-price arm. |
| `STRATA_DF_BRANCH` | off | off | mixer work that reads only the layer input on side graph branches. | "Off by default: on Linux (open modules, GSP) a stage's window with the branches stopped inside its graph when an NVML query came between requests (#905)" (verify.cpp:1092-1094). Do not arm on a serve box. |
| `STRATA_SIDE_MIXER` | 2 on sm_120 CUDA, else 0 | off (card resolved 0) | GDN gates / QSA indexer (mode 1) + gate projection / queries (mode 2) on side graph branches. | dispatch capture: GSQ ran in-line gates; UD's side_gates=1 called "a latency form for SMALL T" (2026-11-06 digest §2). |
| `STRATA_HC_PERSIST` | off | off | persistent-kernel HC read form (`fused_gr_read_persist`; =3 variant). | parity harness exists (hc_persist_parity.cpp); never priced on sm_120 in-tree. |
| `STRATA_DENSE_TC` | off | off | int8-tensor-core dense projections, bit-masked, n≥4 only. Output-changing (fp32 scale-sum order) — acceptance-canary + MMLU law, not parity. | "at T=1 dense_tc is 0.72-0.9x — a window mix concentrated at 2.47 accepted/window must not pay that" (verify.cpp:1489-1495, dense_tc.cu:6). Mostly UNDER the n≥4 gate on this lane. |
| `STRATA_SH_STREAM` | CUDA on / HIP off | on (implicit) | shared expert forked to its own stream per layer; STRATA_SH_FORK_LATE (default on) forks it after the doorbell publish when not AR. | CUDA where it was added; HIP fork costs more than the shared expert (verify.cpp:270-288). |
| `STRATA_VERIFY_QDEDUP` | off | off | experts' q8_1 image of xm made once, shared expert reads it — only when the shared fork is OFF; needs `STRATA_SH_STREAM=0` on CUDA. | verify.cpp:2145-2149. |
| `STRATA_QFUSE` / `STRATA_QFUSE_HC` / `STRATA_LFUSE` / `STRATA_LFUSE_GATE` / `STRATA_LFUSE_PAIR` / `STRATA_MMVQ_IL` | qfuse off; qfuse_hc gfx12-HIP; lfuse off; mmvq_il on | off / n/a / off / on | QFUSE: HC read writes the consumer's q8_1 image (kills a quantize per layer) — but DISARMS the T=1 self-commit (verify.cpp:1211, 3533, #2758). LFUSE: shared-gate-in-router fusion, **gated on `ar_on()`** — inert on the doorbell lane. MMVQ_IL: interleaved head mmvq, on. | S26 family; LFUSE is an AR-only form (verify.cpp:1230-1233). |

### 2c. Commit path (between window and next draft round)

| env | code | L | SERVE change | rationale |
|---|---|---|---|---|
| `STRATA_COMMIT_BATCH` | on | on (implicit) | all GDN layers' conv+state in two launches, QSA indexer appends in one (~70 nodes fewer). The commit runs between window and draft round. | verify.cpp:2766-2768. |
| `STRATA_COMMIT_SYNC` | off (async single-GPU) | off | =1 restores the sync after the commit graph; default `set_commit_async(!multi_gpu)` (generate.cpp:4333) — commit overlaps the draft. | hpp:271-275. Rollback knob only. |
| `STRATA_GDN_SLOTS` | off | off | F-216: =1 serial commit PROMOTES snapshot[m-1] instead of replaying the GDN recurrence (per-layer replay deleted); =2 first audits promote-vs-replay (sync every 64th commit — diagnostic only). | verify.cpp:820-847, hpp:381-390. Exact per the F-216 design; audit boot is the parity proof. |
| `STRATA_NAN_GUARD` | on | on (implicit) | =0 removes the `logits_nonfinite_rows` kernel + mapped read per window (beside the pick; no extra sync). | #879. Micro; removes an observability net — not a recommended perf arm. |

### 2d. Draft chain (host work before every window) — mtp.cpp

| env | code | L | SERVE change | rationale |
|---|---|---|---|---|
| `STRATA_DRAFT_UNGATED` | off | off | T13A #3: enqueue every step graph back-to-back; the readout still trims at first sub-min_p prob. Deletes the per-step host launch bubble (GPU idled while host spun on the previous step's output); pays ~50 us per executed-but-untrimmed step. | "drafts and dprob are bit-identical; what goes is the per-step host launch bubble" (mtp.cpp:1790-1800). Studied in w2-ungated-draft-chain.md. |
| `STRATA_MTP_PRIORITY` | CUDA on | on (implicit) | draft stream at highest priority (with --pipeline-windows). | mtp.cpp:489-497. |
| `STRATA_MTP_SHARED_BRANCH` | CUDA on | on (implicit) | drafter's shared expert on the side stream. | mtp.cpp:1155-1170. |
| `STRATA_MTP_CATCHUP_ALL` | off | off | drafter K/V caught up for the whole verified window incl. rejected rows (more draft-side work, acceptance-shaping). | #783 PR-i. Not a host-work cut. |
| `STRATA_MTP_FULL_HEAD` / `STRATA_MTP_TOP2` / `STRATA_MTP_PROFILE` / `STRATA_DRAFT_TIMING` / `STRATA_MTP_PREFILL_SYNC` | off/on-HIP | off | diagnostics/instruments (full-vocab draft, runner-up drafting, draft timing lines). | mtp.cpp:562-573, 842, 1626, 1474. |

### 2e. Acceptance / keep controller arms (row-count = host-pool-work cuts) — generate.cpp

| env | code | L | SERVE change | rationale |
|---|---|---|---|---|
| `STRATA_ADAPTIVE_SPEC` | off | **on** (31) | W1 cost-model controller sizes the serial window: argmax_T E[tokens]/step_ms(T) from per-depth acceptance EMAs. Cut-only. | W1 ADOPTED 2026-11-02; M15 fit 8.4 + 1.57·rows ms. |
| `STRATA_SPEC_KEEP=<th>` | off | off | offers-only keep rule: the largest K whose prefix product of calibrated acceptance stays ≥ th; the rest of the window's draft rows are withheld before verification. Exact-match verify unchanged — "zero quality surface". | generate.cpp:2396-2440 (5x6 calibrated table). Directly cuts pool rows (misses scale per row). |
| `STRATA_SPEC_LOGISTIC` | off | off | depth tables become logistic priors σ(ω·logit q_max + β), refit by damped IRLS. Offers-shaping. | decode-control2 §3.1/§3.4; "ALL OFF by default (the No-Regression Law)" (generate.cpp:15659-15672). |
| `STRATA_SPEC_ENTABORT` | off | off | serial draft chain stops at the decay knee (survival floor / entropy gate); the estimator's params ride the drafter. Cuts draft-step executions AND rows. | decode-control2 §3.3 rule 3; feeds off LOGISTIC's estimator. |
| `STRATA_SPEC_PROB` | off | **on** (36) | drafts judged by rejection sampling against the drafter's q. | F-282 ADOPTED: M1 +5.8%. |
| `STRATA_SPEC_RATIO` | off | off | exact ratio rescue test replaces the typicality heuristic (sampled traffic, first rejections). | F-27/Y-1 (generate.cpp:16656-16665). |
| `STRATA_SPEC_SURVTOPK` / `STRATA_SPEC_COUPLED` / `STRATA_SPEC_DRAFT_TEMP` / `STRATA_SPEC_TRACE` / `STRATA_SPEC_DEPTH` / `STRATA_SPEC_KEEP_STATS` / `STRATA_SPEC_MIN_TEMP` / `STRATA_SPEC_PROB_DT` / `STRATA_SPEC_PROB_GATE` / `STRATA_SPEC_GUMBEL` | off / off / off / off / off / off / off / off / off / off | **on** (COUPLED 107, DRAFT_TEMP 0.85) | SURVTOPK = batch lane only; TRACE/DEPTH/KEEP_STATS = telemetry; the rest shape the drafter's distribution. | controller.hpp:55-73; spec_prob.hpp:47-69. |
| `STRATA_TYPICAL_ACCEPT` / `STRATA_TYPICAL_DEVICE` / `STRATA_BATCH_RESCUE` | off / off / off | **0.2 / on / 0** | typical-acceptance rescue at first rejection (sampled only); DEVICE = the probe runs on device (no D2H until needed). | F-200 ADOPTED; RESCUE off by the quality law (generate.cpp:358-378, launcher 106). |

### 2f. Host placement / serve-loop arms that touch the per-window host segment

| env | code | L | SERVE change | rationale |
|---|---|---|---|---|
| `STRATA_HOST_CORE` | unset | off | `last`/`sibling`: pins the serve thread before the pool pins its workers (generate.cpp:2999-3006). Shapes the doorbell service latency directly. | --host-core parity; never priced on this box. |
| `STRATA_AUX_CPUS` | unset | off | restricts the CPUs the process may use; the pool's plan follows it. | generate.cpp:3008-3012. |
| `STRATA_POOL_SPIN_US` / `STRATA_POOL_DIET` | 20ms / off | **100 / on** (20, 16) | pool workers park after the spin budget; DIET skips trailing repark waits (epoch claim fences stragglers). | F-227 ADOPTED. |
| `STRATA_VERIFY_ASYNC_LAUNCH` | off | **on** (110) | HostGraphLaunch: the window graph launches on a helper thread — "a blocking instrumented launch cannot also serve the graph's CPU doorbells"; attacks exactly the GPU-reach wait. | F-200 ADOPTED: +4.9/+7.9/+8.9. |
| `STRATA_VERIFY_PROFILE` / `STRATA_VERIFY_TRACE` / `STRATA_VERIFY_NODES` / `STRATA_VERIFY_DEBUG` / `STRATA_COMMIT_TIMING` / `STRATA_DUMMY_NODES` / `STRATA_DBG_NAN` / `STRATA_LOGPOS_TOPK` / `STRATA_TEST_VERIFY_STALL` / `STRATA_TEST_HANDOFF_DROP` / `STRATA_NAN_INJECT` / `STRATA_VERIFY_COMPLEMENT` | off | off | instruments and test hooks; every one ADDS work at serve or boot (DUMMY_NODES exists to price one node's cost). | verify.cpp:96, 587-603, 1021, 1461, 2636, 3342, 3502, 3546, 4353-4375; generate.cpp:6664 (boot-time complement checksum only). |
| `STRATA_WINDOW_GRAPH` | off | off (INERT here) | CUDA 12.4+: the zero-doorbell window captured as ONE conditional WHILE node. `wrap = g_window_graph && ar_on()` — the doorbell lane never wraps. | verify.cpp:244-259, 2554. On an AR box: same one-launch-per-window shape, in-graph control flow exercised; no host-work delta claimed. |
| `STRATA_BATCH_G2` / `STRATA_BATCH_SELF_COMMIT` / `STRATA_QSA_SLOTS` / `STRATA_BATCH_GDN_GROUPS` / `STRATA_BATCH_ATTN_ROWS` | off | off | batch-window forms (multi-slot lanes), not the serial window. | verify.cpp:236-242, 1204-1209, 1613, 1916. |
| `STRATA_ADMIT_MERGE_FIRST` / `STRATA_CKPT_*` / `STRATA_SLOT_*` / `STRATA_ELASTIC_*` / `STRATA_PIPE_BYIELD_NODRAIN` / `STRATA_SSD_*` / `STRATA_IO_*` / `STRATA_LOOKAHEAD*` | off / various | some on | admission, checkpoint, IO-prefetch lanes. Not the per-window verify path (they cut window COUNT or boot/restore cost, not the doorbell chain). | generate.cpp:183-346, 4142-4170, 5482-5506, 5876-6010, 6945-6990. |

## 3. Task 2 — ranking for the small-T serial lane (arms OFF today that can cut host/doorbell/planning work)

Ranked by expected value against the 12.27 ms GPU-reach wait + 5.10 ms host segment. Boot lines are the FULL delta vs the standing launcher (everything else stays as `launch-custom-iq3s-8081.sh` sets it).

| # | arm | boot-test env line | mechanism against the 12.27 | risk |
|---|---|---|---|---|
| 1 | PLAN_PARTIAL | `export STRATA_PLAN_PARTIAL=1` (+ `--pcie-frac 0` on BOTH legs for the bit-identical A/B; second boot re-adds 0.35) | the only remaining way to shorten the per-layer wait chain: miss groups (~5% at 99.5% slot hits) start their resident grouped GEMM without the flagA round trip, overlapping the pool's CPU miss compute | v1 removes the PCIe share in partial groups — placement change at frac 0.35; first A/B must pin frac 0 |
| 2 | DOORBELL_STORE | `export STRATA_DOORBELL_STORE=1` | store-vs-RMW publish: one mapped read (PCIe RTT) fewer per served layer-group ≈ 48/window; ring lands sooner → host pool starts sooner | none identified; ring values identical on CUDA (capture-known constant) |
| 3 | FUSE_WAIT=2 | `export STRATA_FUSE_WAIT=2` | one kernel node fewer per miss group (plan-wait+plan-copy fused; ~2.4 us/node class); =2 avoids the rows fold that was -1.5% on Turing | bit-exact; interactions with armed DEVICE_PLAN/DM need the boot (DM owns the fold on planned groups; FUSE_WAIT owns the miss groups' else-branch) |
| 4 | DRAFT_UNGATED | `export STRATA_DRAFT_UNGATED=1` | deletes the per-step host launch bubble in the draft chain that precedes every window (the host segment's draft half) | executes steps past the trim (~50 us each); net depends on the trim depth at min_p 0.5 |
| 5 | GDN_SLOTS (audit, then promote) | boot A `export STRATA_GDN_SLOTS=2`; boot B `export STRATA_GDN_SLOTS=1` | commit graph after every T≥2 window promotes snapshot[m-1] instead of replaying the recurrence; frees the stream the async commit runs on before the next window | =2 syncs every 64th commit (audit boot only, never a perf leg); =1 changes no bits per F-216 design, audit is the proof |
| 6 | SPEC_KEEP | `export STRATA_SPEC_KEEP=0.5` | withholds low-joint-probability draft rows before verification: fewer pool rows on exactly the windows that read 2.4-3.05x misses | offers-only; TH sweep later; interacts with ADAPTIVE_SPEC's own trim (they compose: cost model sizes T, KEEP trims offers) |
| 7 | LOGISTIC+ENTABORT | `export STRATA_SPEC_LOGISTIC=1 STRATA_SPEC_ENTABORT=1` | stops the draft chain at the acceptance decay knee: fewer draft-step executions and fewer rows per window | decode-control2 arms, all off by the No-Regression Law; acceptance-rate watch required |
| 8 | DF_PDL re-price | `export STRATA_DF_PDL=1` | programmatic edges on ~13 nodes; T38 said neutral, the pin exists to force a deliberate re-price | none (values identical) |
| 9 | HC_PERSIST | `export STRATA_HC_PERSIST=1` | persistent-kernel HC read form; the HC read is per-layer fixed work in the GPU-reach path | never priced on sm_120; parity harness exists — run hc_persist_parity before the boot |
| 10 | QIL | `export STRATA_QIL=1` | fused quantize+interleave at n=2..4: two launches fewer per GDN/QSA layer per group — this lane's T lives in that gate | needs the projection types in the mmvq_il whitelist (GSQ's are); bit-exact bytes |
| 11 | FUSE_HEAD_GR | `export STRATA_FUSE_HEAD_GR=1` | one launch fewer per window (last layer's write folded into the head read) | head-loaded paths only; bit-exact sums |
| 12 | DF_PLE | `export STRATA_DF_PLE=1` | layer-1 PLE post-projections in one pass | one layer per window; needs PLE_BATCH (armed) |
| 13 | HOST_CORE | `export STRATA_HOST_CORE=last` (alt: `sibling`) | serve-thread placement off the pool workers' cores; doorbell wake latency | placement-dependent; cheap to test |
| 14 | SIDE_MIXER=1 | `export STRATA_SIDE_MIXER=1` (then =2) | gates/indexer on side branches (the UD form the taxonomy called a small-T latency play) | branch events add nodes; not a host-work arm — price last |
| 15 | DENSE_TC=1 | `export STRATA_DENSE_TC=1` | int8-TC dense from n≥4 | mostly inert at T≈2-3.9; output-changing (canary law) — only worth a boot if tok/win rises first |

Explicit non-candidates on this lane: `STRATA_WINDOW_GRAPH`, `STRATA_LFUSE*` (ar_on-gated), `STRATA_VERIFY_COHERENT`/`STRATA_QFUSE_HC`/`STRATA_MTP_HIP_PRIO`/`STRATA_MTP_SIDE_STREAM` (HIP-only), `STRATA_BATCH_*` (batch lane), `STRATA_SPEC_SURVTOPK` (batch lane), `STRATA_VERIFY_ALL_RESIDENT=0` (control only — nothing forces AR on), `STRATA_NAN_GUARD=0` (micro, costs a safety net), every instrument in §2f.

## 4. Task 3 — STRATA_VERIFY_DEVICE_PLAN vs the doorbell

Read from verify.cpp:1137-1172 (init), 2093-2131 + 2240-2331 (capture), verify.hpp:290-309, 498-511.

- **What arms it:** `device_plan_ = !all_resident_ && !always_publish_ && (env != 0)`. It is the DOORBELL lane's device planning — the AR graph (all_resident_) is a different, stronger state that plans every group on device and rings nothing. On the GSQ lane (partial complement, serial) it is armed and live; the launcher comment "when all experts resident" is per GROUP, not per model.
- **What it does per group:** `resident_plan` (verify.cpp:2103-2108) counts/starts/dst/tok/ptr built ON DEVICE from `d_res` + `slot_off_d_`, and writes `skip_[grp] = (l-lb_)*G + grp + 1` — the ring whose plan the device built. `skip_` is a 64-byte device table, one word per (layer, group), 0 = "the host's plan is the one". post() then takes the planned chain (verify.cpp:2241-2247): `wait_flag_ge_planned(m_flagA_, ring, skip_+grp)` returns immediately when `skip_[grp]==ring` — no host-plan round trip; `copy_i32_from_mapped_unless` skips the plan D2H copy; waitB is planned-aware too; the CPU-rows wait becomes `wait_flag_ge_or(..., skip_+grp)` (skips only a FULL plan) and `copy_rows_or_zero_from_mapped` zero-fills the planned group's rows.
- **Doorbell interaction — the precise answer:** the ring is NOT shortened. `doorbell_publish*` still fires in pre() for EVERY group (verify.cpp:2103-2125 — the publish is outside the `if (device_plan_)` block, deliberately: the host must see progress to serve miss groups, and `m_seq_` is the only progress signal). The host segment per layer (doorbell wake → `pool()` call → flag publishes) still runs for all 48 layers; for a device-planned group the pool call is planning-only (its CPU share is empty) and its flagA publish is simply IGNORED by the GPU (the skip guard). So DEVICE_PLAN removes the GPU's *dependency* on the host's plan round trip per resident group; it does NOT remove host planning per layer and does NOT remove any wake. The host planning work that remains is the pool's own plan build per served layer.
- **Could it shorten the ring's host segment?** Only through PLAN_PARTIAL's extension: a miss group's resident work then starts under the pool's CPU compute instead of after the plan publish, so the effective serialized host contribution per miss layer drops to the CPU rows alone. That is arm #1 in §3 — and it composes with the already-armed DM fold (dm_wait_combine carries wait_flag_ge_or's guard itself, verify.cpp:2289-2295, 2343-2345).
- **Boundary:** DEVICE_PLAN is refused under `set_always_publish` (device table may lag while windows are in flight) and refuses `set_adapt_gate` (verify.hpp:299-305) — neither is active on this serial lane, so no conflict today. `STRATA_DM` without it prints the inert-request boot line; the launcher arms both, correctly ordered.

## 5. Task 4 — prioritized boot-test plan

Discipline (per tasking): ONE boot per arm, 3 warm probes after prewarm, kill-line **+3% solo or killed; no lane worse** (solo M1, pair, conc from `campaign/probe-lane.py`; the house pattern is `campaign/batteries/gsq-final.sh`: source `campaign/battery-lib.sh`, `cleanup_engine`, launch via the launcher, `wait_prewarm`, probe, kill — 5 boots / last 3 banked when an arm survives its first boot). Control leg = the standing launcher unmodified, same boot order.

| order | boot | env line (delta) | expect | keep if |
|---|---|---|---|---|
| 0 | control + pricing | none, then `STRATA_DM_WAIT_STATS=1` on a second boot | the DM_WAIT_STATS line prices waitA/waitB/waitCPU per window and per layer — this tells whether 12.27 is plan-round-trip (→ arms 1-3), CPU rows (→ arms 6-7), or GPU dense (→ nothing here helps) | never a keep; instrument only |
| 1 | partial plan | `STRATA_PLAN_PARTIAL=1`, both legs `--pcie-frac 0` | GPU wait −(5% of groups × plan RTT + overlap win); frac-0.35 re-test only if leg 1 is flat-positive | solo +3%, no lane worse |
| 2 | doorbell store | `STRATA_DOORBELL_STORE=1` | ~48 mapped RTTs/window off the publish path | solo +3% or (≥0 and waitA-dominated evidence); zero-value-change so a flat-positive keeps on mechanism |
| 3 | fused plan wait | `STRATA_FUSE_WAIT=2` | node-count delta on miss groups; micro but free | solo ≥ 0 with no lane worse (micro class; kill at any negative) |
| 4 | ungated draft | `STRATA_DRAFT_UNGATED=1` | host segment's draft half tightens; watch draft wall in DRAFT_TIMING if needed | solo +3%, no lane worse |
| 5a | GDN slots audit | `STRATA_GDN_SLOTS=2` | "0 mismatches" line after 64 commits — parity proof, NOT a perf leg | audit passes → unlock 5b |
| 5b | GDN slots promote | `STRATA_GDN_SLOTS=1` | commit graph shrinks; next window launches into a freer stream | solo +3%, no lane worse |
| 6 | spec keep | `STRATA_SPEC_KEEP=0.5` | fewer rows on low-acceptance windows → host CPU 4.21 term drops | solo +3%, no lane worse, mimic/M3b in-band (quality watch: offers-only, but measure) |
| 7 | entabort | `STRATA_SPEC_LOGISTIC=1 STRATA_SPEC_ENTABORT=1` | draft chain stops at the knee: fewer steps AND rows | solo +3%, no lane worse, acceptance-rate EMA not down |
| 8 | pdl re-price | `STRATA_DF_PDL=1` | expected ~0 (T38) | kill unless surprise positive; pin stays 0 on flat |
| 9 | hc persist | `hc_persist_parity` first, then `STRATA_HC_PERSIST=1` | HC read form change | parity first; solo +3% |
| 10 | qil | `STRATA_QIL=1` | two launches fewer per layer at n=2..4 | solo ≥ 0, no lane worse |
| 11 | fuse head gr | `STRATA_FUSE_HEAD_GR=1` | one launch/window | micro class |
| 12 | df ple | `STRATA_DF_PLE=1` | layer-1 only | micro class |
| 13 | host core | `STRATA_HOST_CORE=last` | wake latency | solo ≥ 0; try `sibling` only if `last` moves anything |
| 14 | side mixer | `STRATA_SIDE_MIXER=1`, then `=2` | small-T branch latency (the UD form) | solo +3%; expect branch-node overhead to eat it |
| 15 | dense tc | `STRATA_DENSE_TC=1` | mostly under the n≥4 gate today | park until tok/win ≥ 4 (the code's own crossover note) |

Composition note: arms 1+2+3 touch different segments of the same chain (plan RTT / publish / node count) — survivors compose into one confirmation boot, then the standing battery (5 boots, last 3 banked) for the combined recipe before any launcher edit.

## 6. Boundary

- No builds, no boots, no probes run (read-only study; the numbers quoted are the fork's own captures).
- The 12.27 ms line is from the pcie-frac-0.90 sweep leg (`sweep-diag-pcie90.txt`), not the standing frac-0.35 lane; the DM_WAIT_STATS leg (order 0) re-prices it on the standing config before any arm is judged.
- `STRATA_VERIFY_ALL_RESIDENT` has no force-on form: on the released box (partial complement, no hugepages) every window takes the doorbell graph, and the only lever that changes that is complement size — outside this inventory's env-arm frame.
- Launcher-armed arms (DEVICE_PLAN, DM, ASYNC_LAUNCH, FUSE_HITS, PLE_BATCH, QSA_SPLIT_ATTN, ROUTE_RESIDENT, SPEC_PROB, ADAPTIVE_SPEC, POOL_DIET, TYPICAL_*) are the baseline; their `=0` legs are the rollback controls, not candidates.
