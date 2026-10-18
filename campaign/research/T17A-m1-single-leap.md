# T17A — the M1 single-stream LEAP: a fresh hunt for levers outside the T16 queue, external frontier check (2026-10-17)

Topic: M1 single-stream decode, target 250+ on IQ3_S (stretch 270). Method: line-level re-read of
the serial chain (solo loop, drafter, verifier tail, hc/dense/head kernels, commit/graph machinery)
plus a 45-source external sweep (arXiv 2024-2026, SGLang/vLLM/TRT-LLM/FlashInfer/llama.cpp, NVIDIA
CUDA 13.x/Blackwell docs). READ-ONLY session: nothing built, booted, or modified except this file.
Baselines: five-boot M1 mean 198.5 at 2835 clocks; standing recipe now runs SM_CLOCK=3135
(realized ~2940, ~+3.4%). Every price below uses the T15A transfer law: **+20 t/s per −1.0 ms/window**
(probe-mean terms, ±10%).

---

## 1. Executive summary

1. **The T16 queue survives this audit intact** — nothing in it is contradicted, and the external
   sweep *de-risks* its centerpiece (the FastMTP retrain now has production deployments on our exact
   head class: Red Hat/vLLM Speculators 0.6.0, Sep 2026). What this dossier adds is (a) the
   **structural capstone** the queue lacks, (b) three tail levers priced off the in-tree
   instrumentation, and (c) a list of would-be candidates that DIED under code read — recorded so
   no future session re-pays for them.
2. **The one lever with 300-class magnitude is the megakernel-class restructure of the AR verify
   window** — and its evidence base changed materially in 2026: Cohere shipped a *production* decode
   megakernel (30B-A3B MoE, ONE H100, bf16, **bs=1 decode 292 vs vLLM 185 = 1.58x, 62% of
   speed-of-light**) [S32]; Mirage MPK, ForgeMegakernel (arXiv 2609.12379) and Hazy's tp-llama
   corroborate the class [S33-35]. Our own anatomy says the same thing: **W(1) = 8-9 ms for a
   one-row window** against a ~2.1 ms byte floor — i.e. **~6-7 ms/window of fixed node-dependency
   tax (~55% of a steady window)** that no per-kernel rung can touch. The T16 kernel tranche's own
   ceiling arithmetic (T15A §5) tops out ~240-270; the structural class is what lies beyond, and it
   now has production precedent on our exact regime (bs=1 MoE decode). Crucially, the *falsifier for
   its magnitude is already in the tree*: `STRATA_DUMMY_NODES` (verify.cpp:1152-1155) adds k empty
   nodes per layer — one A/B boot measures our per-node tax directly, before any surgery.
3. **The serial chain is genuinely closed** — this is now externally corroborated, not just
   F-223-internal. SGLang's parallel-speculation roadmap (#27462, 2026) concluded the only
   exactness-preserving way to run a drafter ahead of commits is *enumeration over every accept
   outcome*, and that enumeration is exact **only for token-conditioned drafters — EAGLE/MTP-class
   hidden-state drafters lose exactness or overlap** [S10]. Our drafter is hidden-state-conditioned;
   enumeration would multiply draft rows by K+1, which the c_row law (0.48-0.60 tokens/row needed vs
   0.16 measured for +0.8 tokens, T15A §3.10) kills. Do not revisit cross-window draft/verify
   overlap on this contract.
4. **New tail levers found (each small, each cheaply falsifiable):** (i) a **device-side
   conditional-node draft chain** (CUDA 12.4+ IF/WHILE graph nodes; the trim condition
   `prob >= min_p` is already device-resident in `m_prob_`, mtp.cpp:1382) removes the gated loop's
   per-step host launch bubble without F-229's over-draft waste — +1..2.5 t/s; (ii) the **host-side
   sampler** re-does what the window graph already computed and pays a second stream sync
   (verify.cpp:2509-2541) — folding it into the graph (params staged through mapped memory like
   every other per-window input) is +1..2 t/s beyond T15A's sync-elision item; (iii) the **first
   window is never drafted** (`first_window T=1`, generate.cpp:13119) although the drafter's round
   graph could run immediately after prefill — +0..1% probe mean and a visible TTFT win for XS-S
   code; (iv) the standing **60 MiB L2 set-aside is pointed at the GDN states**, whose rec line is
   already near floor (T15A §4.E) — retargeting it (or dual-arming) at the drafter's ~66 MB per-step
   working set (13.8 MB Q2_0 experts + ~52 MB Q4_0 sub-vocab head, re-read 4-6x per window) is
   +0.5..1.5 t/s for an env-class arm.
5. **Died under code read this session (banked negatives):** AR-slim graph (the AR window is ALREADY
   a zero-doorbell, device-planned graph with planned-skip waits — verify.cpp:1558-1561, :1678-1698,
   :3644); shared-expert-into-routed fusion (the tqllm T1.1 residue is already hidden: the shared
   expert forks onto a side stream by default on CUDA — verify.cpp:212-221, :1505-1507); a separate
   "dense BF16 GEMV" lever (the dense line already runs MMQ kernels — verify.cpp:1163-1173 — so the
   IL-cells item covers it; `bf16_gemv*` serves only the small PLE/indexer mats); commit-overlap
   (async commit hides under the draft phase, F-232); QFUSE-family producer-fused quant (rejected,
   standing).
6. **Bottom line:** the leap to 250 is the T16 tranche + retrain, unchanged; the leap past ~270 is
   the structural class (§3 R1), whose magnitude can be priced for one boot's cost via
   STRATA_DUMMY_NODES before anyone commits weeks. The delusional 270+ needs R1 or a better drafter
   than FastMTP (DSpark-class block drafter with confidence scheduling [S4] — a training project,
   parked).

---

## 2. What the code actually does on the serial path (verified this session)

The standing M1 round (solo loop, generate.cpp:13113-13354, one full pass per window):

```
T = 1 + #{dprob[j] >= min_p}   (previous window's draft probs; generate.cpp:13115-13118)
first window: T=1 hard         (generate.cpp:13119)          <-- never drafted (R4)
[lookup/chain policy EV check] (generate.cpp:13124-13146)
penalty_rows + SYNC d_hist H2D (generate.cpp:13189-13193)     <-- sync memcpy; device idle here anyway
ver.run(T)                     (generate.cpp:13206 -> verify.cpp:2338-2564)
   stage -> HostGraphLaunch(exec_[T], ASYNC_LAUNCH=1) -> ar_on: publish PLE flag once (:2368-2378)
   -> launch.finish() + sync #1 (:2479-2488) -> HOST sampler + sync #2 (:2509-2541)  <-- R3
[rescue probes, device-side under TYPICAL_DEVICE] (generate.cpp:13231-13253)
ver.commit(a+1)  async         (generate.cpp:13269; commit graph hides under what follows, F-232)
emit (pipe-buffered printf)    (generate.cpp:13306-13314; F-232 C7)
mtp.set_draft_history + mtp.draft(...)  (generate.cpp:13320-13323 -> mtp.cpp:1562-1775)
   round graph (catch-up + row-a forward + select; mtp.cpp:1311-1357)
   gated step loop: launch step j -> spin h_out_[j-1]/h_prob_ -> trim at min_p  (mtp.cpp:1725-1743)
pcie_auto_step, adapt join, policy.observe; loop
```

Facts this read pinned (each load-bearing for the ranking):
- The draft **graphs are device-chained**: step j's input token is written into `tok_[0]` by
  `mtp_select` inside step j−1's graph (mtp.cpp:1359-1360, :1382); the host gate exists ONLY to
  avoid launching steps the trim will discard. F-229 measured the trade: ungated = −2.73% M1 —
  waste (~50 µs/discarded step) exceeded the launch bubbles it removed. That measurement BOUNDS the
  bubble value at ≈ +2.4 t/s, which is exactly R2's ceiling.
- The AR verify window is already "slim": on `ar_on()` the graph takes `resident_plan` on-device
  (verify.cpp:1558-1561), NO doorbell publish (the else at :1562-1590 is the non-AR path only),
  the host serves no layers (:2368-2378), and the waits are `wait_flag_ge_planned` with skip values
  (:1685, :1698). The capture comment at :3644 says it outright: "the zero-doorbell graph never
  rings". There is no scaffolding left to delete.
- The shared expert runs on a forked stream by default on CUDA (`sh_stream_on()` = true unless
  overridden, verify.cpp:212-221; fork at :1505-1507; the LATE fork variant is deliberately
  suppressed on AR at :1506). Its 0.8 ms line (T15A segment 6) is therefore mostly concurrent with
  the routed-expert line already — the tqllm T1.1 fusion's residue is launch/event overhead only.
- The dense/attn 3.5 ms line dispatches through the MMQ family (`mm` lambda, verify.cpp:1163-1173);
  there is no separate BF16 dense-GEMV lever — `bf16_gemv_fp32_mmvf*` serves only PLE and the qsa
  indexer projections (verify.cpp:1054-1072, :1318-1324, :1421-1442). The IL-cells item (T15A §3.1)
  is the dense lever.
- The head chain is norm+mix (fused_gr) -> q8_1 -> plain `native_mmvq` or IL (verify.cpp:1840-1886)
  — the T16-adjacent IL/Q6_PACKED items; nothing new.
- The hc line's anatomy stands as T15A §3.3 described: one fp32 accumulator per lane,
  `for (j = lane; j < D/8; j += 32) acc += dot8(...)` (fused_gr.cu:126-129) = the 320-FMA dependent
  chain; T16's rungs (UPM_COLS, staged-down, TSUM, multi-acc, HC_Q8) remain the right medicine.
- `STRATA_DUMMY_NODES=k` inserts k empty kernel nodes per layer at capture
  (verify.cpp:1152-1155, "what one node costs in a window") — a purpose-built instrument for the
  node-tax measurement R1 needs.

---

## 3. Ranked candidates (candidate | mechanism | anatomy-priced magnitude | cost/effort | cheapest falsifier | citations)

| # | Candidate | Mechanism | Magnitude | Cost | Cheapest falsifier | Citations |
|---|---|---|---|---|---|---|
| R1 | **Megakernel-class AR window** (persistent per-layer-group step kernel, counter-based barriers, weight prefetch/backfill) | Deletes the ~2,050-node dependency serialization: W(1) 8-9 ms vs ~2.1 ms byte floor ⇒ ~6-7 ms/window fixed node tax; Cohere production bs=1 MoE decode 1.58x vs vLLM at 62% SoL is the class precedent | **−3..−6 ms/window ⇒ +60..120 t/s (250→310-370 class)** | L-XL (weeks; AR-path only; conc path keeps host service) | **FREE: STRATA_DUMMY_NODES 0/1/3 A/B boot** measures per-node cost on this rig; then a one-layer-group counter-barrier pilot | verify.cpp:1152-1155, :2348-2360, :3644; T15A §2.1-2.2; [S32-35] |
| R2 | **Device-side conditional-node draft chain** | CUDA 12.4+ IF/WHILE graph node wraps the step body; condition = `m_prob_[j] >= min_p` already in device memory; kills per-step host launch bubbles WITHOUT F-229's over-draft | +1..2.5 t/s (bounded by F-229's −2.73% = the bubble value) | M-C (prototype-gate sm_120 + capture semantics) | FREE first: `STRATA_DRAFT_TIMING=1` boot reads `g_dt.gap_ms` (instrument exists, mtp.cpp:1751-1763); if gap < 0.05 ms/window, retire | mtp.cpp:1678-1693, :1697-1743, :1751-1763; F-229; [S5, S6] |
| R3 | **Sampler-in-graph** (full sampler, not just sync-elision) | Sampled windows re-sample host-side after sync #2 and discard the in-graph greedy pass (verify.cpp:2509-2541); params are request-constant + `pos0` + device `d_hist` — all stageable; Philox draw is position-keyed so capture-safe | +1..2 t/s (launch+sync ~40-80 µs + discarded in-graph argmax ~10-20 µs) | M, exactness-critical | dt_run vs dt_commit columns on one boot; draw parity via the existing STRATA_SPEC_PROB harness | verify.cpp:2509-2541, :1868-1880; T15A §3.8; [S23] |
| R4 | **First-window draft seeding** (run `mtp.draft(1,[x],p,0)` right after prefill) | The machinery is identical to the standing post-window draft call; window 1 becomes a full-T window instead of hard T=1 (generate.cpp:13119) | +0..1% probe mean; TTFT −8-9 ms on every request | XS-S | The arm itself (one boot); read rows[0] of first requests | generate.cpp:13113-13119, :13320-13323; mtp.cpp:1562-1588 |
| R5 | **L2 set-aside retarget to the drafter working set** | Standing 60 MiB sits on GDN states whose line is near floor (rec 0.23 ms, T15A §4.E); the drafter re-reads ~66 MB (13.8 MB Q2_0 experts + ~52 MB Q4_0 head) 4-6x/window — L2-resident steps 2..k run at L2 bandwidth | +0.5..1.5 t/s | XS (env + one policy-window arm) | One arm + ncu `lts__t_sector` on the draft phase; or straight t/s ABAB | launch-custom-iq3s-8081.sh:25; T15A §4.E; mtp.cpp:1379-1382 |
| R6 | **Post-retrain rider: learned depth gate** (the retrained head's own confidences replace scalar min-p; DSpark-class confidence scheduling) | min-p is a static proxy for a per-position accept-probability the retrained head can emit directly; DSpark trains exactly this head and schedules verify length from it | rider on T16 W2/W5; 0..+4% class | rides W2's training run | The retrain's own gate ladder (T14C G5) re-run with the learned gate | [S4, S28, S30]; SVG-CUSTOM-TODO T16 #5 |
| R7 | Cross-window "one round ahead" draft/verify overlap | DEAD externally corroborated: exact only for token-conditioned drafters (SGLang #27462); hidden-state MTP needs K+1-enumeration → c_row law kill | 0 (structural negative) | — | — | [S10]; T15A §3.10; F-223 |

---

## 4. The top candidates in detail

### R1 — the megakernel-class AR window (the only 300-class lever on the board)

**Anatomy.** T15A measured W(1) = 8-9 ms warm for a one-row window. The T=1 byte floor: hc
26.3 MB/layer × 48 ≈ 1.26 GB, dense ~1 GB, routed hits ~1.0 GB (one row's top-10 × 48 layers ×
~2.05 MB), head 522 MB Q6_K ≈ 3.8 GB ⇒ ~2.1 ms at the 1.79 TB/s GDDR7 roofline. The 6-7 ms
difference is the fixed cost of executing ~2,050 tiny kernels as a dependency chain (~2,050 nodes ×
~3 µs; T12-GPU's node inventory and spacing measurements agree). No per-kernel rung touches this:
the T16 rungs make individual kernels faster, but the window stays a chain of ~2,050 serialized
launches. F-206's "node savings ≠ wall time" was measured for FUSIONS (merging two small kernels —
the merged body still serializes); the megakernel class deletes the serialization itself: one
persistent kernel per layer-group (or per window), task-list driven, integer-counter barriers, warp
specialization, weight prefetch while activations are still in flight [S32-35].

**New external evidence (2026).** Cohere's North Mini Code engine: a single persistent decode
megakernel for a 30B-A3B MoE on ONE H100 — bs=1 decode 292 tok/s vs vLLM 185 (1.58x), 62% of
speed-of-light, no accuracy loss, production API, Blackwell port planned [S32]. Mirage's MPK
compiler [S33], ForgeMegakernel's auto-generated decode megakernel [S35], and Hazy's tp-llama [S34]
independently document the same class. This is no longer a paper gamble: it is a shipped serving
engine on our exact regime (bs=1 MoE decode).

**Fit to our tree.** The AR path is the friendly case: the window graph is already fully
device-driven (zero doorbells, device plan, host parked in one sync — verify.cpp:2368-2378,
:3644); the host-served machinery (pool, doorbells) is what the conc lane needs and can stay on the
non-AR graphs. Our doorbell/flag machinery is precisely the counter-in-global-memory barrier pattern
the megakernel class formalizes. Scope: one persistent kernel walking the 48-layer × (hc read, dense
GEMVs, router, grouped hits, combine) chain with the existing kernels inlined as phases; T ≤ 8 rows
unrolled; the head + sampler + nonfinite guard as the epilogue.

**Magnitude.** −3..−6 ms/window ⇒ +60..120 t/s. Honest range: Cohere got 1.58x over vLLM; our gap
to byte-floor is ~2.4x, so the class ceiling here is higher than 1.58x, but our chain has the
drafter phase and host tail which Cohere does not. The 250→300-370 claim is the plausible band, not
a promise.

**Cost/effort.** L-XL: weeks of kernel engineering, HIGH blast radius, full gate class (bit-parity
harness per phase; the SageAttention #392 sm_120 graph-replay-staleness bug class [S9] is the
warning for any capture/replay restructure).

**Cheapest falsifier (do this FIRST, it is free).** `STRATA_DUMMY_NODES=1` vs `=0` vs `=3` A/B —
the instrument at verify.cpp:1152-1155 was built "to measure what one node costs in a window" and
has never been banked. If the per-node tax reads ≥2.5 µs (⇒ ≥5 ms/window of pure structure), R1 is
justified and priced; if it reads <1 µs, the structural thesis dies for one boot's cost and the T16
tranche + retrain is the whole game.

### R2 — device-side conditional draft chain

The gated loop (mtp.cpp:1725-1743) launches step j only after reading step j−1's mapped
`h_out_/h_prob_` — per surviving step, the GPU idles for one host launch round-trip. F-229 proved
the naive fix (enqueue everything) is net negative: discarded steps cost ~50 µs each and outweigh
the bubbles. CUDA 12.4+ conditional nodes [S5, S6] are the exact mechanism that gets both: a WHILE
node whose body is the step graph, whose condition reads a device word that `mtp_select` already
writes (`m_prob_[j] >= min_p`, mtp.cpp:1382). The trim stays per-step; no host round-trips; no
over-draft. Bounded value: F-229's −2.73% shows the total bubble pot is ≈2.4 t/s; realistic take
+1..2.5 t/s. Prototype-gate: sm_120 conditional-node support under CUDA 13.4 + capture semantics +
mapped-memory reads inside body graphs; check `cudaGraphConditionalParams` device-handle rules
[S5, S6]. Free first step: one `STRATA_DRAFT_TIMING=1` boot reads `g_dt.gap_ms` (the inter-step gap
accumulator, mtp.cpp:1755-1759) — if gaps < 0.05 ms/window the lever retires without code.

### R3 — sampler-in-graph

For sampled/penalized requests (all our probe traffic), the graph's in-window greedy pick
(verify.cpp:1868-1880) is discarded and the sampler re-runs host-side after the window sync, then
syncs AGAIN (:2509-2541). The stated blocker — "a captured kernel would replay the same draws
forever" (:2501-2504) — does not actually bind: draws are Philox(seed, pos0+t), keyed by POSITION
(:2502-2504), and `pos0` is exactly the kind of per-window scalar the engine already stages through
mapped memory everywhere else (h_step_, h_pos_, d_hist). Moving the sampler into the graph with
staged params saves the second sync + launch (~40-80 µs) plus the wasted in-graph argmax
(~10-20 µs) ⇒ +1..2 t/s. Exactness-critical (draw parity is checkable against the host path with the
STRATA_SPEC_PROB machinery, verify.cpp:2513-2535). This EXTENDS T15A §3.8 (sync-elision, +1-2 t/s);
the graph move is the same class and supersedes it if it lands.

### R4 — first-window draft seeding

`first_window` forces T=1 (generate.cpp:13119); the first `mtp.draft` call only happens after
window 1 commits (:13320-13323). But the draft call for window 1 needs exactly what prefill already
produced: the last prompt token `x` at cell `p` with `a=0` — the same `draft(1, [x], p, 0)` shape
the loop runs every round. Calling it right after prefill seeds `drafts[]/dprob[]`, so window 1
runs the min-p trim like every other window. One extra drafter round (~0.15-1 ms GPU, overlapped
with the host's post-prefill work) buys (T−1) × ~2.3 ms of marginal rows on window 1 of every
request. Probe-mean effect is small (+0..1% — the probe's request mix decides), but TTFT drops by
up to a full window's time, which is user-visible in interactive use. XS-S code, one boot to price.

### R5 — L2 set-aside retarget

`STRATA_L2_SETASIDE_MIB=60` (launcher line 25) arms the access-policy window over the GDN states —
but the rec line it protects is already near floor (0.23 ms/window, T15A §4.E). The drafter,
meanwhile, re-reads the SAME ~66 MB every step of every chain (13.8 MB Q2_0 drafter experts + the
~52 MB Q4_0 40,525-row draft head): steps 2..k of a chain hit the same lines steps ago evicted.
Retargeting the set-aside (or dual-arming with a smaller GDN window) makes those reads L2-hits.
XS effort; falsifier is one arm plus `ncu --metrics lts__t_sector` on the draft phase. This is a
NEW application site for the already-adopted mechanism, not the closed experts-in-L2 idea (T7.C's
do-not-bother was about the 85 GB expert tiers; the drafter's 66 MB fits).

---

## 5. Re-prices of T16 items on new external evidence (allowed re-pricings, nothing re-litigated)

- **T16 W2 FastMTP retrain — DE-RISKED.** Red Hat shipped FastMTP-style head fine-tuning in
  Speculators 0.6.0 as the production recipe for vLLM (Sep 2026), explicitly on the
  single-shipped-MTP-head class we have [S2]; the vLLM speculators docs now carry the MTP algorithm
  as first-class [S3]; DSpark's comparisons corroborate that MTP-style heads respond to
  inference-matched training [S4]. Our GO verdict (T11.R3) and gate ladder (T14C) stand; the
  ecosystem now runs the same recipe at production scale.
- **T16 W1a/W1b hc rungs — corroborated ceiling math.** "Mind the Memory Gap" (arXiv 2503.08311)
  confirms decode GEMV/flat-GEMM stays memory-bound with ~1 FLOP/byte arithmetic intensity [S23] —
  consistent with our hc line being 3.8x off its byte floor via a latency chain (T15A §3.3), i.e.
  the rungs attack the right thing (issue order + sum order), not FLOPs.
- **T16 #5 min-p re-stair — gains a post-retrain successor.** AdaEDL (entropy lower-bound stopping,
  ICML 2025) [S30] and HeteroSpec (ACL 2026) [S28] publish 10-57% / EAGLE-3-beating gains from
  confidence-adaptive stopping; DSpark trains a dedicated survival-probability head for it [S4].
  After W2 lands, the retrained head's own per-position confidences can replace the scalar min-p
  (R6 rider) — the same lever the literature keeps re-finding, in the form our engine can adopt
  without contract surgery.
- **MTP-at-bs=1 class sanity check.** llama.cpp's native MTP (merged May 2026) roughly doubles
  Qwen3.6-27B decode on GB10 [S29, S31], with ~3x class reported on a 4090 [S30-note: S32-list,
  pedroalonso]. Our 2.47 tokens/window at AL 3.25-3.4 sits inside the class band; the FastMTP bet
  is calibrated against sibling engines, not against theory alone.
- **FlashInfer SM120 groupwise GEMM (v0.6.15+, Sep 2026) — DOESN'T transfer to our decode matvec**
  (NVFP4 tensor-core path; our bytes analysis shows every tensor-core format costs MORE bytes than
  IQ3_S at bs 1-2 — T7.C verdict stands), but it is the watch-item for the day quantized grouped
  GEMM ever beats per-blob matvecs on sm_120 [S43, S44].

---

## 6. Contradictions and their resolution

1. **"Run the drafter ahead / overlap draft with verify" (this tasking's serial-chain-escape hint)
   vs F-223.** Resolved EXTERNALLY: SGLang #27462's design work reached the same wall — the only
   exact form is enumeration over accept outcomes, which is exact only for token-conditioned
   drafters [S10]. Ours is hidden-state-conditioned; the c_row law closes enumeration. The serial
   chain is a contract property, not an engineering gap.
2. **"Shared-expert fusion (tqllm T1.1) +4-6%" (T6-era queue) vs this read.** The fork is default-on
   on CUDA (verify.cpp:212-221, :1505-1507) — the fusion's fusion-target (the separate serial
   launches) no longer exists as a serial segment. Residual value = launch/event overhead only.
   Do not re-queue T1.1 on the old +5.9% (that measurement was on tqllm's tree where no fork
   existed). T6.4 is hereby retired with this citation.
3. **"AR-graph slimming" (T12-GPU-era idea) vs this read.** Already built: zero-doorbell capture,
   device plan, planned-skip waits (verify.cpp:1558-1561, :1678-1698, :3644). What remains is R1's
   much deeper structural move, not node deletion.
4. **F-229 (ungated draft negative) vs R2.** Not a contradiction: F-229 tested the ONLY two
   policies the code had (gated vs ungated); conditional nodes are the third policy (per-step trim
   at zero host cost) that neither arm implements. R2's ceiling is derived FROM F-229's bound.
5. **Cohere 292 t/s vs our 198.** Different model (30B-A3B bf16 vs our larger MoE at IQ3_S with
   expert offload); the transferable number is the 1.58x-over-vLLM ratio and 62%-of-SoL, not the
   absolute. Stated here so nobody quotes 292 as a comparable.

---

## 7. Open questions

1. What does the STRATA_DUMMY_NODES A/B print on this rig (µs per node per window)? — decides R1's
   entire business case. One boot.
2. What does `g_dt.gap_ms` accumulate per window at standing? — decides R2. One boot with
   STRATA_DRAFT_TIMING=1 (the instrument is built; its report has never been banked in any log I
   could find — cohorts log contains only the stall line).
3. Does `cudaGraphConditionalParams` + capture behave on sm_120 under the vendored CUDA 13.4,
   including mapped-memory reads inside the body graph? — 20-line prototype (T8.A's device-graph
   restrictions note applies to device LAUNCH; conditional nodes are host-launched and may evade it).
4. Does the in-graph sampler reproduce host-sampler draws bit-exactly on the probe corpus
   (STRATA_SPEC_PROB harness)? — decides R3.
5. Post-W2: does the retrained head emit calibrated per-position confidences usable as the depth
   gate (R6), or does min-p remain the better-calibrated proxy? — the T14C gate ladder answers
   incidentally.

## 8. Implications

- Execute T16 as queued (W0 re-bank at 3135 → W1a/b kernel rungs → W2 retrain → W5 re-stair). This
  audit changed none of it and de-risked W2.
- Spend the two free boots BEFORE any new code: STRATA_DUMMY_NODES A/B (R1's falsifier) and
  STRATA_DRAFT_TIMING (R2's falsifier). Between them they price the two structural candidates for
  the cost of an afternoon and decide whether the 300-class lane exists on this contract.
- If R1's falsifier fires (node tax ≥2.5 µs): scope the AR-path megakernel as the campaign's next
  L-tier project, with the Cohere/Hazy/MPK/Forge four as the design references and the parity
  discipline (per-phase bitwise harness, SageAttention #392 staleness class as the named hazard)
  as the gate.
- R3/R4/R5 are the tail polish: +2..5 t/s combined for S-M total effort; R4 additionally buys TTFT
  on every request.
- The delusional 270 remains gated on R1 or a DSpark-class drafter project; nothing cheap reaches it.

---

## 9. SOURCES (45; each: URL — relevance — verdict)

1. https://arxiv.org/abs/2509.18362 — FastMTP: recursive teacher-forced MTP head training, the T16 W2 recipe — **APPLIES** (draft-only, lossless, our exact head class).
2. https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads — Red Hat productionizes FastMTP fine-tuning in Speculators 0.6.0 for vLLM (Sep 2026) — **APPLIES** (same single-native-MTP-head mechanism; de-risks W2).
3. https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp — vLLM speculators MTP algorithm docs — **APPLIES** (recipe plumbing for our head class).
4. https://arxiv.org/abs/2607.05147 — DSpark: trained confidence head schedules verification length per request (2026) — **PARTIAL** (its block-diffusion drafter is out of scope; its confidence-scheduled LENGTH gate is the R6 rider on our min-p).
5. https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes — conditional IF/WHILE nodes in CUDA graphs (12.4+) — **APPLIES** (R2's mechanism).
6. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html — CUDA graph conditional nodes: device-memory condition handle, body graphs — **APPLIES** (R2 semantics/limits).
7. https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains — CUDA 13.1: green contexts runtime API, Tile — **PARTIAL** (green contexts = conc lane; Tile not applicable to our hand-CUDA kernels).
8. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html — CUDA 13.4 Update 2 release notes (our toolkit line) — **APPLIES** (currency check; R2 prototype-gate target).
9. https://github.com/thu-ml/SageAttention/issues/392 — sm_120 silently-wrong output when a captured graph replays with new inputs — **APPLIES** (the named hazard class for R1/R2 graph restructures; validates our parity-gate law).
10. https://github.com/sgl-project/sglang/issues/27462 — Parallel Speculative Decoding roadmap: enumeration buffer, "one round ahead driven by commits" (fetched in full) — **APPLIES as a structural negative** (exact only for token-conditioned drafters ⇒ closes cross-window overlap for our hidden-state MTP; corroborates F-223 externally).
11. https://www.lmsys.org/blog/2025-07-17-mtp — SGLang MTP blog: draft_token_num/batched MTP serving — **PARTIAL** (batched-serving shape; our single-stream window differs but the draft-graph-per-step pattern is shared).
12. https://docs.sglang.ai/advanced_features/speculative_decoding.html — SGLang spec docs (EAGLE/MTP/DFLASH/NGRAM) — **PARTIAL** (all alternatives need training or trees; our native MTP + exact-match is the differentiator).
13. https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html — AMD ROCm MTP for DeepSeek V3 + SGLang — **DOESN'T** (HIP stack) but corroborates single-NextN-module MTP design.
14. https://reducto.ai (blog: when does accepted speculative decoding translate into throughput/latency, Jul 2026) — acceptance→wall-clock conversion analysis — **APPLIES** (independent statement of our c_row law: acceptance pays only while rows cost less than they return).
15. https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus — vLLM spec survey incl. MTP/EAGLE-3/DFlash/DSpark — **PARTIAL** (ecosystem map; no single-stream-5090 data).
16. https://github.com/vllm-project/vllm/issues/37754 — FlashInfer+MTP crash under serve (Mar 2026) — **DOESN'T** (their bug; landmine only — spec+backend interaction fragility).
17. https://forums.developer.nvidia.com/t/qwen-qwen3-6-35b-a3b-and-fp8-has-landed/366822?page=2 — decode ~-40% with spec in some single-GPU vLLM setups — **PARTIAL** (spec can LOSE at bs=1 when the draft phase is host-heavy — exactly the failure our gated/coupled design avoids; context for the 250 bar).
18. https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference — vLLM spec guide (Dec 2025), 1.4-1.6x — **PARTIAL** (background).
19. https://nvidia.github.io/TensorRT-LLM — TRT-LLM DeepSeek R1 MTP implementation/optimization docs — **PARTIAL** (batched, multi-GPU orientation; their MTP graph patterns informed our design).
20. https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html — B200 min-latency playbook (Mar 2026) — **PARTIAL** (overlap+graph techniques; B200/tcgen05 results do not transfer to sm_120).
21. https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference — NVIDIA spec-decoding intro, EAGLE-3 deployment (Sep 2025) — **PARTIAL**.
22. https://arxiv.org/html/2506.11309v1 — SwiftSpec: ultra-low-latency scaled speculative decoding — **PARTIAL** (distributed disaggregation; not our box).
23. https://arxiv.org/html/2503.08311v2 — "Mind the Memory Gap": decode stays memory-bound GEMV, ~1 FLOP/byte — **APPLIES** (corroborates the byte-floor thesis behind T16's kernel rungs and R1's ceiling math).
24. https://arxiv.org/abs/2411.15162 — FR-Spec: frequency-based draft-vocabulary pruning, up to ~1.97x — **PARTIAL** (our draft head is already a 40,525-row sub-vocab; the remaining trim is marginal).
25. https://arxiv.org/html/2506.22694v2 — VocabTrim: training-free draft-vocab pruning — **PARTIAL** (same as above; possible micro-rider on the drafter head, +0..1%).
26. https://github.com/ggml-org/llama.cpp/issues/25187 — FR-Spec-style draft-vocab trimming discussed for native-MTP GGUF — **APPLIES** (sibling engine, same trade).
27. https://arxiv.org/abs/2410.18351 — AdaEDL: entropy lower-bound draft stopping, +10-57% vs static (ICML 2025) — **APPLIES** (post-retrain successor to the scalar min-p; R6 evidence base).
28. https://arxiv.org/abs/2505.13254 (+ ACL 2026 https://aclanthology.org/2026.acl-long.589/) — HeteroSpec: heterogeneity-adaptive verification effort, beats EAGLE-3 — **PARTIAL** (tree-effort allocation; we are chain-only, but the confidence-adaptive gate transfers).
29. https://news.ycombinator.com/item?id=48025248 — llama.cpp native MTP merge discussion (May 2026) — **APPLIES** (sibling engine ships MTP-at-bs=1; 2x class on GB10).
30. https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090 — Qwen MTP on a 4090, ~3x class — **APPLIES** (consumer-GPU MTP multiplier calibration).
31. https://github.com/ggml-org/llama.cpp/issues/27469 — exposing MTP/spec in llama.cpp public C API (Aug 2026) — **PARTIAL** (ecosystem maturity signal).
32. https://cohere.com/blog/megakernels — Cohere production decode megakernel: 30B-A3B MoE, one H100, bs=1 292 vs vLLM 185 (1.58x, 62% SoL); counter barriers, warp specialization, weight prefetch (fetched in full) — **APPLIES** (R1's headline precedent on our exact regime).
33. https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17 — Mirage MPK: compile an LLM into one persistent kernel — **APPLIES** (R1 class, compiler route).
34. https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro — Hazy tp-llama: whole-model kernel fusion for low-latency decode — **APPLIES** (R1 class).
35. https://arxiv.org/html/2609.12379v1 — ForgeMegakernel: auto-generated whole-step decode megakernel — **APPLIES** (R1 class, 2026).
36. https://arxiv.org/html/2312.12456v1 — PowerInfer: hot/cold expert locality on consumer GPUs — **PARTIAL** (our hits line is at roofline and 97.7% resident; offload locality is not an M1 lever here).
37. https://developers.redhat.com/articles/2025/09/12/run-qwen3-next-vllm-red-hat-ai-step-step-guide — Qwen3-Next (hybrid GDN + MTP, our architecture family) on vLLM — **PARTIAL**.
38. https://recipes.vllm.ai/Qwen/Qwen3-Next-80B-A3B-Instruct — official Qwen3-Next recipe (hybrid attention + MTP) — **PARTIAL**.
39. https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3 — Red Hat EAGLE-3 speculator release — **PARTIAL** (separate-drafter class; our native MTP path differs).
40. https://github.com/vllm-project/vllm/issues/24660 — MTP+custom-GPU bug on Qwen3-Next in vLLM — **DOESN'T** (landmine only).
41. https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm — production TRT-LLM spec decode (May 2025) — **PARTIAL**.
42. https://rocm.blogs.amd.com/software-tools-optimization/speculative-decoding---deep-dive/README.html — AMD spec-decoding deep dive (Mar 2025), 2.31x class — **PARTIAL**.
43. https://flashinfer.ai/releases — FlashInfer release highlights: native SM120/121 groupwise GEMM (Sep 2026) — **DOESN'T** for our IQ3_S decode matvec (NVFP4 GEMM costs more bytes at bs 1-2 — T7.C law); watch-item.
44. https://github.com/flashinfer-ai/flashinfer/issues/4006 — v0.6.15: NVFP4 MoE on Blackwell SM12x — **DOESN'T** (same reason).
45. https://arxiv.org/abs/2406.02532 — SpecExec: massively parallel tree verification for consumer GPUs / RAM offload — **DOESN'T** (tree = verify-contract surgery, dead by c_row law at our row cost; offload regime not ours).

Source counts by category: literature (arXiv/ACL) 15 (1, 4, 14, 22, 23, 24, 25, 27, 28, 35, 36, 42, 45, plus 30's blog-practitioner and 20's tech blog are practitioner — see below); specifications/docs 6 (5, 6, 7, 8, 12, 19, 38); practitioner/blogs 13 (2, 13, 15, 17, 18, 21, 30, 32, 33, 34, 37, 41, 42); repos/issues/data 11 (9, 10, 11, 16, 26, 29, 31, 39, 40, 43, 44). Overlaps resolved by primary-listing each source once; 45 total.

---

## 10. Methods note (incl. unproductive searches, not repeated)

Code read this session (all line refs verified, not inherited): generate.cpp (13101-13405 solo loop);
verify.cpp (205-221 sh_stream, 1120-1200 gr_read/mm, 1495-1630 fork/router/shared, 1678-1720 waits,
1840-1890 head, 2330-2564 run tail, 2604-2625 fetch_dma, 3115-3175 batch path, 3644 zero-doorbell
comment); mtp.cpp (85-120 catchup env, 1311-1387 captures, 1562-1775 draft/draft_launch);
fused_gr.cu (81-165 gr_down/gr_up); layer.hpp (100-155 MoE block + split rationale); launcher
launch-custom-iq3s-8081.sh (standing env). T15A's fused_gr line cite (:128-131) is :126-129 in the
current tree — same code.

Unproductive searches: (1) STRATA_DRAFT_TIMING output in all custom-iq3s*.log — the instrument has
never been run to completion; only a stall line exists in cohorts.log. Its report is a free
pending measurement. (2) No per-node cost number exists anywhere in the campaign record —
STRATA_DUMMY_NODES was built for exactly this and never banked (found at verify.cpp:1152).
(3) The T6-era "T1.1" label collides (tqllm shared-expert fusion vs the doorbell partial plan);
F-222's NEUTRAL applies only to the doorbell one — resolved here by code read (§6.2). (4) Searched
for a published coupled-RNG + sharpened-draft benchmark (T7.B.5's claim it does not exist): 2026
literature still shows none — the claim survives, now as a checked negative. (5) Searched for any
2026 production bs=1 MoE megakernel besides Cohere: Mirage/Hazy/Forge are the field; none targets
sm_120 yet — Cohere's "Blackwell planned" is the opening our box could exploit as evidence, not as
a dependency.

Law compliance: no T16 item re-appeared as new; R6/R7 and the §5 re-prices carry new evidence; the
CLOSED list (FUSE_WAIT, LFUSE, EcoSpec tree-filling, DOORBELL_STORE, norm-fold, PDL, QFUSE,
host-axux pinning, dense-MMQ prefill/F-230, IndexShare/F-214, GDN slot-promotion/F-216, GR_WIDE/
F-217, onepass planner/F-225, DRAFT_UNGATED/F-229, batch G=2/F-223, emit/HostGraphLaunch/admission
syncs/F-232) is honored — nothing above re-proposes any of them.
