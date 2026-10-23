# SVG-CUSTOM FEATURES — the core differentiators of this fork

Curated 2026-10-23 from SVG-CUSTOM.md (F-01..F-273). Every claim carries its journal reference and
its measured number. This is the "what makes this stack ours" document; the chronological record
stays in SVG-CUSTOM.md, the doctrine in AGENTS.md, the live queue in SVG-CUSTOM-TODO.md.

**The scoreboard line:** engine-only gains of **+17.8% to +33.4% over upstream stock at identical
quant bytes** (the 2×2 same-quant matrix, F-234a: custom-IQ3_S 188.8/187.8/272.0 vs stock
141.5/146.8/220.2 solo/pair/conc; custom-Q2_0 +17.8/+23.5/+24.3) — held through the v0.2.1 merge —
with quality floors (MMLU-Pro 95.0 mean, mimic 87.9) never traded. Standing five-boot means have
climbed 198.5 → 212.5 (merge epoch) → elastic epoch, at 74–87% speculative acceptance.

> **The lifecycle era (F-272..F-276):** the engine now treats VRAM as a schedule, not a partition —
> see §1a below for the full mechanism.

---

## 1. The memory architecture — "never take the long path"

**MAGMA — the pinned resident complement (the founding logic, v1's name).** The entire expert mass
this GPU cannot hold lives as a page-locked, device-mapped host complement (~40–42 GiB) built at
boot and re-warmed post-pin (prewarm2). The file tier NEVER reads at serving time — an expert the
cache misses arrives over pinned memory at PCIe speed, not from disk. This is the foundation every
later win compounds on (F-0xx through F-64; the boot line: "mapped pinned cache complement ready,
39.36 GiB in 18 s").

**The elastic VMM triad (F-272/F-273 — adopted).** Session slots, KV, and the expert cache share
ONE virtual-memory spare-chunk pool. Slot arenas are reserved at stable virtual addresses and
mapped physically only while conversations use them: promoting a slot costs 12–17 ms off the decode
path, demoting returns it in ~10 ms — and the expert cache immediately eats the freed chunks and
refills hot experts from the routing profile. Idle capacity becomes cache instead of parked VRAM.
Measured: concurrent +4.0% (3/3 interleaved round wins) from the +1,300 slots; **9 concurrent
clients at 6-client throughput parity with zero solo tax** (F-274's F-271 question answered);
rollback is one env (STRATA_ELASTIC_SESSIONS=0).

**The doorbell graph duality.** Every captured window shape exists in TWO graph families —
zero-doorbell (all experts resident) and doorbell (experts out of VRAM: "a prompt loan, a shrink or
a swap"). Residency changes swap families; graphs are never recaptured. Combined with the cache
living in a stable-VA VMM range, this makes live cache shrink/grow invisible to CUDA graphs — the
guarantee other engines buy with indirection tables (verify.cpp:2496-2511; the T33 capture audit
proved all four capture families bake only slot/lane/model-affine memory).

**The adaptive expert tier + routing credit.** Runtime expert residency is governed by measured
routing: profile-driven initial fill (static + learned), PINCAP victim protection, dormant-expert
eviction, decode-share/lag/pool-spin tuning, and prediction credit (PREDICT_CREDIT/BUDGET — proven
at 103k context: 128 t/s at 97.6% hits). The engine learns its own traffic.

**The conversation cache.** 48 GiB of host-side prefix reuse — returning conversations resume from
cached context instead of re-prefilling (the sub-millisecond cache-hit pattern).


## 1a. The lifecycle era — VRAM as a schedule (F-272..F-276)

The elastic VMM triad above is completed by a lifecycle that keeps it in its best mode, every term
measured on our own logs:

- **Park, not drop (STRATA_ELASTIC_PARK).** Demote runs the conversation through the ConversationCache's
  own parking path (host D2H ~50-63 ms, in true idle) instead of discarding it; a returning
  conversation restores in **9–15 ms** versus the **275–326 ms** re-read it previously paid (~25–30×).
- **The spare watermark (STRATA_ELASTIC_WATERMARK).** One invariant — the spare pool always holds one
  arena's worth of physical chunks (0.75 GiB, ~24 superchunks; demote releases only the excess) —
  converts every promote from a synchronous cache-eviction walk (11.9–21.2 ms, measured 59/59 in the
  pre-fix logs) into a **background 2.2–8 ms map**, and demote unmaps from 6–15 ms to **2.0–2.8 ms**.
  This is Llumnix's two-threshold scheduling band made concrete in VMM chunk form.
- **True-idle demotion (STRATA_ELASTIC_TIMED_SCAN).** Demotes fire from an idle timer, not on line
  arrival — the 128–310 ms bill that used to land on the first user after a quiet stretch is gone;
  hysteresis holds (never mid-conversation, never with a queued admission).
- **VMM superchunks (STRATA_VMM_CHUNK_MIB=32).** Measured per-chunk-op cost on this GSP driver is
  **29–51 µs** (~25× the published A100 table — an independently-replicated finding that re-priced the
  whole tier), so the arena unit moved to 32 MiB: **24 driver operations per arena instead of 373**;
  the vmm layer was designed to accept any multiple of the hardware minimum, so this is a parameter,
  not a redesign.
- **True tails (STRATA_SERVE_TELEMETRY).** The engine has always timed every window and thrown it
  away, and the server recorded per-request *mean* gaps — so every ITL p99 quoted before F-276 was a
  mean-of-gaps artifact. The instrument now exposes per-window p50/p99/max, genuine per-token gap
  distributions, admission waits, and promote/demote/park counters at /metrics. Verdicts from here on
  see the truth.
- **Adoption evidence (F-276 battery):** M1 +1.4%, pair +2.7% (ahead in every interleaved round),
  conc +0.7% vs same-window control, M3b in-band — with the battery's single best M1 (228.3 t/s)
  recorded on a *nine-slot* configuration, a boot class that could not serve at all before the
  elastic era. Honored negatives on the record: parked-state LZ4-class compression is bit-exact but
  ~1.01× on int8 KV (structure-aware coding owed); the GDN gate-prescale identity moved the
  exact-match acceptance canary (90% → 75–81%) and stays off pending attribution — an algebraic
  identity moving acceptance is the F-230 phenomenon and ships on nothing unexplained;
  STRATA_L2_SETASIDE_MIB=72 is dead by device cap (persisting L2 = 60 MiB, the boot line proves it);
  the 36.8 GiB complement keeps cudaHostAlloc on its one-shot path because a single
  cudaHostRegister of that span stalls the boot (the paced path does take 2 MiB pages).

**The governing doctrines** (AGENTS.md, both user-issued): law 4b — **no terminal trade-offs** (a
lane-negative result is a scoping problem: the keep rule that lost −4.4% solo composed with the dm
fold into +4.9% pair); law 9 — **pathway mindfulness** (every proposal names the pathway it shortens
and the measured term it removes; preference order: remove a stage, fold two into one, move work
under an existing wait, then — last — add a mechanism). And the measurement laws that keep it
honest: ABAB interleave, quiet-window five-boot means for sub-4% effects, the single-engine law,
parity-before-battery. Externally validated: three 2026 systems papers (TreeWY, SpecLA, ReplaySSM)
each built the alternative to our chain design and concluded it loses — our dead laws are the
field's consensus.

## 2. The execution architecture

**COHORTS — the two-cohort verifier reactor (F-233).** A second verifier lane (ver_b) with a
two-phase batch_stage_rows/batch_go reactor and a transition fence: cohort B's pool work runs under
cohort A's GPU windows. Concurrent throughput +24.8% (306.8 vs 245.9, every round >304) — the move
that cleared 300 aggregate. Solo/pair ride an untouched path below 4 active slots.

**Captured fixed-shape windows with device-side plans.** Decode windows are CUDA-graph-captured per
shape with routing decisions consumed ON DEVICE (device_plan) — data-dependent MoE routing inside
graphs, the mechanism vLLM's closed spec-length RFC named as future work.

## 3. The speculative stack

**Coupled MTP, exact-match, with device-side rescue.** The drafter drafts with the target's chain
and Philox draw (verifiable bit-exact acceptance), temperature-scaled proposals (0.85×), gated by
spec-min-p; the device-side typical-acceptance rescue (F-200) recovers borderline windows on-GPU.
Acceptance holds at 74–87% across the whole campaign.

**The suffix drafter + arbitration policy.** Conversation-cache lookup drafting chained with MTP
under an economics policy (measured acceptance × window costs decide who drafts) — a mechanism our
sibling-fork survey found ahead of the family ("we are the donor", T23).

**The calibrated keep rule (F-269, in-tree, default-off).** The draft_cum joint-product cutoff with
a per-position calibrated acceptance table fitted on our own 1,615-window trace — and the law that
came from testing it: on fixed-shape captured graphs, withheld rows are free options; row-cutting
pays only under contention (where it composes with the dm fold to +4.9% pair).

**The dm fold (F-270).** The MoE tail's CPU-wait + row-copy + combine folded into ONE kernel with
pre-poll staging — parity-proven bitwise over 309,760 columns, +2.8% M1 on the pipeline.

## 4. Kernel arms (adopted, each parity/battery-proven)

GDN_CHUNKED (WY-chunked DeltaNet prefill recurrence, 1.68× at 32K, −141–168 ms/chunk) · FUSE_HITS
(the moe_hit_add→combine ZADD fold) · POOL_DIET (trailing repark waits skipped) · norm_fast (the
split-norm register kernel, pooled +5.2% M1) · mmvq_il · DMA_BATCH (84.9% fewer DMA submit cycles)
· KV prefetch · L2 set-aside over GDN state · the T18 pool-dispatch crash guards.

## 5. The methodology (a feature in itself — it produced everything above)

ABAB interleaved batteries (arm→ctl→arm→ctl settles arm-vs-drift) · the quiet-window five-boot law
(sub-4% effects need pooled cross-window means — earned by F-263's three-adoption reversion) ·
parity-first porting (bitwise harnesses before batteries) · probe-metrics stability capture
(TTFT/ITL/E2E percentiles + clock/temp/power + golden-hash drift alarm riding every leg) · the
spec-trace instrument (per-window draft probabilities banked for offline refits) · the KL yardstick
(F-230-anchored numerics gate) · the single-engine law (F-271) · and AGENTS.md law 4b: **no
terminal trade-offs** — a lane-negative result is a scoping problem, not a verdict.
