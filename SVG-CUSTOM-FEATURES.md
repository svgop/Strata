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

---

## 6. The research epoch (F-359..F-376 + the T/X/W sessions, 2026-10-27 → 11-02)

Curated from the T-epoch (SVG-CUSTOM.md's tail + the commit journal + `campaign/tracker/lanes.csv`).
The era's law: **falsifications are first-class** — every entry below carries its number, and the
era's negative results (E1, SMT, X1, X2, W9-composite, W12) are banked in the same ledger so no
future session re-pays for them.

**The shadow store (F-359/F-360/F-367).** A second, deliberately lighter quantization (Q2_0) whose
only consumer is the byte-bound CPU lane: the artifact of record ships in `data/packs/q2_0`, a
tmpfs placement cache is re-established by the launcher on every boot (a silent reboot-time loss
was found and normalized), and the lane pays 75 µs/entry instead of 152. Clean A/B:
**+2.9/+3.8/+20.8%** (solo/pair/conc).

**Two engines, one ledger (F-370, the dsv41 port).** DeepSeek-V4.1-Flash runs in-tree as its own
target on the same serving surface. Its decode anatomy is measured to the per-phase bytes: the
16-thread CPU lane over 110 GB of RAM-resident experts runs at **46–50 GB/s of the box's 50 GB/s
ceiling** (DS4_PROF, per-entry accounting) — the measured fact that redirected the whole DS line
from kernel work to byte cuts. Wins banked on it: `--experts 5` (+49/+47/+19%), then `--experts 4`
(+15.1/+17.4/+16.1, MMLU 90.0 vs 88.57 — the quality gate clean), DS4_ATTN_SPLIT (+9.4/+7.2/+7.5),
the BF16 router gate (bit-identical, +2.8/+2.4/+3.2).

**Truth-first telemetry (D9/D1).** The decode decomposition split with event-bracketed head GPU
time — the measurement that killed the standing "head is 40%" belief (true head: 0.29 ms; the rest
was CPU-lane wait draining into the final sync), plus per-entry lane GB/s. D1 stages the sampler
ahead of the round's one stream drain (one host sync per token instead of two). PF-5
(`DS4_SETTLE_ROUNDS`) queues the post-prompt expert-cache settle and returns — the drain overlaps
the first decode tokens instead of blocking cold TTFT up to 6 s.

**Coverage-quota residency (T62/E4, env-gated).** VRAM expert slots follow the profile's
heat-per-byte across layers by marginal coverage gain, not a uniform per-layer fraction
(+3.4/+4.3/+0.6 measured — a positive lean under the adoption bar, kept armed for composition).

**The adaptive-speculation controller (W1, adopted 2026-11-02).** The complete cost-model
controller (`src/spec/controller.cpp` — E[tokens]/step_ms(T) over window sizes, min-gain guarded,
acceptance learned from the session's own per-depth EMAs) now sizes the serial verify window:
cut-only, after the min_p walk, before the keep rule. Telemetry proves the mechanism (avg window
3.1–3.9 at ~85% acceptance held). Battery-neutral on the standard probe (all lanes inside noise);
the measured +35–38% class is live deep-context traffic — the owed live-session watch carries the
verdict there.

**The lane-trend tracker + the chart (the methodology's instrument).** `campaign/tracker/`:
append-only lanes.csv, the incline law (a standing line never steps down; falsified arms never
append), `build-trend.py` regenerating the chart with solo/pair/conc trend panels **and a
prefill-by-prompt-size panel** (both engines' history from first light to today). The campaign
laws (AGENTS.md): the incline law (12), the through law (13 — a regression stays open until it is
a net win), nothing-ships-half-done (14), and **the needle priced before it runs** (15 — an arm
whose expected win sits inside the drift band is composed or dropped, not measured).

**The arXiv research corpus (2026-10-31).** Four parallel researchers scraped the literature
against the measured walls; the synthesis (memory + `campaign/ROADMAP-dsv41.md`) ranks the
techniques with evidence and flags the dead ends — and four of its cheapest recommendations
(X1 Fate-prediction, X2 TC scorer, W9 composite, W12 GDN-split) were then implemented, measured,
and honestly falsified on this hardware, each with its mechanism named. The remaining validated
queue: the DS round CUDA graph (X3), micro-group conc pipelining (X4), speculative verification
with reuse-aware gating (X6) — designs published in the roadmap.

**The tail-skip arm (T63, adopted 2026-11-02).** Upstream's `STRATA_ROUTE_TAIL_SKIP` (a
CPU-missed expert that every token of a verify window ranks ≥ the rank is neither copied nor
computed) was OFF in our standing config the whole time — `--batch 6` defeats its CUDA default.
Arming it at rank 7 measured **+10.2% solo / +4.4% pair** (2v2 interleaved, conc flat), the
golden hash identical to control (the skipped tail changed no output on the golden set), MMLU
98.6 + 95.7, mimic = the banked control. Rank 6 is output-changing and dominated — killed.

**The wall, measured (2026-11-02).** `campaign/probe-bandwidth.c` (userland AVX-512, no root):
the DRAM ceiling is real at **49.5 GB/s** and the CPU lane rides 93–100% of it; the expert-block
access shape costs nothing; the thread curve is flat; THP can't even collapse here. The
move-bytes-faster family is closed with measurements. And the DS histogram (400K token-layers,
`DS4_ADAPTIVE_K_STATS`): **DeepSeek's top-4 routing is flat** — 92% of 4th picks carry ≥11% of
the routed mass, so gate-mass adaptive-k on DS is priced out before a single armed boot (law
15). Fewer bytes now means re-quantization (VQ/codebooks) or expert merging — those are the open
lane levers.

**Conversation persistence across restarts (T66, 2026-11-02).** The parked conversations now
survive an engine restart: at exit they drain to one session file per entry
(`slot-<serial>.bin`, the same session-file v1 as SAVE/RESTORE), trimmed to the deepest
checkpoint; at boot they restore newest-first under the RAM preflight. A restart resumes at the
parked tail instead of re-reading the prompt — the 42.7 s @261K cold re-read becomes a ~4 s
disk read. Armed by `STRATA_CONV_SLOT_DIR` (the launcher points it at
`/home/svg/LLM/data/conversation-slots`); round-trip verified live.
