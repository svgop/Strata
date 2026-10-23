# T35 — THE MEMORY/STORAGE HIERARCHY AS BUILT, AND WHAT SHAVES THE ms BETWEEN TIERS

Date: 2026-10-23 (research session 2026-10-22/23). READ-ONLY session: no boots, no builds, no commits, no GPU
work. Scope: (1) the full hierarchy as built, as a tier diagram with TODAY'S measured cost per hop (the live
log read is the 2026-10-22 19:15:26 boot, `custom-iq3_s.log`, --batch 9, STRATA_ELASTIC_SESSIONS=1, reserve
2600 — the current standing shape); (2) external methods/formulations mapped to OUR structures, building on
(not re-treading) T33 (materialization/serving), T26-6 (memory tiering/cold start), T17E (serving-leap
negatives), T13B/C (host loop/DMA), F-200/F-201 (MoE offload, zen5, pcie, nvme briefs); (3) the wasted-
hardware audit. Every internal claim carries a file:line or log:line anchor read this session; every external
claim carries a numbered source. Companion consumed, not repeated: T33-priorart-{materialization,serving}.md,
T33-ourtree-verification.md, T33-serving-delta-notes.md, T26-6, T17E, T13B, F-201-{zen5,pcie-doorbell,nvme},
F-200-moe-offload, F-263/F-271/F-272/F-273 journal entries.

---

## 1. THE HIERARCHY AS BUILT — TIER DIAGRAM WITH TODAY'S COSTS

```
                                   +-----------------------------------------------+
                                   |  TIER 0 — VRAM (RTX 5090, 32 GB, 96 MB L2)    |
                                   |  expert cache 7,503 slots / 14.23 GiB         |
   decode window, 16.39 ms         |  (batch-6 boots: 9,390 / 17.81 GiB)           |
   [log:16018]                     |  hits 96.2-99.4% warm; hits line 87-92%+      |
   <-------------------------------+  of GDDR7 roofline (T12-MEM/T12-GPU)          |
        ^   |   36.75 VRAM hits/  +-----------------------------------------------+
        |   |   layer-window               ^  adapt swaps (blob H2D, decay 0.85,     |
        |   v   (log:16018)                  |  PINCAP 48 / MIN_AGE 2 / DORMANT 64;   |
        |   +--------------------------- ----+  6,620 swapped in 107 rounds,          |
        |   | doorbell (flagA/B/_) host-        |  adapt wait 0.96 ms/window)           |
        |   | publish + plan per layer-group    |                                       |
        |   |                                   v                                       |
        |   +-------------------------------- TIER 1 — PINNED HOST RAM ----------------+
        |   0.7% of routed entries:           35.71 GiB page-locked+mapped complement   |
        |   PCIe DMA, frac 0.35,              (log:12897-12906; 8-20 s to pin+build)    |
        |   0.24 ms/layer-window,             built by 5,134-slot stage-lend + profile  |
        |   12.0 GB/s probed link             prefill 2.3 s @ 6,644 MB/s (log:40,57)    |
        |   DMA_BATCH -84.9% submits          CPU pool serves 3.2% of routed entries    |
        |   [F-12; T13B 1.4]                  = 1.03 ms/layer-window (log:16018)        |
        |                                     15 workers, CPUs 1-15, spin/park 100 us   |
        |                                     + KV host shadow: 1.55 GiB pinned TOTAL   |
        |                                       (flat across slots — log:1576)          |
        |                                     + conversation cache 48 GiB, mlocked      |
        |                                       (park 43.7-51.9 ms/conv, log:111-147)   |
        v                                     + 107.3 GiB left to the page cache        |
   KV STREAMING (QSA)                         (log:57)                                  |
   32,768 of 131,072 cells resident;                                                    |
   12,684 B/cell; 94.29% block hits VRAM,                                               |
   3.9 MiB from RAM per request (log:15389)                                             |
   +---------------------------------------------------------------+                    |
   | ELASTIC SESSION TRIAD (F-272/273, ADOPTED) — one VmmChunk      |                   |
   | spare pool at 2 MiB granules under KV / cache / slot arenas    |                   |
   | slot arena = 373 chunks x 2.09 MiB = 0.73 GiB, stable VA       |                   |
   | PROMOTE  11.3-19.6 ms (map 10.8-19.1 from spare OR cache tail, |                   |
   |          zero 0.5-0.6) [log:1701-1739, 2163-2194, 4058-4088]   |                   |
   | DEMOTE   25.6-41.1 ms (unmap 6.3-13.1 -> spare; the cache EATS |                   |
   |          the spare and refills from the profile -> spare=0)    |                   |
   |          [log:2063-2066, 2673-2676, 3201-3204, 4423-4424]      |                   |
   | ADMISSION slot-take (conversation copy-in): 32-58 ms, on the   |                   |
   |          service thread = 14% of probe wall (T26-6 c.1)        |                   |
   +---------------------------------------------------------------+                    |
        |                                                                                 |
        v                                                                                 v
   TIER 2 — FILE / PAGE CACHE                                                          NVMe
   0.1 GiB at steady state; EWMA read-ahead gate correctly CLOSED                      (tier dead
   [log:57; T26-6 b.1 tier-2]; mmap + FADV_RANDOM (expert_source.cpp:894)              at steady
   COLD START (back-to-back boots): pin_depends_on_reclaim fires -> drop_file_cache    state)
   (MADV_DONTNEED + POSIX_FADV_DONTNEED whole-file, expert_source.cpp:3474-3484)
   -> 44-55 GB re-read inside requests 1-2, 7-109 major faults, +0.67 ms/window pool
   term for the whole cold leg (T26-6 a.1; T17B 1.1-1.4). prewarm2.sh = the adopted
   re-warm package (+3-5% means est.); learned-profile boot lever DEAD (T18).
```

Per-hop costs at TODAY'S operating point (batch-9 boot, log:16018 anatomy: 16.39 ms/window = run 14.68
[GPU-reach wait 0.58 + CPU experts 3.35] + commit 0.37 + emit 0.38 + adapt wait 0.96; 315.6 rows/s wall;
slot-draft acceptance 68.7%):

| hop | today's cost | anchor |
|---|---|---|
| VRAM hit line | 87-92%+ of roofline — DONE (no headroom in bytes) | T12-MEM #1, T12-GPU |
| VRAM<->RAM adapt swap | 0.96 ms/window exposed wait; 6,620 blobs/107 rounds backgrounded | log:16018 |
| RAM->GPU (PCIe share) | 0.7% of entries, 0.24 ms/layer-window; 12.0 GB/s link; frac 0.35 | log:57,16018 |
| RAM->CPU pool (miss compute) | 3.2% of entries, 1.03 ms/layer-window, 17-38 GB/s pool vs 60-70 DRAM | log:16018; T11.R4(a) |
| KV resident window | 94.29% block hits; 3.9 MiB RAM reads/request | log:15389 |
| KV host shadow | 1.55 GiB pinned TOTAL, flat across slots (log:1576 "6 host pins per slot set, 1.55 GiB total"; CORRECTS this file's first draft and the F-272 "bounded --batch x 1.55 GiB" framing — T35-elastic-refinement's finding, log-confirmed) | log:1576; layer.cpp:903-914 |
| slot promote/demote | 11.3-19.6 / 25.6-41.1 ms per event, off the decode path | log above |
| conversation park/restore | 43.7-51.9 ms park; 32-58 ms slot-take on the service thread | log:111-147,15503; T17B 0.5 |
| file tier | 0 (gate closed); cold boot 44-55 GB re-read | T26-6 a.1 |
| boot pin | 8-20 s complement build; 2.3-2.6 s cache prefill | log:12904,14381,40 |

Provenance note: the "long-ctx 226-243 t/s warm" band is operator-provided (assignment); the journal's
older deep band was 104-128 t/s pre-F-26 (F-26: PCIe x8->x16 halved the miss/streaming cost). The deep band
is not re-measured in this session; treat 226-243 as the operator's current number, unverified here.

---

## 2. RANKED NEXT-TIER LEVERS

Grading per the F-263 law: any sub-4% candidate needs five-boot quiet-window means; levers marked
"adversarial-lane" act on events (promotes, admissions, cold legs) where the bar is event-cost, not mean.

| # | lever (code/zero-code) | hop it shortens | expected ms/% with arithmetic | risk | cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **VMM superchunks**: coarsen the elastic triad's chunk unit from the 2 MiB granule to 32-64 MiB handles (vmm.cpp: CUDA branch takes `CU_MEM_ALLOC_GRANULARITY_MINIMUM` as chunk; the HIP branch already rounds up to a chosen 2 MiB — the CUDA side can do the same at any multiple) | promote map + demote unmap + kvg grow/trim + cache-tail release | promote map 373 ops x 29-51 us = 10.8-19.1 ms measured -> 24-12 ops x ~40 us = **0.5-1.0 ms**; demote unmap 6.3-13.1 ms -> **0.25-0.5 ms**; kvg ops same ratio. Promote+demote events go from ~40-60 ms to ~2-4 ms total | coarser elasticity: cache-tail release steps of 32 MiB ~= 17 experts (release is bulk/tail anyway); KV grow steps 32 MiB ~= 2,660 int8 cells; spare-pool granularity (acceptable — spare holds whole arenas today). vmm.hpp map_range already batches SetAccess per run | S-M code; no boot needed to price the API side: a 20-line standalone microbench (cuMemCreate/Map/Unmap at 2 MiB vs 32 MiB x 373) off any running engine — the same class as F-252's mma microbench |
| 2 | **Parked-state lossless compression** (CacheGen-class delta+entropy, host-side only): compress conversation snapshots (conv cache) and the KV host shadow at park/demote; decompress to EXACT int8 bytes on restore | conversation park/restore; slot-take 32-58 ms; conv-cache depth | slot-take bytes halved -> ~16-29 ms copies; admissions are 14% of probe wall (T26-6 c.1) -> **+3-7% conc composed with the banked overlap port**; conv cache 48 GiB holds 2-3x more conversations (fewer evictions on 4-deep chat, the F-28 thrash class); shadow side minor (1.55 GiB flat — log:1576) but 262k-switch restore copies halve too | decompress CPU on restore (~ms-class, hidden in the same lane as today's copy); zero acceptance risk because restore is byte-exact int8 (no F-230 exposure — this state is never read by compute before restore) | **zero-boot**: the logs already emit snapshot bytes + copy ms (log:111, log:15503) — the byte/cost split is measurable from any results file; zstd -1 on an exported snapshot file prices the ratio without touching the engine |
| 3 | **Hugepage the pinned arenas** (F-201-zen5 #6 CARRY, externally re-grounded): mmap 2 MiB-aligned + madvise(MADV_HUGEPAGE) + touch + cudaHostRegister(Mapped\|Portable) for the 35.71 GiB complement, the KV shadows, and the conv cache pages — the register path already exists (paced registration + kv_host_alloc's register-first fallback, expert_source.cpp:3597-3660, layer.cpp:577-610) | CPU-pool GEMV reads of the complement (dTLB/IOTLB pressure); pin time; page-table RAM | pool at 17-38 GB/s vs 60-70 DRAM (T11.R4(a)): 4 KB pages -> 2 MiB recovers the TLB class; F-201-zen5 #6 prices **+5-15% on the staging/pool path** -> CPU-experts 3.35 ms/window term shrinks ~10-15% = 0.3-0.5 ms/window; pin 8-20 s -> fewer pages to lock (9.35M PTEs -> 18.2k at 2 MiB; Farshin: IOTLB-miss class tax) | THP is `madvise` on this box (/sys) so the hint is honored only if we madvise; WSL2-class partial-pin caps do not bind here (Linux bare metal); zero output bits | one boot pair; the pool GB/s and pin seconds are already in the logs; A/B = today's boot vs a hugepaged build |
| 4 | **Spare-pool floor (churn kill)**: after demote, keep ONE arena's chunks (373) in the spare; feed only surplus to `kvg_cache_eat_spare` (generate.cpp:11530-11546). Today every demote feeds the cache and every promote re-evicts the cache tail — a ~373-expert evict/refill ping-pong per cycle | promote-from-cache path; adapt-swap traffic around arrivals | promotes stay spare-fast (~12 ms) after bursts instead of re-paying the cache-tail release; kills ~700 MiB/cycle of evict+refill swap traffic (~100 ms of adapt-stream H2D per promote+demote pair at ~7 GB/s). Cost: 0.73 GiB held out = ~385 slots ~= F-254's negligible-M1 class, ~-1% conc at F-183's slope — adversarial-lane, arrivals only | the floor competes with the cache exactly when 9 clients just left (the cache wants the chunks); hysteresis flag decides | S code + the F3 battery; zero-code partial: raise STRATA_ELASTIC_DEMOTE_S (fewer cycles; env exists) |
| 5 | **CCD split of the pool** (F-201-zen5 #1 CARRY — now with the measured placement): host+doorbell pollers on the V-cache CCD (CPUs 0-7/16-23, 96 MiB L3 — verified /sys/devices/system/cpu/cpu0/cache/index3), ALL heavy GEMV workers on the fast CCD (CPUs 8-15/24-31), per-CCD queues | CPU-experts 3.35 ms/window term; fence/flag variance | ~400-800 ns/call fabric transit avoided + LOCK/MFENCE diet (F-201-zen5 #3): <=0.1-0.3 ms/window class + tail flattening. TODAY the plan is package-aware only (pool.cpp:120-181): workers 1-15 straddle both CCDs, 7 of them share the V-cache CCD with the host | scheduling changes move measured-sensitive spin/park code; battery-gated (bits unchanged, timing changed) | S-M code; falsifier: one boot with workers pinned 8-15 via the affinity mask (engine plan honors the process mask) reading ms_pool/wait from the batch line |
| 6 | **L2 set-aside 60 -> 72 MiB** (T15C flagged, never tested): STRATA_L2_SETASIDE_MIB=72 | GDN recurrent state residency in L2 (112 MiB span, 3 streams — log:85) | GB202 L2 = 96 MB enabled on the 5090 (Blackwell whitepaper); typical persisting cap = 75% = 72 MiB. +12 MiB set-aside = +20% state kept persisting; plausible **+0-2% M1** (GDN state is read+written every layer) | 12 MiB less L2 for streaming expert reads (the hits line is at the roofline — losing shared L2 could tax it); F-248 battery decides | **zero-code**: one env boot; the boot line prints the accepted carve ("60 MiB window / 60 MiB carve") — if the device clamps below 72 the line says so for free |
| 7 | **Admissions overlap port** (banked T26-4/T26-6, restated for completeness): reader-thread + stream the slot-take copies | admission 14% of probe wall | **+3-6% conc** (banked pricing); composes multiplicatively with lever 2 (fewer bytes to overlap) | L-cost, in-flight-guards class | already priced; no new work here |
| 8 | **pcie-frac re-sweep on the true base** (T15C/T25-4 #6): tools/calibrate.py has never run | miss-path split CPU-vs-PCIe | F-263: 0.55 vs 0.35 = no reproducible composed gain — only a calibrated sweep in a quiet window can move this; expected ~0 | the F-263 noise law applies in full | **zero-code**: run the existing tool in a quiet window |
| 9 | KV shadow dedup/COW across sessions (shared system prompts, T33-priorart-serving #5's COW) | pinned RAM, park bytes | workload-dependent (~0 for the probe set) | prefix-sharing infra | park-bytes log shows reused_kv_bytes=0 today (log:111) — no shared-prefix traffic to win yet |

Not levers, re-confirmed: prefetch/prediction class (T17E's verdicts stand: 97.7%-hit regime, 2608.12103's
0.3% measurement); UVM; compressed-in-RAM experts; cacheless loading; AMX (no AMX on AM5).

---

## 3. THE TOP-3, ARGUED

### 3.1 VMM superchunks (lever 1) — the promote/demote hop stops being a cost class

The elastic triad is F-273's adopted centerpiece and its remaining price is the VMM op count. Measured
today: a slot arena is 373 chunks of 0.73 GiB (log:1701 — 2.09 MiB each = the 2 MiB granularity reported as
`CU_MEM_ALLOC_GRANULARITY_MINIMUM` on GB202, vmm.cpp:64-67); a promote maps all 373 in 10.8-19.1 ms and a
demote unmaps them in 6.3-13.1 ms. That is 29-51 us per op — consistent with vAttention v3's own correction
of its Table 3: mapping 2 MiB pages takes ~40 us per cuMemMap+cuMemSetAccess call, "CUDA lacks a batched
virtual memory API", and their answer (a fused vMemMap, background threads) is a proposal plus a hiding
strategy, not a cure [S10][S11]. We already hide (promotes run at the admission quiesce point, demotes on
the idle scan — F-272); the arithmetic says the better cure is to stop paying per-2MiB at all: 32 MiB
handles cut the op count 16x (24 ops/arena) and 64 MiB cuts it 32x, taking promote+demote from ~40-60 ms
of driver time to ~2-4 ms. The HIP branch already proves the pattern in-tree: it ignores a 4 KiB reported
minimum and works in chosen 2 MiB chunks "any multiple of the minimum" (vmm.cpp:203-207) — the CUDA branch
just never took the same freedom. The code surface is vmm.cpp's chunk constant plus the two places that
size spares; graphs are unaffected (stable VA either way; the capture audit holds — F-272). Risk is
granularity slack, bounded by ~one chunk per grow/release. This also re-prices T33: the dossier accepted
"~25-30 ms per 0.78 GiB at 2 MB granules, run in the background" (T33-ourtree v3, T33-priorart-materialization
H3); with superchunks the term drops an order of magnitude below "backgroundable" to "ignorable", and the
S_min=2 warm floor could even shrink (frees ~1.4 GiB -> ~+750 slots at the F-273 conc-pays rate).

### 3.2 Parked-state lossless compression (lever 2) — the one KV-adjacent lever the acceptance laws permit

The external KV-compression field (KIVI 2-bit per-channel/per-token [S1], KVQuant 4-bit dense-and-sparse
[S2], SlabQuant-class skew-aware slab quantization — that exact name could not be verified, see 5) attacks
KV the engine READS; our F-230 law prices any such bits change at ~-2.5% decode through exact-match
acceptance flips, and T17F already parked sub-int8 residency on the same logic. But TWO state classes are
never read by compute before restoration: (a) the conversation-cache snapshots (48 GiB, park 43.7-51.9 ms,
snapshot ~858 KB/token observed — log:111-147; restore = the 32-58 ms slot-take that is 14% of probe wall),
and (b) the KV host shadow — 1.55 GiB pinned total, flat across slots (log:1576; the per-session framing in
the assignment and F-272's "bounded --batch x 1.55 GiB" overstate it — see 4.6). CacheGen's structure-aware
coding (delta between adjacent-token K rows + arithmetic coding after quantization) buys 3.5-4.3x on exactly
this class [S3]; even a first-cut zstd on the int8 layout prices the ratio with zero engine code (falsifier).
Compression here is restore-exact: bytes out == bytes in, so no acceptance exposure, no M3b gate — the only
gates are timing and pin accounting. Value, corrected arithmetic: admission/park copy bytes halved ->
slot-takes ~16-29 ms (**+3-7% conc composed with the banked overlap port**, the T26-6 top host item), and
the 48 GiB cache holds 2-3x more conversations (the F-28 deep-chat thrash class dies). The shadow side is
now a minor rider (its flat 1.55 GiB is not a RAM problem; compressing it still shrinks restore/262k-switch
copies proportionally). This lever exists because our hierarchy has a restore-only tier — most engines' KV
compression cannot say that.

### 3.3 Hugepage the pinned arenas (lever 3) — the CPU-miss tier's bandwidth is TLB-starved

The CPU pool serves 3.2% of routed entries at 1.03 ms/layer-window (log:16018) over a 35.71 GiB pinned
arena allocated by cudaHostAlloc (4 KB pages, ~9.35M of them, expert_source.cpp:3528-3536). The pool's
aggregate 17-38 GB/s against ~60-70 GB/s real DRAM (T11.R4(a)) is parallelism/accounting-limited per
T11.R4 — but the F-201-zen5 brief already identified the physical limiter under it: 4 KB DMA/GEMV pages
pay an IOTLB/dTLB tax that 2 MiB pages recover, priced +5-15% on the staging path [S9: Farshin 2023 PeerJ
CS — the IOTLB wall measurement; 2 MiB pages restore near-line rate]. The box is ready: THP=madvise
(/sys), HugePages_Total=0, and the tree already owns the register path (paced cudaHostRegister machinery
for exactly this arena, expert_source.cpp:3597-3660; kv_host_alloc's register-first fallback, layer.cpp:577-610).
The change is allocation-shaped only — bits identical, so it gates on timing logs alone (pool GB/s, pin
seconds, ms_pool), which the engine already prints. Secondary payoffs: the 8-20 s pin gets shorter (18k
pages to lock instead of 9.35M) and page-table memory drops by GBs. Honest cap: this is a
0.3-0.5 ms/window-class lever on the miss tier, not a leap — the same honest ceiling T17E put on all
miss-tier work.

---

## 4. CONTRADICTIONS WITH BANKED POSITIONS (primary evidence only)

1. **T33-priorart-materialization H3's VMM cost table (cuMemMap 2 us) vs our measured promote arithmetic.**
   Our logs: 373 map ops in 10.8-19.1 ms = 29-51 us/op on GB202/CUDA 13.4 (log:1701-1739). vAttention's own
   v3 text states ~40 us per cuMemMap+cuMemSetAccess call and no batched API exists [S10][S11]. Resolution:
   Table 3's 2 us is the A100-era map-only figure; the load-bearing number on our driver is the ~40 us
   class, which UPGRADES lever 1 from T33's "background it" to "reduce the op count". T33's design is
   otherwise confirmed (stable VA + spare pool is exactly right; ourtree-verification's triad framing holds).
2. **F-263 (frac 0.55 falsified) vs the pcie-frac theory.** No contradiction to press: F-263's five-boot law
   is the correct instrument and the frac knob stays 0.35. The only residue is that the frac was never
   sweep-calibrated on the true base (tools/calibrate.py never run — T25-4 #6); lever 8 keeps it open at
   zero code.
3. **F-230 law vs KV compression literature.** The papers' quality claims (KIVI 2-bit near-fp16, KVQuant
   4-bit ~clean) are generation-quality claims under perplexity metrics; our contract is exact-match
   acceptance where a 1.8e-4 KL flips decisions (F-230's measured threshold, T31's yardstick bands). Both
   hold — they measure different things. The parked-state compression route (3.2) is the reconciliation:
   compress what compute never reads.
4. **Requant honest pricing (+1.9-2% M1 direct at 97.7% hit; bytes -8.3%)** — F-273's adoption data
   (conc +4.0%, 3/3 rounds, "the slot term pays on conc") confirms and extends it; no challenge.
5. **T26-6's "lone boot pays nothing"** — re-confirmed on today's boot: no drop fired (no "really free" line
   after log:12897), 107.3 GiB page cache available (log:57). The cold-start law stands.
6. **The KV-shadow pin bound (F-272/T15B framing vs measurement).** This file's first draft carried the
   journal's "bounded --batch x 1.55 GiB" (F-272) and the assignment's per-session framing; the boot line
   says "6 host pins per slot set, 1.55 GiB total" (log:1576) and T35-elastic-refinement measured it FLAT
   across slots. Resolution: the shadow is per-process, not per-session — the per-session figure is a bound
   the code never reaches on this config. Adopted here; the compressed-shadow rider in 3.2 is re-priced
   accordingly. (F-272's statement stays true as an upper bound, not as the measured cost.)
7. **Sibling dossiers (same-day T35-elastic-refinement.md, T35-serving-latency.md) — overlap reconciliation.**
   Superchunks (their #4) and the spare watermark (their #2) appear in both files: theirs from lifecycle
   forensics (59/59 promotes ran the synchronous worst mode; the bg path never fires; the idle scan bills
   arrivals the demote bill), mine from hierarchy arithmetic plus the vAttention v3 ~40 us/call grounding.
   Their refinement of my contradiction #1 is adopted: the per-call cost is GSP-RPC class, so vAttention's
   Table-3 microsecond figures do not transfer — the measured 29-51 us/op is the number, and op-count
   reduction is the cure. Their lever 1 (timed-wait idle scan) and lever 3 (STRATA_SLOT_COPY_DIRECT, zero
   code) are additional lifecycle items this file defers to them on; their lever 5 (park-on-demote) is the
   KEEP-state complement of this file's lever 2 (COMPRESS the parked bytes) — they compose, and their
   1,361 ms re-read vs 63 ms park measurement is the strongest single number on that hop. T35-serving-latency's
   host-isolation lever is the OS-side half of this file's wasted-CPU finding; the CCD split (lever 5 here)
   is the engine-internal half. No conflicts found beyond the pin bound (4.6).

---

## 5. HONEST NEGATIVES (searched, does not apply / does not exist)

- **"SlabQuant"** — no such paper findable under that name (multiple query forms; the KV-quant field's
  verified population is KIVI/KVQuant/KVTuner/NQKV/SINKQ/KVSink-class [S1][S2][S12][S13]). The assignment's
  "PQA" resolves to Product Quantization Acceleration (ISQED'23/TODAES'24) and its multiply-add-free
  attention descendant NoMAD (NeurIPS'24) [S14] — a compute-kernel lineage, not a KV-storage lever for us
  (our KV reads are indexer-bounded, T17F).
- **GPUDirect Storage** — bypasses the page cache by design (O_DIRECT/cuFile) [S6][S7]; our cold-start 44-55 GB
  re-read is a page-cache REFILL we deliberately want (prewarm2), and our whole tier-2 economics assume
  buffered reads (F-201-nvme #8). GDS = negative with mechanism; revisit only if the working set ever
  exceeds RAM (it cannot: 46.8 GiB pack vs 184 GiB).
- **InfiniGen-class speculative KV prefetch for the non-resident window** [S4] — cuts offloaded-KV PCIe
  bytes 5-10x by fetching only attended tokens, but that is approximate attention: output-changing under
  exact-match verify (T9.B trap class). Our decode-side streaming is already 94.29% VRAM-hit and 3.9 MiB
  RAM/request (log:15389) — nothing to win at today's depth; park for the 262k/1M lanes, where it arrives
  gated anyway.
- **Second copy engine / more CE parallelism** — consumer GB202 CE count is fixed; DMA_BATCH already
  collapsed submit cycles 84.9% (F-12, banked); the transfer lane's remaining residue is frac calibration
  (lever 8). Transfer-graph capture of the miss DMA cannot bake a data-dependent miss list; HostGraphLaunch
  for the batch path is DEAD (F-232 mechanism read).
- **Swap/pagefile** — exists on this box (8 GiB swap, 1.77 GiB in use, swappiness 10, /proc/meminfo); all
  engine state that matters is pinned, mlocked, or VMM-mapped — unswappable by construction. Raising swap
  buys nothing, risks kswapd noise. Non-lever.
- **NUMA placement of the complement** — moot on AM5: one NUMA node (lscpu), all PCIe through the single
  IO die (F-201-pcie #5). The real question is the CCD split, which is lever 5. The V-cache CCD is CPUs
  0-7/16-23 (96 MiB L3, /sys verified), fast CCD 8-15/24-31 (32 MiB); today's worker plan straddles them.
- **SMT-sibling workers (CPUs 16-31, idle except aux=16)** — candidate on paper (pool is parallelism-limited;
  more threads could lift DRAM utilization) but they would co-schedule with the spin/park host-critical
  threads on the V-cache CCD; rejected as a cheap arm, folded into lever 5's per-CCD design as a possible
  fast-CCD-sibling rider.
- **vTensor's prefix-reuse machinery** [S5] — targets multi-LLM/paged-KV serving; our conversation cache
  already does LCP prefix reuse and shows reused_kv_bytes=0 (log:111) — no shared-prefix traffic to
  monetize yet (lever 9 parked).
- **PowerInfer/DeepSpeed-Inference lineage** [S8] — hot/cold placement = our profile preload + adaptive
  tier (T17E 4.1 already mapped MoE-Infinity/Fiddler/KTransformers); nothing new transfers at 97.7% hits.
- **llama.cpp #28248** — still a draft RFC as of this session [S15]; T33 already banked its architectural
  lesson and ourtree-verification struck the map-table (stable VA + doorbell duality delivers it). Watch
  only.
- **Blackwell L2 beyond 96 MB** — the full GB202 die has 128 MB; the 5090 enables 96 (whitepaper [S6]).
  The fused-off 32 MB is not reachable by software; the reachable headroom is the 60->72 MiB set-aside
  (lever 6).

## The wasted-hardware audit, consolidated

1. **The second CCD is used but unplanned** (biggest finding): pool workers 1-15 straddle V-cache and fast
   CCDs; the host + doorbell pollers share the V-cache CCD with 7 GEMV workers; the F-201-zen5 split is
   banked and unexecuted; 16 SMT threads idle. Everything needed to fix it exists in the affinity plan
   (pool.cpp) — it just never had a CCD concept.
2. **L2 set-aside 60 of 96 MB** (cap ~72): the +12 MiB arm has never been tested (T15C flagged).
3. **AVX512-VNNI present and compiled** (lscpu flags; CMakeLists:1494) but per F-201-zen5 #5 it pays only
   with repacked int8 tiles — the re-quant pack build is the trigger; until then the VNNI pipe idles.
4. **Pin/dMA bandwidth headroom**: 12.0 GB/s probed PCIe link vs the ~26 GB/s class F-201-pcie measured on
   AM5; the miss path uses <1% of window — headroom with no consumer (T17E closed the prefetch lane).
5. **Page cache**: 107.3 GiB available, ~44-55 GiB used warm — correct and healthy; prewarm2 keeps it so.
6. **Swap**: non-lever (above).

---

## 6. SOURCES

Internal (read this session): src/core/expert_source.cpp (:100-102, :894, :3449-3660, :3528-3536, :5868-5884);
src/core/vmm.cpp (:55-151, :203-207); include/strata/core/vmm.hpp (:40-75); src/core/device.cu (:381-430);
src/core/layer.cpp (:577-610, :895-914); src/core/session.cpp (:50-110); src/kernels/cpu/pool.cpp (:120-210,
:609-660); src/program/generate.cpp (:5382-5537, :7939-8154, :10536-10537, :11500-11640, :16988-16989);
launch-custom-iq3s-8081.sh; custom-iq3_s-8081.json; custom-iq3_s.log (:1, :8, :33-57, :85, :111-147, :12897-12906,
:14373-14383, :15063-16033, :15389, :15503, :16018-16019); /proc/meminfo, /proc/sys/vm/swappiness,
/sys/devices/system/cpu/cpu*/cache/index3, lscpu (2026-10-23).

Banked campaign (consumed, not repeated): T26-6-memory-host-hwlock.md; T33-priorart-materialization.md;
T33-priorart-serving.md; T33-ourtree-verification.md; T33-serving-delta-notes.md; T33-dynamic-session-allocation.md;
T17E-moe-serving-leaps.md; T13B-concurrency-scheduling.md; F-200-moe-offload-agent.md; F-201-zen5-host-agent.md;
F-201-pcie-doorbell-agent.md; F-201-nvme-agent.md; SVG-CUSTOM.md F-26 (:1053), F-230 (:7886), F-232 (:~7960),
F-263 (:8899-8945), F-271 (:9096-9172), F-272/273 (:9154-9186); SVG-CUSTOM-TODO.md T12-MEM/T12-GPU (:900-985).

External:
- [S1] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache — ICML 2024, arXiv:2402.02750 (per-channel
  K / per-token V; 2.6x memory, near-fp16). https://arxiv.org/abs/2402.02750
- [S2] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization — NSDI 2024,
  arXiv:2401.18079 (per-channel + dense-and-sparse outliers + pre-RoPE + NuQ). https://arxiv.org/abs/2401.18079
- [S3] CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving — SIGCOMM 2024,
  arXiv:2310.07240 (delta + arithmetic coding, 3.5-4.3x size reduction, adaptive streaming).
  https://arxiv.org/abs/2310.07240
- [S4] InfiniGen: Efficient Generative Inference of Large Language Models with KV Cache Management — OSDI 2024,
  arXiv:2406.19707 (speculative prefetch of attended tokens' KV from CPU). https://www.usenix.org/conference/osdi24/presentation/lee
- [S5] vTensor: Towards Efficient Computation of Multi-LLM Serving Via Dynamic KV Cache Memory Management —
  arXiv:2407.15311 (VMM-based KV, prefix sharing). https://arxiv.org/abs/2407.15311
- [S6] NVIDIA RTX Blackwell GPU Architecture whitepaper (GB202 128 MB L2, RTX 5090 96 MB enabled).
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- [S7] NVIDIA GPUDirect Storage announcement + Benchmarking/Configuration Guide (O_DIRECT/cuFile bypasses the
  page cache; when-it-helps criteria). https://developer.nvidia.com/blog/gpudirect-storage ;
  https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html
- [S8] PowerInfer: Fast LLM Serving with a Consumer-grade GPU — SOSP 2024 (hot/cold neuron placement, up to
  ~11x vs llama.cpp in its regime). via ACM DL.
- [S9] Farshin, Barbette, Maguire, Kostic — "Overcoming the IOTLB wall for multi-100-Gbps Linux-based packet
  filtering", PeerJ Computer Science 2023 (2 MiB DMA pages recover the IOTLB-miss throughput drop).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10280580
- [S10] vAttention: Dynamic Memory Management for Serving Deep Learning — ASPLOS'25, arXiv:2405.04437 (v3):
  ~40 us per cuMemMap+cuMemSetAccess call; 120 calls for 240 MB; no batched VMM API; fused-vMemMap proposal;
  background-thread mapping. https://arxiv.org/html/2405.04437v3
- [S11] CUDA Driver API, Virtual Memory Management (cuMemCreate/cuMemMap/cuMemSetAccess/cuMemGetAllocationGranularity).
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
- [S12] KVTuner (ICML 2025, sensitivity-aware mixed-precision KV; surfaced in the KVQuant citation graph);
  NQKV arXiv:2505.16210 (per-token normal-distribution KV quantization) — field survey only, not adopted.
- [S13] SINKQ (2-bit KV with sink preservation, OpenReview) / KVSink arXiv:2508.04257 — sink-token-aware KV
  quantization; recorded as field population, gated out by F-230 for our read path.
- [S14] PQA: Exploring the Potential of Product Quantization in DNN Hardware Acceleration (TODAES 2024,
  https://dl.acm.org/doi/10.1145/3656643) + NoMAD-Attention (NeurIPS 2024) — the PQA/product-quantization
  lineage resolved.
- [S15] llama.cpp discussion #28248 (RFC: persistent expert slot pool for MoE CPU offload, draft) +
  issue #20757 (two-tier GPU+RAM expert cache). https://github.com/ggml-org/llama.cpp/discussions/28248
- [S16] AMD 9950X3D single-V-cache-CCD scheduling coverage (Guru3D "3D V-Cache on a single CCD"; amd_pstate
  preferred-core class) — practitioner corroboration only; the primary ground truth for THIS box is the
  local /sys L3-id read (CCD0 = CPUs 0-7/16-23, 96 MiB).
- [S17] vLLM #17612 (vAttention-style physical memory management discussion) — engine-side adoption state.

## METHODS NOTE (so nothing is repeated)

Productive: full reads of the T33 quintet, T26-6, T17E, T13B, the four F-200/F-201 briefs; the live log's
elastic/promote/demote/batch lines (the only place the per-op VMM cost is recorded); local /sys and /proc
reads that settled the CCD map, swap, THP, and NUMA questions without a boot. Searches that produced
nothing and should not be repeated: "SlabQuant" under any spelling (name unverifiable); "PQA" as a
KV-specific quantization method (it is the product-quantization acceleration paper); "HoME"-class tiering
already recorded as unfindable by T33. Note for the next reader: the assignment's "L2 set-aside only 60 MiB
of 96" is really "60 of the ~72 MiB persisting cap on the 96 MB enabled L2" — the 96/128 split is a fuse
fact, the 72 is a driver cap, the 60 is our knob. The assignment's "226-243 t/s warm long-ctx" figure is
operator-provided and was not re-measured here; the F-26-era deep band (104-128 t/s) is the last
journal-anchored number.
