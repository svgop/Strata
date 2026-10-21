# T26-6 — MEMORY TIERING, HOST LOOP, AND THE HARDWARE-LOCK WATCH

Date: 2026-10-21. Scope: (a) the cold-start gap traced to code, with the fast-warm spec re-issued after
T18; (b) our expert tier walk (GPU / pinned-RAM / file) vs basalt, real deltas only (family-falsified
ideas excluded per the negative ledger); (c) the host loop priced; (d) the consumer-Blackwell
fp32-accum "lock" question settled from every public source reachable; (e) the +792-slot VRAM
arithmetic for the 262k switch. READ-ONLY session: no boots, no builds, no GPU, no edits; the 8081
server stayed live (its log was read). Anchors are current working-tree line numbers, verified this
session. Companion docs consumed, not repeated: T25-3, T17B (§1 cold forensics, §2.2 #2), F-200-moe,
F-201-{nvme,pcie,zen5}, T23-basalt-portstudy, T15B, AUDIT-2026-10-18, SVG-CUSTOM F-246..F-252,
SVG-CUSTOM-TODO T18 + REQUANT SPEC.

---

## (a) COLD-START MECHANISM AND THE FAST-WARM SPEC

### a.1 The mechanism, end to end (code + today's log)

1. **The boot pins a ~39 GiB page-locked complement of the expert set in host RAM.** After the GPU
   cache is sized and pre-filled, FileExpertSource allocates the complement via `cudaHostAlloc`
   (`src/core/expert_source.cpp:3527-3536`, mapped/page-locked with a device alias). Today's boot:
   `custom-iq3_s.log:426654` "allocating 38.70 GiB page-locked cache complement" → `:426661` "mapped
   pinned cache complement ready: resident 38.70 GiB, pinned 38.70 GiB in 8 s" → `:426663` "resident
   RAM mode: 38.70 GiB of experts in RAM (page-locked), 9390 in the GPU cache; adaptive swaps exchange
   them with the GPU cache (no file reads)".
2. **The driver's one-shot page-lock cannot wait for kernel reclaim, so when really-free RAM is short
   the boot first DROPS the model files' page cache.** `pin_depends_on_reclaim(free_now, bytes,
   reserve)` is `free_now < bytes + reserve` (`expert_source.cpp:100-102`); when true on Linux the
   boot calls `drop_file_cache()` (`:3511-3513`), which is `madvise(MADV_DONTNEED)` on the engine's
   own shard mappings + `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on the shard fds — the
   fadvise form drops the WHOLE file's cache (`:3474-3484`). Guarded by `STRATA_PIN_GUARD` (default 1,
   `:3503-3506`); the guard must stay: #1250's host OOM-kill is the documented failure it prevents
   (comment `:3490-3495`). Log signature when it fires: `FileExpertSource: 39.34 GiB really free for a
   39.19 GiB page-locked complement; the model files' cached pages were released: 89.12 GiB free`
   (`custom-iq3_s.log:407776`, and five more boots in this log).
3. **Why every battery boot triggers it, and why a lone boot may not.** The drop fires only when
   MemFree at pin time is below complement+reserve. Boot N's first requests pull the ~46-60 GiB of
   shards through the page cache (see 4); at boot N+1 that cache is exactly what makes MemFree short →
   the drop fires → boot N+1 re-reads them. Today's 14:30 boot (a restart after the box had free RAM)
   did NOT drop — no "really free" line between `:426654-426661` — and its first request read only
   297.1 MB from storage with 60,683.4 MB already page-cached (`:426715`). The cold-start gap is
   therefore a BACK-TO-BACK-BOOT phenomenon, which is precisely the five-boot-mean measurement shape.
4. **What the drop costs the first leg.** With the cache released, the first prefill routes the lanes'
   expert pairs through the cold mapping: the T17B boots measured `file tier I/O this request: the OS
   read 44,514-54,803 MB from storage (7-109 major faults)` inside request 1-2 (T17B §1.1, log lines
   quoted there), hit rate ramping 88.1→95.9→97.8→98.6→99.0→99.1% over requests 1-10, +0.67 ms/window
   of CPU-experts pool term persisting the whole cold leg (T17B §1.3), plus ~6 lazy batch-graph
   captures and ~3.4k adaptive-tier swaps landing in the first lane (T17B §1.4). Leg-level: cold 267.2
   vs warm 301.6 rows/s in-window (+12.9%), wall 272.0→299.5 (+10.1%).
5. **The learned profile is written but never read back.** Config boots `--expert-profile
   data/expert-profile.bin` (static, Oct-14) in ALL standing configs (`custom-iq3_s-8081.json:49-50`,
   `campaign/cohorts-iq3_s-8081.json:49-50`, `custom-iq3s-262k.json:49-50`, `custom-iq3s-1m.json:53-54`)
   while saving the learned one every 30 MINUTES and at QUIT
   (`--expert-profile-save-every` is minutes, generate.cpp:1054-1056, 2775-2777; writer
   `generate.cpp:10852-10870`, timer `:12301`; `data/expert-profile-learned.bin` mtime Oct 21 15:37).
   The intended round-trip (generate.cpp:661 "as a profile `--expert-profile` reads") was T17B lever 2.

### a.2 The fast-warm package — RE-SPECIFIED (one lever is dead)

- **T18 already ran the learned-profile arm and REJECTED it** (SVG-CUSTOM-TODO.md:1332-1334): battery
  M1 +4.9, confirm M1 +1.6 / pair +5.0 / **conc −2.0**; MMLU 92.9+94.3 = 93.6 mean, one run under the
  93.3 floor and −2.1 pts vs the standing 95.7 — "REJECT (routing-changing quality cost not justified)".
  The T17B +4-6% five-boot-mean band was priced for prewarm+learned-profile COMBINED; the learned lever
  is now KILLED under the quality law and its conc sign was negative anyway.
- **The surviving package is lever 1 + lever 3, and both are in ONE staged artifact:**
  `campaign/prewarm2.sh` (T17B v2, staged, NEVER RUN — AUDIT-2026-10-18.md:263 "instruments staged:
  ... prewarm2.sh ... none run"). It (1) asserts the pin completed (greps the complement-ready line),
  (2) re-populates the page cache over pack+shards (dd pass), (3) replays 2 shorts + one ~11k-token
  prefill + a 6-way concurrent burst — the burst forces the lazy cohort batch-graph captures
  (T17B lever 3) and warms pools/adapt tier. Its prompts are deliberately disjoint from probe-lane
  constants so no prefix reuse contaminates the probe.
- **Change required: NONE in code/config.** The package is operational: boot the standing launcher,
  wait for the complement-ready line, run `campaign/prewarm2.sh <log>`, then probe. (Optional hygiene,
  not required for the falsifier: append the prewarm2 call to the battery boot sequence so all five
  legs get it identically.)
- **Expected gain (estimate, marked):** mechanisms addressed = 1 (storage stall + readahead pressure,
  largest), 2 (pool-term inflation, +4.7% of window on the cold leg), 3 (lazy captures, via the burst).
  Not addressed: the first-request hit-rate ramp that the learned profile would have raised (dead
  lever). Re-priced band: **~+3-5% on five-boot means, ~+8-13% on any first leg** (T17B's +4-6%/+10-13%
  minus the learned-profile share; no measurement re-priced it after T18 — this is an estimate).
- **Falsifier (one boot pair, unchanged in shape):** cold = standing launcher, straight to probe;
  warm = same launcher + prewarm2 before the probe. Read three things: (i) first
  `file tier I/O this request` line ≈ 0 MB storage read (today's warm-boot reference:
  297.1 MB, `custom-iq3_s.log:426715`); (ii) first-leg batch-summary CPU-experts term ≤ ~2.4 ms/window
  (if it still reads >2.4, the pool-term cause is NOT cache refill — T17B §1.3's falsifier); (iii)
  `captured the batch window` lines appear before the probe, not during it.

### a.3 Do NOT do (re-confirmed this session)

- `STRATA_PIN_GUARD=0` (skip the drop): #1250 OOM-kill class, documented at `expert_source.cpp:3490-3495`.
- Guard=2 (paced steps): kswapd reclaims the same pages — T17B already priced "no expected win".
- Booting `--expert-profile data/expert-profile-learned.bin`: T18 KILLED (quality).
- Eviction-side `POSIX_FADV_DONTNEED` on drops: `expert_source.cpp:5868-5871` comment — it made every
  eviction a re-read (~200 major faults); the tree already does MADV_DONTNEED-only there.

---

## (b) TIERING: OUR WALK AND THE BASALT DELTA TABLE

### b.1 What we run (verified at today's boot, `custom-iq3_s.log:426611-426715`)

- **Tier 0 — VRAM:** ExpertCache 9,390 slots / 17.81 GiB (`:426648`, `:426649` class; auto sizing print
  `:426642`: "19.91 GiB free, 2000 MiB reserved (+141 MiB for the draft head) -> 7184 slots" is the
  pre-fill basis; the resident-experts mode pre-fills to the full 9,390 from the profile in 2.6 s at
  7,267 MB/s). Policy: GPU computes cached experts; routing differs from CPU rounding (boot warning
  lines `:426650-426653`).
- **Tier 1 — pinned RAM:** the 38.70 GiB mapped page-locked complement of everything not in the GPU
  cache (`:426661-426663`). Misses are DMA reads of the mapped complement (device alias,
  `expert_source.cpp:3528-3536`) — "read by the GPU over PCIe" 0.6-3.2% of routed in recent live
  lines (`custom-iq3_s.log:427522-427581`). Adaptive tier swaps VRAM↔RAM under decay 0.85
  (`--adapt-decay 0.85` in config; swap machinery generate.cpp:427-432+; ADAPT_LAG=2, PINCAP 48,
  MIN_AGE 2, DORMANT 64 in the launcher).
- **Tier 2 — file/page cache:** the GGUF shards mapped read-only, STRATA_COLD_OVERLAP=4
  (page-cached experts computed from the mapping), io prefetch with whole-blob preads on 8 I/O
  threads, and the decode read-ahead gate (`custom-iq3_s.log:426642` class line). The gate: EWMA
  0.8/0.2 of the share of window demand bytes the page cache held (`expert_source.cpp:1670-1684`),
  capacity check every 16 windows vs cgroup/MemAvailable room (`:1686-1747`), usefulness rule
  (<0.10 used share → close; probe every 256 windows, `:1748-1771`). Steady state today: "cache room
  2.23 of the experts ... page cache held 0% of the decode's reads (EWMA), closed ... 0.0 MB of expert
  reads" (`:427489+`) — correctly closed: the RAM tier holds what decode would read; an 18 GiB RAM-tier
  experiment measured read-ahead at −9..−16% decode (comment `:1749-1760`).
- **Copy engines:** `--pcie-frac 0.35 --pcie-mode dma` + `STRATA_DMA_BATCH=1` (84.9% fewer DMA submit
  cycles, T13B) for the miss path; complement copy at boot (8 s warm). Drive self-probe once per
  process picks O_DIRECT vs cached per drive (`file_tier_probe`, `expert_source.cpp:515-549`, #3237).

### b.2 Delta table vs basalt (real deltas only; coalescing/ForesightSwap/pool/cold-overlap-alternatives
excluded per T25-3 §8's negative ledger)

| # | mechanism | ours (anchor) | basalt (anchor) | verdict for us | price / falsifier |
|---|---|---|---|---|---|
| 1 | Elastic KV under batch slots (per-session 32k backing + take-back + give-back) | REFUSED with batch: `--kv-grow` requires `o.batch == 0` (generate.cpp:5198-5201 per T25-3 §2.2); we run fixed `--kv-resident 32768` | default-on: `kKvElasticFloor` generate.cpp:2963-2967; running.md:19-26; commits 8ded6974/7388717e/c9d1705b | the enabling mechanism for parallel-8 (T17B cand 1, +15-25%) without shrinking the cache; standalone vs kv-resident ~+0-2% | M-L; zero-code falsifier FIRST: log per-request max KV depth on the conc probe (T25-3 #2) — if no session exceeds ~32-48k cells, kv-resident is already free and only the parallel-8 funding question remains |
| 2 | Prompt-read-beside-windows, always-on + window-stream priority | machinery in-tree (`STRATA_PIPED_CONCURRENT`, generate.cpp:13812-13848) but opt-in and structurally refused by our armed `--pcie-mode dma` | default-on (commit 81ab8aef) + stream priority (95f3f535; generate.cpp:5190, verify.hpp:118) | NEW, one boot: `STRATA_PIPED_CONCURRENT=1` with `--pcie-mode kernel` (or DMA_BATCH=0) — the honest A/B is dma+no-overlap vs kernel+overlap (DMA_BATCH's 84.9% submit saving is the counterweight) | T17B cand-4 class +6-8% conc (admissions = 14% of probe wall); refusal reason prints if any exclusion binds |
| 3 | Tier responsiveness (`--adapt-every 1` at decay 0.85) | every-4 default (config sets decay only) | their tuning note (5090, 400 W): every-1 = +3.6% prose after topic switch, +4.2% conversation, neutral structured; ≥0.97 decay −3-6% | NEW, S, one 3v3 boot; ~0 expected on our fixed probe set, the value is topic-switchy traffic | +2-4% band on switchy traffic |
| 4 | Read-ahead/EWMA decode gate | HAVE (expert_source.cpp:1670-1771), correctly closed at our operating point | not found in their tree (their tier is pack-profile-ordered; no EWMA gate in what T25-3 read) | KEEP — ours ahead; do not import anything here | — |
| 5 | Copy-engine shape | DMA_BATCH + mapped pinned complement (CE reads RAM tier directly) + drive probe | pinned ring on a copy stream for prefill streaming (prefill.hpp:8); reads-beside priority (see #2) | our DMA_BATCH banked; the only actionable residue is #2's stream priority | see #2 |
| 6 | Placement profile packaging | boot reads STATIC profile; learned file written every 30 min, never read (see a.1.5) | pack-embedded profile in `.basalt` (model-file.md:13,31); maya writes per-answer (`SETTINGS.md:439-447`) | AMENDED: booting the learned file is DEAD (T18 quality). The live residue is pack-build-time: bake the STATIC profile into the re-quant assembler (T23 portstudy #8) — placement-only, not routing-adaptive | revisit at the re-quant pack build; A/B static-profile-shipped vs auto-size |
| 7 | Memory floors / VRAM-aware admission degradation | none (we size at boot; fail a request when VRAM short) | basalt running.md:74-91 per-request MemAvailable guard; Xolotl calibrated need(B) + half-width retry (V41 13c) | NEW, robustness-only (no scoreboard gain); spec to file for ctxoff/1M lanes and foreign-VRAM days | no config-only falsifier in our tree (T25-3 #4) |
| 8 | Tier-decay default | 0.85 armed (config) | their 0.93 single-card substitution (generate.cpp:5721-5723) | MOOT (we arm 0.85; T25-3 contradiction 1) | — |

Nothing else in the family tiering beats what we run: our EWMA gate, drive probe, DMA-batch and
conversation cache (49 GiB, LCP across parked+active) are equal or ahead (T25-3 §0.1, §3.1).

---

## (c) HOST-SIDE: WHAT IS LEFT IN THE LOOP

1. **Admissions on the main thread between windows — the only measurable line.** 1.6 s of an 11.56 s
   probe wall (14%); ~36 slot-takes × 32-58 ms restore copies (T17B §0.5, §2.2 #4). Hiding half is
   priced **+6-8% conc**, L-cost (side thread/stream + the existing refuse-swaps-during-reads guard,
   generate.cpp:10511 per T17B). Zero-code first step: sum `slot takes ... (copied in X ms)` in the
   results files. CONFIRMED as the top host item.
2. **Per-round host serials:** ~2-2.6 ms of ~28 ms round (staging both cohorts + both tails) + adapt
   join 0.25 ms/window (T17B §2.1) — candidate 5 (+2-4%, pipelining + tail reorder). CONFIRMED,
   already priced; unchanged.
3. **Python serve loop: NOT a cost.** Per batched token: one `queue.get`, `int(line.split()[2])`,
   list append, dict update, generator yield (serve/server.py:2003-2012); the pump thread does a
   startswith chain per line (server.py:1158-1187). At 6 streams / ~300 rows/s the line volume is
   ~10^3/s of µs-scale work against 28 ms windows — unmeasurable. The 10 s heartbeat and 0.5 s
   queue-timeout paths (server.py:1455-1464) only fire on silence. KILLED (no action).
4. **aux-cpus / host-core: armed but OFF by default, cheap to try.** Today: 15 pool workers on logical
   CPUs 1-15, host thread on CPU 0 "(draining too)" (`--host-core first`), and "strata aux cpus: threads
   that are neither pool workers nor the host go to CPUs 16 (off at start; a request's aux_cpus=1 turns
   it on)" (`custom-iq3s-cohorts.log:17427-17428`; per-request key path serve/server.py:1404-1409;
   wiring generate.cpp:5926-5943). F-201-zen5 prices the CCD-split class at ~400-800 ns per expert-call
   of fabric transit avoided — on our 2.0-2.3% miss tail this is a ≤0.1 ms/window-class item
   (ESTIMATE). NEW, S: one aux_cpus=1 diagnostic leg; no scoreboard expectation.
5. **Profile-save fsync: obsolete concern.** F-201-nvme #1 priced an "every-30-rounds" save as a
   0.5-2 ms serialization point; the flag is MINUTES (generate.cpp:1054-1056) and the config sets 30 →
   the write happens between requests at most twice an hour (writer runs in the serve loop between
   requests, generate.cpp:10852-10870). Per-request cost ≈ 0; only if the cadence is ever lowered does
   the F-201-nvme sidecar/async-writer recipe matter. KILLED at current cadence.
6. **F-201-zen5 engine items stand, unexecuted, as previously priced** (hugepage+re-register the ~39 GiB
   pool; strip LOCK/MFENCE from admission/dispatch; graded backoff; fence consolidation after an rdtsc
   measurement): staging path +5-15% class and fence host-component deletion. Not re-verified this
   session; ordered after admissions work. CARRY.

---

## (d) HARDWARE-LOCK WATCH: consumer Blackwell and the fp32-accum rate

**What is measured (ours):** F-252 (SVG-CUSTOM.md:8550-8572; `campaign/mma-accum-ratelock.cu`, sm_120,
pure-register mma.sync m16n8k16, no memory): fp16-accum 476 TFLOPS vs fp32-accum 238 (fp16-in and
bf16-in both 238), ratio 1.99/1.97 stable across ABAB legs on the live box.

**What is documented (NVIDIA, primary):** the RTX Blackwell whitepaper lists "Peak BF16 Tensor TFLOPS
with FP32 Accumulate: 209.5/419 (dense/sparse)" for the RTX 5090 — the same 2:1 segmentation NVIDIA
documented for GA102 (3090: 142 TFLOPS FP16-acc-FP16 vs 71 FP16-acc-FP32, GA102 whitepaper). So the
half-rate is SPEC on consumer Blackwell, exactly mirroring Ampere consumer — our measurement lands on
the documented ratio at observed clocks. Server Blackwell does not take the penalty at the same rate
(B200-class), and one NVIDIA-dev-forums report claims RTX PRO 4000 Blackwell does NOT halve fp16/bf16
with fp32-accum (UNVERIFIED forum claim) — the single public fact suggesting the consumer rate is a
SKU configuration rather than a fixed clock-domain.

**What the unlock actually is (repo read, this session):** github.com/Zagitta/gpu-unlock is a Linux
kernel module run pre-driver from the initramfs: it combines the CMP unlocker's "V67/PLM" payload with
the UEFI approach's timing, writes SPEED SELECTORS (SS0 = 0x88888888 — per the README the selector for
the FULL BF16 rate; SS1 = 0x8, effect untested), then performs a Function Level Reset so the stock
driver re-binds. Chip gate is PCI-ID: GA102/GA104 GeForce+CMP ONLY; professional A-series, datacenter,
and ALL OTHER CHIPS are explicitly rejected; tested only on an RTX 3090; requires a cold card (power
cycle, not warm reboot). The README documents no fp32-accum unlock and no measured TFLOPS table; the
fp32-accum attribution comes from the Fateev thread (F-252's source item), whose own Strata numbers
price the unlock at **+10-12% prefill, decode FLAT** on the 3090. No Blackwell/sm_120/RTX-50 mention
anywhere in the repo.

**Verdict:** The lock BEHAVIOR exists on our card (measured) and is whitepaper-spec, not an accident.
The unlock TECHNIQUE is Ampere-chip-specific: its payload and selector semantics come from pinned CMP
reference material (GA102-class mining cards) that has no Blackwell counterpart, and the module
rejects our chip by ID. **No Blackwell unlocker exists.**

**What would need to exist for a Blackwell unlock (each independently absent today):**
1. Discovery of the GB202 speed-selector/PLM interface (the equivalent of SS0/SS1) — not public;
   Ampere's came from CMP firmware reference material.
2. A pre-driver payload for GB202 (the V67 image is GA102 firmware); the pre-driver FLR flow itself
   (initramfs, cold card) is chip-agnostic and would port.
3. Evidence that consumer-GB202's fp32-accum rate is selector-controlled rather than fused — the
   RTX PRO report (if verified) would be that evidence; nobody has verified it on GB202, and a fused
   hard-segmentation would make any selector write a no-op.
4. Someone doing it — no project, issue, or commit announces one.

**Impact if one ever appeared:** prefill-class only (+10-12% family-measured on the 3090; our prefill
WMMA waves and any TC-hc mma plan are the fp32-accum consumers). Decode is int8-expert/CUDA-core —
untouched (F-252 consequence 3, consistent with the 3090 report). The TC-hc plan stays priced at 238
TFLOPS; the 476 fp16-accum pipe breaks bit-identity at K=2560 and needs the KL yardstick before it is
even a candidate (F-252 consequence 2). WATCH-ONLY.

---

## (e) VRAM BUDGET: THE +792 SLOTS AND 262k CONTEXT

**The re-quant yield (REQUANT SPEC, SVG-CUSTOM-TODO.md:1302-1315):** recut 12 cold layers to
(IQ2_S, Q2_0), copy the rest: experts 46.84 → 42.97 GiB (−8.3%), slots 9,337 → ~10,129 (**+792**),
corrected pricing +2-4% M1 / +3-6% conc (slot term saturated at 97.7% hits).

**Where it should go — expert cache, by arithmetic:**
- 262k switch at batch 6 costs +224 MiB/slot-session (+1.34 GiB total, T15B §1.3, observed 0.73→0.95
  GiB/session), i.e. **−~700-725 slots** at 1.897 MiB/slot (T15B's band: 9,337 → ~8,630, −7.5%).
  Today's base is 9,390 slots / 17.81 GiB (`custom-iq3_s.log:426648-426649`), so the un-mitigated move
  is ~8,665 slots ≈ −725.
- With the re-quant: ~10,129 − ~724 ≈ **9,405 slots at 262k** (the TODO's 10,129 × 0.925 = 9,369 ≥
  9,337 — the same conclusion on the older base): the +792 FULLY ABSORBS the 262k slot risk and leaves
  the cache at-or-above today's level. F-183 slope (±513 slots = ±4.9% conc / ∓6.9% M1, T17B table #1)
  says the un-absorbed −725 would be a −7%-class conc exposure; absorbed, it is zero.
- **KV gets nothing:** kv-resident is a depth lever, not a slot source (F-22b law; the 20,480 squeeze
  was a prefill regression at 90-145k depth); the 65,536 variant costs ~470 slots
  (SVG-CUSTOM.md:243-244) for a term the conc probe may never exercise (T25-3's open question 1:
  instrument KV depth first). The 262k streaming tail grows in PINNED HOST RAM, not VRAM: 1.55 →
  3.09 GiB (12,684 B/cell slope, T15B §1.4), VRAM pool unchanged (kv_plan mode 1, layer.cpp:633-654).
- **Draft head gets nothing:** +141 MiB in the reserve line (`custom-iq3_s.log` auto-sizing print);
  slot drafters are 51 MiB private K/V each, already inside the session math. Depth-independent.

Net: re-quant slots → expert cache; 262k's extra KV lives in host RAM; draft head is noise. The
launcher trap re-issued: `launch-custom-iq3s-262k.sh` is STALE (T15B §2) — missing 10 standing arms,
carrying ROUTE_RESIDENT=0.5 + EXCHANGE_ROTATE=1; refresh from launch-custom-iq3s-8081.sh before it
ever boots.

---

## (f) PLAN IMPACT VERDICTS

| item | verdict | exact content |
|---|---|---|
| Fast-warm package | **AMENDED** | prewarm2.sh post-boot (page-cache re-warm + replay incl. capture-forcing burst) = the whole surviving package; the learned-profile boot lever is KILLED (T18: MMLU 93.6 < 93.3 floor, conc −2.0). Expected +3-5% five-boot means / +8-13% first leg (ESTIMATE, re-priced after removing the dead lever). Falsifier: ONE cold-vs-warm boot pair; read first `file tier I/O` line ≈ 0 MB, pool term ≤ 2.4 ms/window, captures before probe. Zero code, zero config. |
| Tiering changes | **CONFIRMED** (as the T25-3 board; nothing added, nothing killed by this walk) | order: (1) KV-depth instrumentation (zero code, prices elastic-KV), (2) `STRATA_PIPED_CONCURRENT=1` + `--pcie-mode kernel` boot (+6-8% class), (3) `--adapt-every 1` boot (+2-4% switchy only). EWMA gate, drive probe, DMA-batch, conversation cache: KEEP — ahead of or equal to basalt. |
| Host-side items | **CONFIRMED (admissions) / NEW (aux-cpus) / KILLED (Python loop, profile-save fsync)** | admissions overlap = the only measurable host line (+6-8% class, L; zero-code read first). aux_cpus=1 diagnostic leg (S, ≤0.1 ms/window est.). Python serve loop unmeasurable. Profile save now 30-minute cadence — fsync concern obsolete. F-201-zen5 pool/fence items CARRY as priced. |
| gpu-unlock watch | **CONFIRMED as watch / KILLED as action** | half-rate is measured AND whitepaper-spec on consumer Blackwell; module is GA102/GA104-PCI-ID-only, technique (V67/PLM payload + SS0 selector + FLR) is Ampere-firmware-specific; no Blackwell unlocker exists; even if one appeared the value is +10-12% prefill class, decode flat. Re-check triggers: any Blackwell speed-selector publication, any verified RTX PRO fp32-accum parity report, or a Blackwell unlocker repo. |
| 262k VRAM plan | **CONFIRMED** | re-quant's +792 slots → expert cache; fully absorbs the −~724-slot 262k move (≈9,405 ≥ 9,390 current), zero F-183 exposure; KV stays resident-32768 with the deep tail in pinned host RAM (+1.54 GiB host, not VRAM); draft head is a 141 MiB non-item. Refresh the stale 262k launcher before any boot. |

---

## (g) UNKNOWNS, WITH CONFIDENCE

1. **Post-prewarm2 pool-term residual** (if the CPU-experts term stays >2.4 ms/window with zero
   storage read): CPU ramp vs fresh-arena page work — confidence HIGH that one A/B boot decides it
   (T17B §1.3 falsifier); LOW on which cause wins.
2. **Prewarm2-only mean gain** (+3-5% is an estimate; the original +4-6% was never split across
   levers): MEDIUM-LOW until the boot pair runs. The T18 confirm data (learned-profile conc −2.0)
   supports the direction of the amendment, not the size.
3. **KV-depth distribution on the conc probe** (decides whether elastic-KV matters beyond parallel-8
   funding): UNKNOWN, HIGH confidence the zero-code instrumentation answers it.
4. **RTX PRO Blackwell fp32-accum parity** (the forum claim): UNVERIFIED — LOW confidence in the
   claim itself; it is the only public hint that a selector-class knob exists on Blackwell.
5. **Whether the complement's 38.70 GiB includes stage-lend/padding beyond the strict pair
   complement** (17.81 GiB GPU + 38.70 GiB RAM exceeds the 46.84 GiB pack figure): code shows
   `append_stage_lend_regions` inside the complement plan (`expert_source.cpp:3449-3452`), so likely
   yes — MEDIUM confidence, cosmetic for this dossier, worth one sizing read before anyone
   re-derives tier budgets from the two numbers.
6. **Clock-attribution of F-252's absolutes** (476/238 measured under a live 96%-util GPU): the RATIO
   is the instrument and is stable; absolutes vs spec (419/209.5 dense at boost) differ by clocks —
   HIGH confidence in the law, no weight on the absolute TFLOPS.

## SOURCES (all read this session unless marked)

- Our tree: `src/core/expert_source.cpp` (:100-102, :3474-3524, :3528-3536, :515-549, :1664-1771,
  :5868-5884); `src/program/generate.cpp` (:427-432, :5926-5943, :661-665, :1054-1056, :10852-10870,
  :12301); `src/core/weights.cpp:518-580` (STRATA_HC_REQ8 load_hc_q8); `serve/server.py`
  (:1130-1152, :1154-1194, :1404-1409, :1444-1505, :1985-2101); `custom-iq3_s-8081.json`,
  `campaign/cohorts-iq3_s-8081.json`, `custom-iq3s-262k.json`, `custom-iq3s-1m.json` (:49-53 each);
  `launch-custom-iq3s-8081.sh`; `campaign/prewarm.sh`, `campaign/prewarm2.sh`;
  `custom-iq3_s.log:426611-427600` (today's boot: pin, warm first request, gate lines, hit rates),
  `:407776-411428` (drop-fired boots); `custom-iq3s-cohorts.log:17427-17428`.
- Campaign docs: SVG-CUSTOM.md F-246-F-252 (esp. F-248 scoreboard, F-252 :8550-8572);
  SVG-CUSTOM-TODO.md T18 (:1323-1339) + REQUANT SPEC (:1302-1315); T17B-conc-leap.md §1-§2;
  T25-3-serving-scheduling.md (whole); F-201-{nvme,pcie,zen5}-agent.md; F-200-moe-offload-agent.md;
  T23-basalt-portstudy.md; T15B-262k-switch-plan.md (§1.2-1.4, §2); AUDIT-2026-10-18.md:263.
- basalt @ fb0944c7 (`/home/svg/LLM/data/references/basalt`): `include/basalt/prefill/prefill.hpp:8`;
  docs/running.md:17-26, 37-38, 64-65 (read spot-checks; the line-level anchors are T25-3's, read at
  the same pin).
- External (hardware-lock): github.com/Zagitta/gpu-unlock README (fetched, this session); NVIDIA RTX
  Blackwell GPU Architecture whitepaper (209.5/419 BF16-fp32acc figure via search result surfaces:
  whitepaper PDF, HN thread 44997957, bestgpusforai/TechPowerUp corroboration); NVIDIA dev-forums
  RTX PRO 4000 Blackwell fp32-accum thread (UNVERIFIED claim, cited as such); GA102 whitepaper (71/142
  spec figure). The Fateev thread itself was NOT reachable this session — its content is carried via
  F-252's own record, marked accordingly.
