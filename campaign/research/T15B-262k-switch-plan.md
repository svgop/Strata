# T15B — Switch the standing context window 131,072 → 262,144: trace, switch plan, test plan, risks

**Date:** 2026-10-16 (session). Repo state: `main` @ 34d3e4bc, working tree clean except `data/expert-profile-learned.bin`.
**Mode:** read-only scoping. No build, no boot, no config writes.
**Question (operator):** what exactly does `--max-context 262144` change on the standing `custom-iq3_s-8081.json` recipe, what is the switch diff, what decides "no noticeable difference" vs the 131k five-boot means, and what can break.

---

## Summary

1. **The switch is a one-line config diff and almost everything about it is already measured.** `custom-iq3s-262k.json` already exists in the repo root and differs from `custom-iq3_s-8081.json` in exactly `--max-context` (131072 → 262144) plus the log path. F-181/F-182 (2026-10-14) already booted and served this exact config on this tree, and A/B'd it against 131k at short usage: M1 −0.2%, pair −0.8%, 6-conc −1.3% — statistical zero.
2. **The QSA top-k capacity fix (upstream c7544ef6+4a9669ff) is NOT needed — verdict: ALREADY-THERE by different construction.** Our tree never carried eddo's static-table `block_topk_kernel` (the 262,144-cell bug class). Our selection is a 4-pass radix over pooled blocks plus an sm_90+ cluster variant whose shared memory is already dynamic, sized by capacity, with a per-device opt-in and graceful fallback. Parity suites in-tree cover capacity 262,144 and 524,288; the 1M-context boot (Oct 14, same kernel file that is in the tree today) is the existence proof at 8× the target capacity.
3. **The only real fixed costs at 262k (batch 6):** slot sessions +~224 MiB each (+1.34 GiB total), which costs the expert cache **~700 slots (9,337 → ~8,630, −7.5%)**, and pinned host K/V +1.54 GiB (1.55 → 3.09 GiB). `--kv-resident` stays 32,768 (streaming grows 98k → 229k cells/layer). Conversation cache: unchanged (4 slots, depth-independent 240.7 MiB snapshots — byte-identical measured at both boots). Prefill chunking unchanged.
4. **kvg_ensure is OFF in the standing config and stays OFF** (`--kv-grow` refuses `kv-resident > 0` and `batch > 0`): no thrash risk, no env deltas. The T10 elastic-EXTENDED K/V port (78652d23+e6ed2a50) is irrelevant to this switch; it only matters in `--kv-grow` mode.
5. **The measurable costs:** short-usage batteries ≈ noise (F-182); depth costs decode ~−11% at 244k depth (F-181, once-per-window O(depth) selection/scoring term). The one number NOT yet measured on the current F-233 recipe is the 3v3 at 262k — the conc lane is the sensitive one, because −700 slots moves in the F-183 direction (fewer slots → conc down, M1 flat-to-up).
6. **Biggest procedural trap:** `launch-custom-iq3s-262k.sh` is stale (Oct 14 18:17) — it is missing 10 standing env arms (GDN_CHUNKED, FUSE_HITS, POOL_DIET, COHORTS=2, TYPICAL_DEVICE, VERIFY_ASYNC_LAUNCH, SPEC_DRAFT_TEMP=0.85, ATTN_DAILY, …) and carries `ROUTE_RESIDENT=0.5` + `EXCHANGE_ROTATE=1`. A manual boot through it would not be the standing recipe. Batteries are safe: `battery-config.sh` sources the 8081 launcher's export block (F-214 lesson).

---

## 1. What `--max-context 262144` changes in OUR tree (the trace)

### 1.1 Cells, the indexer budget, and the selection kernels

- **1 cell = 1 token** for the QSA layers: boot prints `KV streaming: 32768 of 262144 cells per QSA layer in VRAM` at `--max-context 262144` (custom-iq3_s-262k.log:5).
- Geometry (compiled-in, checked against GGUF): `idx_block = 4`, `idx_top_k = 2048` (include/strata/kernels/qsa.hpp:103,112). Selection width = `min(n_kv, idx_top_k + idx_block − 1)` = **min(n_kv, 2051)** (qsa.hpp:127-137) — the top-2048 cap holds **by construction at any depth**; 262k changes what the selection scans, never what it returns.
- Per call at capacity C: `max_blocks = C/4 + 2` (qsa_topk_parity.cpp:34) → **32,770 blocks at 131k, 65,538 at 262k** (2× the scan).
- The kernels (src/kernels/cuda/qsa_select.cu):
  - `qsa_block_topk` — 4-pass radix select over the query's blocks, keys in registers when blocks fit ("contexts up to ~135K cells"), else the same threads over shared-memory keys (qsa_select.cu tail of block_topk dispatch + header comment qsa_select.hpp:17-27). Its shared histograms are **capacity-independent** (`hist[TK_T/32][256]`, qsa_select.cu:604,703).
  - `qsa_block_topk_cluster` (sm_90+, S19) — dynamic shared memory = `ceil(max_blocks / CL_N) × 4 B` (qsa_select.cu:1240). At 262k: 65,538/8 = 8,173 keys × 4 B = **31.97 KiB < the 48 KiB default — no opt-in needed at 262k** (131k uses 8 KiB). Past ~786k cells it opts in per device and checks cluster residency; if that fails it **returns false and the one-CTA kernels take the call** (qsa_select.cu:1256-1263) — a capacity ceiling produces a slowdown, never a corruption.
  - `kTopkMaxCells = 32768` (qsa.hpp:282) bounds only the **legacy per-token single-block kernel** (`topk_512`), which errors out loudly if exceeded (qsa.cu:761-763); it is not the served path's limit.
- **Port-or-already-there verdict on c7544ef6+4a9669ff: ALREADY-THERE (not needed).** The bug c7544ef6 fixes is eddo's `block_topk_kernel` keeping `(row, warp)` tables sized for exactly 262,144 cells in 32 KiB of static shared memory. Our kernel family is a different algorithm with capacity-sized dynamic smem from the start. Evidence, three independent legs:
  1. Code read: no static capacity-sized tables anywhere in qsa_select.cu; the only capacity term is the dynamic `smem` above.
  2. In-tree parity: `qsa_topk_parity.cpp` cases include `{262144, 262144}` and `{300001, 524288}` (line 114-115); `qsa_topk_active_parity.cpp` opens "Capacity 262144, live prefill up to the register boundary, and the legacy fallbacks".
  3. Empirical: `custom-iq3_s-262k.log` (booted 2026-10-14 12:03, served to 16:25, deep prompts incl. 157k cold) and `custom-iq3_s-1m.log` (booted 12:06 with YaRN ×4, served at 937k depth — 23.6M streamed block reads in the tail). `qsa_select.cu` last changed in 8d1c9d64 at 2026-10-14 01:09 — **before both boots**, so the boots ran the byte-identical kernel file that is in the tree now (git log -- src/kernels/cuda/qsa_select.cu).
  - Consequence: c7544ef6+4a9669ff stay on the T10 list as **upstream-convergence items only** (their kernel, their limit); there is nothing to port for 262k. (If we ever want >1M on 3090-class smem, the cluster opt-in ceiling and the one-CTA fallback cost are where to look — not this switch.)

### 1.2 KV pool sizing and the kv-resident interplay

- `kv_plan` (src/core/layer.cpp:633-654): with `g_kv_resident = 32768 > 0` and main layers in streaming `mode = 1`, the **VRAM pool stays the resident window (32,768 cells/layer) at both 131k and 262k**; only the streamed (host) portion grows: 98,304 → 229,376 cells/layer. The drafter gets its own ring (`mode = 2`, "the drafter has no indexer") — unchanged.
- The resident minimum is 20,480 (`qsa_kv_resident_min`, layer.cpp:660). F-22b's law stands: **kv-resident is a depth lever, not a free slot source; 32768 stays** (SVG-CUSTOM.md:962-969; the 20480 squeeze was a prefill regression at 90-145k). The 65,536 twin-rig variant (SVG-CUSTOM.md:243-244, ~470 slots) is NOT part of this switch.
- What grows in VRAM per slot session is the **QSA state around the pool**: pooled indexer rows (`pooled_rows = max_cells/4 + 2` rows × 128 dim × 4 B = 128 B/cell), the streaming residency map, page table, and — primary state only — the rope tables (256 B/cell, `with_rope`, n_rot = 64, layer.cpp:669-687, layer.cpp:844). The per-session QSA decode buffers are sized by the selection cap (2051), not by max_context (layer.cpp:923) — unchanged.

### 1.3 VRAM arithmetic and the expected expert_slots delta (batch 6)

Anchors are the two real boots (same tree, same binary lineage):

| line (boot log) | 131k (custom-iq3_s.log, Oct 16, batch 6) | 262k (custom-iq3_s-262k.log, Oct 14, batch 4) |
|---|---|---|
| slot session size | 0.73 GiB each | 0.95 GiB each |
| sessions total ("the slot sessions take …") | 4.37 GiB (×6) | 3.79 GiB (×4) |
| expert cache auto print | 19.80 GiB free → 7,142 slots | 20.12 GiB free → 7,291 slots |
| expert cache actual | **9,337 slots / 17.71 GiB** | **9,529 slots / 18.08 GiB** |
| KV streaming | 32,768 of 131,072 cells; 1.55 GiB pinned | 32,768 of 262,144 cells; 3.09 GiB pinned |
| RAM complement (FileExpertSource) | — | 41.44 GiB |

Geometry of the per-session delta at +131,072 cells: rope 131,072 × 256 B = 32 MiB (primary only) + 12 states × (pooled 131,072 × 128 B = 16 MiB + maps) = 192 MiB → **+224 MiB/session**, matching the observed 0.95 − 0.73 = 0.22 GiB exactly (F-182's "slot-session arena 0.88 GiB going 131k→262k" = 4 × 0.22 at batch 4).

**Prediction for the switch boot (batch 6, standing recipe):**

| quantity | 131k standing | 262k predicted | basis |
|---|---|---|---|
| slot sessions total | 4.37 GiB | **~5.71 GiB** (+1.34) | 0.224 GiB × 6, observed per-session |
| expert cache actual | 9,337 slots (wobble: live /metrics read 9,454 once, T13A §4.1) | **~8,630 slots (band 8,550-8,700)** | −1.34 GiB ÷ 1.897 MiB/slot (17.71/9,337) |
| expert cache auto print | 7,142 | ~6,67x | same free-VRAM delta at the auto basis |
| KV resident window | 32,768 cells/layer | **32,768 — unchanged** | kv_plan mode 1 |
| KV pinned host | 1.55 GiB | **3.09 GiB** | measured at the 262k boot |
| RAM expert complement | ~40.1 GiB class | **~41.4-42.7 GiB** (+~1.3) | fewer GPU slots → bigger complement (262k boot measured 41.44) |
| draft head / slot drafters / L2 carve / verify windows | — | unchanged | depth-independent |

### 1.4 Host RAM

- Pinned K/V: linear in streamed cells. Fit over the three measured boots (1.55 / 3.09 / 12.38 GiB at 131k / 262k / 1M): slope ≈ **12,684 B/cell**, fixed ≈ 380 MiB — within 3% of the analytic 12 QSA layers × int8 K+V (4 KV heads × 128 dim × 1.0625 codes+scales × 2) = 13,056 B/cell. At 262k the streamed 229,376 cells/layer × 12 layers is the 3.09 GiB.
- Total host delta of the switch ≈ +1.54 (pinned KV) + ~1.3 (complement) ≈ **+2.9 GiB against MemAvailable 171.6 GiB** — a non-issue, consistent with F-181's 129/184 GB used at 937k depth with the 49 GiB cache absorbing streamed KV.

### 1.5 Prefill path

Unchanged by the switch: `--prefill auto:32768` prints the same `prompt chunk auto: 32768 tokens, a 96-slot ring` at both capacities; chunking, KV prefetch prefixes (12), and the borrow path are depth-independent in structure. What changes is only the depth the ladder can reach: standing plateau 6.0-7.0k t/s to 120k (five-boot); F-181 measured 6,534 t/s at 157k cold and 5,711 fresh t/s at 244k on the Oct-14 recipe — expect a mild decline past 131k, not a cliff.

### 1.6 Conversation cache at fixed 49,152 MiB

- Slot count is **4** — the default `--conversation-cache-slots` (generate.cpp:715); the config sets only the MiB budget. Unchanged by the switch.
- Snapshot cost is **depth-independent for the checkpoint itself**: `snapshot_bytes = 240,717,704` measured **byte-identical** at the 131k and 262k boots for the same 300-token parks (GDN recurrence + PLE history + indexer tails; the KV of parked prefixes lives in the arena/prompt cache, not duplicated per snapshot). The byte budget would only bind near 4 × (241 MiB + full-depth KV ≈ 3.3 GiB) ≈ 14 GiB ≪ 48 GiB.
- Eviction policy (conv_cache.hpp): pinned root + LRU among the rest, 4-slot cap — identical at both capacities; observed churn (evictions=0 steady-state at parked=2; superseded-copy drops) is probe-traffic, not depth.

### 1.7 kvg_ensure: off, and staying off

`kvg_ensure` (generate.cpp:6889-6981) is gated by `--kv-grow` / `STRATA_KV_GROW` **and** requires `kv_resident <= 0 && batch == 0 && !vram_elastic && no remote/peer` (generate.cpp:4484-4496). The standing config has kv-resident 32768 and batch 6 → the elastic K/V never engages, at 131k or 262k. **The heat-ordered D2D give-and-take and its 36-56 ms-class admission copies are out of scope for this switch**; the T10 elastic-EXTENDED port (78652d23+e6ed2a50, "+165 slots at 131k") is only relevant if we later build a `--kv-grow`-mode lane. `/metrics vram_elastic=0` stays 0 (that flag is `--vram-elastic`, the #533 segment cache, also unset).

---

## 2. The switch plan

**Config diff — exactly one json** (custom-iq3s-262k.json already in the repo root and verified: `diff` of the args arrays against custom-iq3_s-8081.json = one line):

```diff
--- custom-iq3_s-8081.json (args)
+++ custom-iq3s-262k.json (args)
@@ -12 @@
-  "131072",              # --max-context
+  "262144",
@@ (log key)
-  "custom-iq3_s.log"
+  "custom-iq3_s-262k.log"
```

Everything else byte-equal: kv int8, kv-resident 32768, prefill auto:32768, conversation-cache-mib 49152, vram-reserve-mib 2000, spec 6, mtp-q4 all, parallel 6, port 8081.

**Env deltas: none.** No new STRATA_* arms; nothing to set for capacity. (`STRATA_KV_GROW` stays unset on purpose.)

**Launcher discipline (the load-bearing rule):**
- For batteries: `bash campaign/battery-config.sh custom-iq3s-262k.json <label>` — it sources the 8081 launcher's full standing export block (F-214 lesson) and boots server.py directly (Launch Law).
- For manual/real-use boots: do NOT use `launch-custom-iq3s-262k.sh` as-is. It is stale (Oct 14 18:17): missing ATTN_DAILY, GDN_CHUNKED, FUSE_HITS, POOL_DIET, COHORTS=2, TYPICAL_DEVICE, VERIFY_ASYNC_LAUNCH, SPEC_DRAFT_TEMP=0.85; carries ROUTE_RESIDENT=0.5 (standing: 0.75) and EXCHANGE_ROTATE=1 (not in standing). Refresh its export block from launch-custom-iq3s-8081.sh (or have it exec the same env) before it becomes the daily driver.
- Launch Law: config "args" is rewritten by setup runs; launchers must start serve/server.py directly (both do).

**Expected boot differences (batch 6):** sessions line reads ~5.7 GiB; expert cache ~8,6xx slots / ~16.4 GiB; auto print ~6,6xx; `KV streaming: 32768 of 262144 cells … 3.09 GiB of pinned RAM`; RAM complement ~+1.3 GiB (copy-through takes ~1 s longer); no new warnings — the 262k boot log is the reference for what healthy looks like. Boot-time validation is only `max_context < 2^31` (generate.cpp:2741) and the prompt-fit check; no kernel-capacity rejection exists (and the 1M boot passed).

**Rollback:** one arg — set 131072 back (or re-run the 8081 launcher). No state to clean; the learned profile keeps updating in both configs.

---

## 3. The test plan (what decides "no noticeable difference")

Baselines to beat are the five-boot means on the F-233 recipe (SVG-CUSTOM.md:7998): **M1 198.5 (185.7-211.3) / pair 197.6 (194.7-202.9) / 6-conc 312.4 (300.2-322.7) / M3b 74.9% (72.2-76.0)**; prefill plateau 6.0-7.0k to 120k; warm long-ctx decode 226-243; MMLU-Pro 95.7/94.3.

1. **3v3 ABAB, the standing decider.** Six boots alternating `battery-config.sh custom-iq3_s-8081.json ctrl-rN` / `battery-config.sh custom-iq3s-262k.json ctx262-rN`, interleaved across the day (the interleave settles it; the F-182 drift lesson — both arms read ~177 at the wrong hour vs ~195 at 01:00). Compare M1/pair/conc/M3b per the no-regression law: the switch passes short-usage if the B arms sit inside the five-boot spreads; priors say −0.2/−0.8/−1.3% (F-182) with the wider −708-slot conc exposure (F-183 sensitivity below).
2. **Context ladder extended to 250k fresh prompts.** `ctx-probe.py 4096 16384 32768 65536 98304 122880 147456 163840 196608 229376 245760` on a fresh B boot (unique seeded prompts — no cache reuse). Pass: the 16k-120k plateau holds ≥6.0k t/s and the new rungs degrade smoothly (anchor: 6,534 @157k, 5,711 fresh @244k, F-181); no knee, no error, no OOM.
3. **Long-ctx decode + acceptance (M3b-at-depth).** `depth-lane.py --depth 240000 --decode-tok 1500` on B: warm decode expected in the ~175-215 t/s class (F-181 read 173.9 on the weaker Oct-14 recipe whose short decode was ~195 vs today's 226-243), acceptance ≥ ~75%. Plus M3b inside the 3v3 batteries (74.9% mean band) — this is the "deeper KV shifts logits" canary: any path that depended on depth would move acceptance first.
4. **Quality gate: 2× MMLU-Pro on B** (band floor 91; five-boot 95.7/94.3). Warranted even though the switch is not routing-changing in code: −700 slots changes the GPU-vs-CPU expert-rounding mix (the boot's own rounding warning), the F-72/F-200 precedent for cache-residency-changing arms.
5. **Boot-log watchlist on every B boot:** the four predicted lines (§2); `STRATA_ADAPT_OVERLAP` refusal line still expected (DMA_BATCH on); `/metrics` vram_elastic=0, expert_slots in the 8,5xx-8,7xx band; no `qsa: topk_512: n_kv > kTopkMaxCells` (that would mean a legacy-path regression — not expected at any depth; the block kernels serve).

Decision rule: adopt 262k as standing if (1) passes in-band and (4) clears the floor; the ladder/depth rungs (2, 3) characterize the deep lane rather than gate it — their numbers become the 262k scoreboard entry.

---

## 4. Risks (go/no-go)

| # | Risk | Class | Verdict |
|---|---|---|---|
| 1 | **Indexer top-k capacity bug at 262,144 cells** (the c7544ef6 class: static smem tables sized to 262,144) | correctness | **CLEARED.** Our kernel has no static capacity tables; cluster smem is dynamic (31.97 KiB @262k, under the 48 KiB default); overflow falls back, never corrupts; parity covers 262,144 and 524,288; the 1M boot ran the identical kernel file. |
| 2 | **kvg_ensure thrash on admission** (36-56 ms copies growing with depth) | latency | **MOOT.** Elastic K/V refuses kv-resident>0/batch>0 — off in both configs; no copies, no grows, no trims. |
| 3 | **Conversation-cache eviction pressure at 4 slots × deeper convs** | latency/quality | **NON-ISSUE.** Slot count 4 fixed; snapshots depth-independent (240,717,704 B byte-identical across both boots); byte budget binds only ~14 GiB of 48 GiB at full-depth parks; policy unchanged. |
| 4 | **QSA decode growth if selection scans more cells** — decode was flat to 120k; does it hold to 262k? | measured cost, not a bug | The flat law stands **because** the depth-growing terms are once-per-window, not per-token (T13A §4.4). The per-window O(n_bid) scores+topk term doubles 131k→262k; net measured effect at 244k depth: **−11% decode** (173.9 vs the ~195 class, F-181). Short usage unaffected (F-182). This is the price of the deeper lane, paid only at depth. |
| 5 | **Expert-cache slot loss ~700 → conc lane** — the one mechanism that moves the standing scoreboard at short usage | the real go/no-go | −1.34 GiB sessions → −708 slots. F-183 measured +513 slots → 6-conc +4.9% / M1 −6.9% (halving when stacked); F-182's batch-4 A/B (−460 slots) read only conc −1.3%. Net prior: conc −1..−4%, M1 flat-to-slightly-up. The 3v3 decides; a breach is the known concurrency-vs-cache trade (F-183's vram-800 shelf arm is the labeled counterweight, one cp away), not a defect. |
| 6 | **Stale 262k launcher** (env drift) | procedural | Real and silent: a manual boot through it runs 10 arms short and would confound any A/B. Mitigation in §2 (battery-config.sh sources standing exports; refresh the launcher before daily-driver use). |
| 7 | Prefill/KV streaming at 244k+ on the current recipe | unmeasured on THIS recipe | Ladder rungs 147k-245k exist in no current-recipe log (F-181 was Oct-14 recipe). Characterize, don't gate. |

**GO, conditional on the 3v3 conc reading and the MMLU floor.** Nothing in the trace supports a no-go on correctness or stability grounds; the config has already booted, served deep, and A/B'd clean on this tree.

---

## Contradictions and their resolution

- **T10 framing ("CHECK whether our tree already has this fix") vs the branch facts.** c7544ef6/4a9669ff/78652d23/e6ed2a50 are on remote `eddo/custom` (921 commits past merge-base 296b59d2), NOT ancestors of our HEAD (104 commits our own line). Resolved: not present as commits, **and not needed** — the defect class they fix does not exist in our kernel (§1.1, three-leg evidence).
- **Journal "262k daily driver still parallel:4" (SVG-CUSTOM.md:6676) vs the file:** custom-iq3s-262k.json (mtime Oct 14 17:56) says `"parallel": 6`. The file was updated after that note; the note is stale. The json matches standing parallel=6 — good for the A/B.
- **Expert slot count 9,337 vs 9,454:** the five-boot-era boot printed 9,337; T13A's live /metrics read 9,454 on a later boot (T13A §4.1). Boot-to-boot wobble (profile/adapt state); all slot predictions here carry a ±~100 band.
- **"+89 MiB" vs "+141 MiB draft head" between the Oct-14 and Oct-16 boots:** minor boot-recipe drift across those days (batch-MTP adoption window); not capacity-related; does not affect the delta arithmetic (which anchors on the session-size lines).

## Open questions

1. Conc sensitivity at −708 slots on the F-233 recipe — the 3v3's job (prior band −1..−4%).
2. Warm decode at 240k+ on the current recipe — only the Oct-14 anchor (173.9) exists; ladder + depth-lane will set the 262k scoreboard row.
3. Deep-conversation parks (>131k tokens) have never been exercised in any kept log (max parked 684 tokens in the 262k log; F-181's warm legs rode the deterministic-filler prompt cache). Byte math says fine; first real deep park is worth one glance at the cache lines.
4. Whether c7544ef6+4a9669ff (+78652d23) should still be ported for upstream convergence — out of scope here; T10 retains them ranked 5/4.

## Implications

- The switch is cheap, reversible, and mostly already paid for: F-181/F-182 did the heavy lifting; the only new information the campaign needs is the 3v3 on the F-233 recipe plus the 147k-245k ladder rungs.
- The standing scoreboard's exposure is concentrated in one number (conc, via the slot count). If conc dips 2-4%, the honest framing is F-183's trade, and the vram-800-style counterweight exists as a labeled variant — not a silent stack.
- The depth lane this opens is real but priced: ~−11% decode at 244k, prefill −5..−12% past 131k, acceptance holding ~79% — and the 1M shelf config remains the escape hatch for anything deeper.

## Sources

Primary (code, this tree, HEAD 34d3e4bc):
- include/strata/kernels/qsa.hpp:103-137 (idx_block 4, idx_top_k 2048, selection width), :279-282 (kTopkMaxCells, legacy)
- src/kernels/cuda/qsa_select.cu:604-705 (capacity-independent histograms), :875-960 + :1230-1300 (cluster kernel, dynamic smem = ceil(max_blocks/CL_N)×4, opt-in + occupancy check + fallback)
- src/kernels/qsa_topk_parity.cpp:34,114-115 and src/kernels/qsa_topk_active_parity.cpp:1-92 (capacity 262,144 / 524,288 parity cases)
- src/core/layer.cpp:629-687 (kv_plan, kv_pool_bytes, qsa_state_bytes, qsa_kv_resident_min), :910-923 (buffers sized by cap not max_cells), :844 (n_rot 64)
- src/program/generate.cpp:4484-4516 (kv-grow gating, KV streaming print), :6855-6990 (kvg_ensure/kvg_trim), :2741-2747 (max-context validation), :714-716 (conversation-cache defaults), :2000 (--max-context parse)
- include/strata/program/conv_cache.hpp (retention policy), include/strata/core/session.cpp:60-124 (session_bytes)

Logs and measured artifacts (this repo):
- custom-iq3_s.log (Oct 16 standing boot, batch 6: 0.73 GiB/slot, 9,337 slots, 1.55 GiB pinned)
- custom-iq3_s-262k.log (Oct 14 12:03-16:25 boot at 262144: 0.95 GiB/slot, 9,529 slots, 3.09 GiB pinned, 157k cold prompt served)
- custom-iq3_s-1m.log (Oct 14 12:06 boot at 1,048,576 + YaRN ×4: 12.38 GiB pinned, 937k depth served)
- custom-iq3_s-8081.json / custom-iq3s-262k.json (args diff = one line) / launch-custom-iq3s-8081.sh (Oct 17) / launch-custom-iq3s-262k.sh (Oct 14, stale env)
- campaign/battery-config.sh (sources standing exports), campaign/battery-boot.sh, campaign/probe-lane.py (M1/pair/6-conc/M3b), campaign/ctx-probe.py (fresh-prompt ladder), campaign/depth-lane.py (depth probe)

Journal and dossiers:
- SVG-CUSTOM.md F-181 (5815-5860: the ladder table, 262k/1M verdicts), F-182 (5862-5917: capacity-is-free A/B −0.2/−0.8/−1.3%), F-183 (5918-5981: slot-sensitivity arms), F-22b (962-969: kv-resident law), five-boot scoreboard (7998-8013), lines 243-244, 6145-6146, 6200, 6676, 6845-6846
- SVG-CUSTOM-TODO.md T10 (721-790: the ranked port list, items 4-5)
- campaign/research/T13A-decode-critical-path.md §4.4 (flat-decode mechanism: depth costs once-per-window)
- git: merge-base HEAD..eddo/custom = 296b59d2; c7544ef6/4a9669ff/78652d23/e6ed2a50 present on eddo/custom only; 8d1c9d64 (2026-10-14 01:09) = last qsa_select.cu change, before both Oct-14 capacity boots

Methods note (unproductive/deferred searches): the exact n_head_kv/head_dim GGUF metadata was not read directly — the 12,684 vs 13,056 B/cell fit (3%) was accepted as identification; the +89/+141 MiB draft-head discrepancy was left unresolved (immaterial); deep-park cache lines at >131k tokens do not exist in any kept log (checked, absent — listed as open question 3 instead).
