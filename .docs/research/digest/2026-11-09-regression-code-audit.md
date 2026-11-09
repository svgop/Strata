---
type: research-digest
date: 2026-11-09
topic: Regression code audit — does any commit in 4152f114..HEAD change the DEFAULT decode serving path vs the Nov-6 bank (GSQ IQ3_S 229.1/224.8/303.5), given the hugepage-pool tree and the current launcher env (STRATA_HUGEPAGE_ARENAS=1, pool 45076 pages)
machine-context: Same box as the bank (Zen5 + RTX 5070-class, 184 GiB RAM, boot-armed 2 MiB hugetlb pool of 45076 pages = 88.2 GiB). Engine serves GSQ IQ3_S; current readings 175-198 solo legs in logs/custom-iq3_s.log vs 222-266 in the bank era. Audit is analysis-only; no code changed.
status: verified
sources: git 4152f114..HEAD diffs (read in full for the six commit set), src/core/{pinned.cu,expert_source.cpp,shadow_experts.cpp} at HEAD and at 4152f114, include/strata/platform/hugepage.hpp, launch-custom-iq3s-8081.sh, logs/custom-iq3_s.log (662k lines, 292 boots, eras segmented by the dated `engine started` markers), campaign/results/{hostmem-leg0-verdict.md,fork-ud-first-bank.md}
---

# Regression code audit: 4152f114..HEAD vs the Nov-6 bank (2026-11-09)

## Verdict up front

**No commit in the audited set changes the default decode serving path for the worse.** All six commits that touch `src/core/{pinned.cu,expert_source.cpp,shadow_experts.cpp,host_mem.cpp,layer.cpp}` or `include/strata/platform/hugepage.hpp` are allocation-time (boot) changes; at serve time they leave the per-window path (complement blob pointers, shadow `blob()` reads, KV host shadow reads, slot arenas, copy/DMA enqueue) arithmetically and mechanically identical, and the backing-class changes (4K→2 MiB on the complement, tmpfs→memfd-hugetlb on the store) are serve-equal-or-better.

**One commit-window regression existed and is already closed:** the pool-blind budget cap (complement sized 17.39/19.37/19.80 GiB on the Nov-8 13:07-13:12 boots, no lend line, legs 164-191 t/s) — fixed by e8484c36, which restores Nov-6-era sizing (37.85-37.93 GiB, full lend coverage).

**The residual deficit vs the bank predates the entire hugepage tree.** The serve-time anatomy step (+2.0-2.5 ms/window in the in-run component; pair GPU-reach wait 9.2→11.5 ms; solo legs 237-266→190-205) happened between the Nov-7 12:17 and 15:30 boots — before 07530eb7 (Nov-7 20:10), the first hugepage commit. It was measured and named by the pre-hugepage campaign: leg-0 battery (0704fc1e) GSQ 206.3/186.7/258.9 = −10..−19% vs banks on pre-hugepage code, and the regression hunt (0a45cf44, Nov-7 15:38) exonerated code via the banked-era binary and named box state as the last variable.

## Scope and method

- Audit set: `git log --oneline 4152f114..HEAD -- src/core/pinned.cu src/core/expert_source.cpp src/core/shadow_experts.cpp src/core/host_mem.cpp src/core/layer.cpp include/strata/platform/hugepage.hpp` → six commits. `host_mem.cpp` and `layer.cpp` are untouched in the range.
- f95455a6 (single-owner launchers) touches no engine file; audited as launcher-level for completeness.
- Serve-time evidence: `logs/custom-iq3_s.log`, boots segmented by dated markers. Bank era = Nov-6 19:39-21:42 battery boots (the 229.1/224.8/303.5 mean window) plus the Nov-7 morning control boots; comparison era = Nov-9 10:22-10:56 boots (pool + memfd + budget fix + arenas armed). Config matching by the anatomy line's own fields (window count ~550/720 and avg rows 3.1-3.2/4.7-4.8 identify pair/conc sustained cadence).

## Boot-era map (complement sizing, the (c) answer)

From `mapped pinned cache complement ready` / `room for the complement` lines:

| Era | Boot(s) | Complement | Lend line | Backing |
|---|---|---|---|---|
| Nov-6 bank battery | 19:51-20:02 | 37.52-37.68 GiB | 5184-5186 of 5186 kept (full) | one-shot cudaHostAlloc, 4K |
| Nov-7 morning (healthy) | 11:16-12:17 | 37.55-37.68 GiB | 5184-5186 kept (full) | one-shot cudaHostAlloc, 4K |
| Nov-7 evening legs | 19:27-20:40 | 37.59-37.79 GiB | full | THP fallback (0.01/37.67 accepted), then HUGETLB POOL from 20:32 |
| Nov-8 capped (the real regression) | 13:07/13:10/13:12 | **17.39 / 19.37 / 19.80 GiB** | **absent** (no lend coverage) | HUGETLB POOL, budget-capped |
| Nov-8 first memfd boot | 12:58 | 37.97 GiB | 5192 kept (full) | pool + memfd store |
| Nov-9 (current launcher env) | 10:22-10:41 | **37.85-37.93 GiB** | **5187-5189 of 5189 kept (full)** | HUGETLB POOL, paced, memfd store |

Nov-6 need-side target was 37.52-37.95 GiB; Nov-9 is 37.85-37.93 GiB — the same need-driven class (drift follows GPU-cache slot count, see below). The budget fix did **not** grow the complement past the bank era; it un-capped it back to the bank era. Lend/PCIe arithmetic: full lend coverage in both eras, borrow counts equal to lend counts, routed PCIe share 0.3-0.9% (bank) vs 0.3-0.7% (Nov-9). **No split-arithmetic change.**

## Commit-by-commit verdicts

### 07530eb7 — 2026-11-07 20:10 — huge_alloc tries MAP_HUGETLB first; M3 page-class announce; smaps `ht` counting — **BOOT-ONLY**
- What changed: `huge_alloc()` (hugepage.hpp) attempts `MAP_HUGETLB|MAP_HUGE_2MB` first with the THP/madvise path as fallback; `pinned.cu` gains `announce_page_class()` (one boot line per ≥1 GiB arena; `STRATA_ARENA_PAGES=strict` exit 70) and the `hugepage_measure/report` signatures grow the `ht` output.
- Serve-time path: none. No per-window code path is touched; the announce runs once per arena at construction; the smaps reader is boot-only.
- Boot-time hazard (observed, later fixed): at this commit `huge_alloc` had no size gate, so ~90 per-layer KV host shadows × 0.26 GiB could draw the pool before the complement's cover check (the timeline evidence in e00b9c16's message). On the current tree the gate exists (e00b9c16) and today's boots show the shadows ending 4K/THP-rejected — the same class as Nov-6 (THP=madvise never promoted them then either; the per-shadow `hugepages:` lines simply did not exist at base).
- Log evidence: KV host shadow lines `0.00-0.10 GiB of 0.26 GiB in 2 MiB pages (THP accepted/rejected)` on all Nov-9 boots; KV streaming and slot-copy lines unchanged in shape.

### 511bece6 — 2026-11-07 20:17 — the paced flip for the complement — **BENIGN** (boot route change; serve-equal-or-better)
- What changed: with arenas armed and `huge_pool_covers(bytes)` true, `pin_cache_complement` takes the paced path (huge_alloc → fault in 1 GiB steps → ONE `cudaHostRegister(Mapped|Portable)` over the faulted prefix) instead of one-shot `cudaHostAlloc(Mapped)`.
- Serve-time scrutiny (item a): the registration is a single span cut at an **expert boundary** — the code scans `offsets` and backs `w` off so no blob straddles the registered/unregistered edge (a blob crossing the edge would be refused by `cudaMemcpyAsync`; the cut prevents that). Blob alignment against 2 MiB pool pages is a non-issue: blobs stay packed in file order at unchanged offsets; reads are `arena_base + offset`, and unaligned host reads inside registered memory behave identically regardless of page size. Device alias comes from `cudaHostGetDevicePointer` over the registered span — the same zero-copy DMA contract as `cudaHostAllocMapped`; the enqueue/copy path is unchanged. Per-window cost vs 4K cudaHostAlloc: none negative; dTLB/IOTLB reach strictly better (the F-201-zen5 mechanism this campaign prices at +5-15% on the staging path). Line 3885's own invariant says it: a paced registration covering the whole arena is as good as a cudaHostAlloc'd one.
- Log evidence: complement `ready` in 13-22 s on pool boots vs 24-36 s one-shot at base; per-layer-window CPU experts bank 1.13-1.68 vs Nov-9 0.80-1.68; anatomy CPU-experts term bank 2.05-3.47 ms vs Nov-9 1.90-2.85 ms. No regression in the CPU lane.

### 2e19d10e — 2026-11-07 20:27 — pool priority in `reserve()` (<8 GiB skipped) — **BOOT-ONLY**
- What changed: sub-8-GiB arenas (session/conversation slot arenas, 24 × 0.97 GiB) skip the pool and take the same general anon+THP/madvise path they had at 4152f114 (verified: base `reserve()` already had the aligned-mmap + conditional MADV_HUGEPAGE chain; HEAD adds only the skip note and the hugetlb first-try for ≥8 GiB).
- Serve-time: slot arenas' backing class is unchanged vs Nov-6; slot copy lines (`slot N takes T tokens (copied in X ms)`) show the same shape and magnitudes.

### e00b9c16 — 2026-11-07 20:31 — huge_alloc ≥8 GiB gate (mirrors pinned.cu) — **BOOT-ONLY**
- What changed: `huge_alloc` skips the pool below 8 GiB unless `STRATA_POOL_SMALL_ARENAS=1`; restores the complement's `huge_pool_covers()` check winning the boot ordering race.
- Serve-time: none. It decides who draws the pool at boot; today's boots prove the outcome (complement HUGETLB POOL 37.9x GiB every boot).

### 8aa066b1 — 2026-11-08 12:57 — the store goes engine-level (memfd) — **BENIGN** (serve-equal-or-better; boot +5.7 s populate)
- What changed: `shadow::arm()` maps the store via `memfd_create(MFD_HUGETLB|MFD_HUGE_2MB)` + ftruncate + chunked memcpy populate + `mprotect(PROT_READ)` when `huge_pool_covers(want)`; falls back to the plain file `mmap(PROT_READ, MAP_SHARED)` otherwise; `STRATA_SHADOW_MEMFD=0` kills it.
- Serve-time scrutiny (item b): `blob()` is untouched pointer arithmetic (`st.base + layer_off[layer] + expert*BLOB`) in both eras. The mapping keeps `MAP_SHARED` semantics; hugetlb-shmem pages and tmpfs-shmem pages are the same page class for the read path (no syscall, no copy — straight loads from the mapping). 2 MiB pages by construction (no THP dependence), fully prefaulted by the populate (no fault-on-read stalls in windows), and `mprotect(PROT_READ)` is a set-once boot action with zero per-access cost — the store never took a write path at serve time anyway. TLB reach improves (31.64 GiB on 2 MiB pages ≈ 16k entries vs ~8M PTEs). No per-window regression mechanism exists in this diff.
- Log evidence: bank `STRATA_SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0: ... 31.64 GiB; decode CPU misses compute from it` vs Nov-9 same line + `2 MiB pool pages (memfd, populated in 5.7 s)`. Same store, same path, same bytes.

### e8484c36 — 2026-11-09 10:22 — the pool-blind budget fixed — **BENIGN** (restorative; the pre-fix cap was the real commit-window regression and is closed)
- What changed: the resident-memory budget in `pin_cache_complement` adds `huge_pool_free_bytes()` when arenas are armed (the same correction dsv41's copy budget has carried in ds4_os.hpp).
- Scrutiny (item c): see the boot-era map. Nov-9 complements 37.85-37.93 GiB vs bank-era 37.52-37.95 GiB — same need-side target; lend coverage full in both; hit rates equal; PCIe share equal. The Nov-8 capped boots (17.39-19.80 GiB, no lend line, legs 164-191 t/s, the operator-measured solo 120-165) were caused by the ABSENCE of this correction, not by the commit; the commit is the fix. `huge_pool_free_bytes()` reads /proc/meminfo once at boot — no serve-time call site.

### f95455a6 — 2026-11-09 10:38 — single-owner store rule in the launchers — **BOOT-ONLY** (launcher file; outside the audited engine set)
- The engine's memfd builds from the canonical pack; the /dev/shm fallback copy is skipped when the pool covers the store. Store content and mapping are identical; the change removes dead resident weight (31.6 GiB held twice) and page-cache squeeze. No serve-time path.

## Era-vs-era serve-time log line comparison

Config-matched sustained lines (window cadence and avg rows identify the lane; bank = Nov-6 19:57/19:59/20:02 + Nov-7 11:16/11:34/12:17 boots; current = Nov-9 10:22/10:27/10:30/10:38 boots):

| Line pattern | Nov-6/7 bank era | Nov-9 (current env) | Read |
|---|---|---|---|
| `strata batch:` CONC sustained (w≈710-730, rows 4.7-4.8) | ms/window 13.41-14.45; run 12.11-13.36 (gpu_wait 0.39-0.67 + cpu 2.05-3.37); adapt 0.44-0.81; rows/s 256-277 | ms/window 15.52-16.21; run 14.01-14.74 (gpu_wait 0.34-0.59 + cpu 1.90-2.62); adapt 0.68-1.05; rows/s 224-242 | **+2.0-2.2 ms/window**, all in the in-run component NOT attributed to gpu_wait or CPU experts (run − gw − cpu: 8.9-9.6 → 11.5-12.2); routing shares unchanged |
| `strata batch:` PAIR sustained (w≈550-565, rows 3.1-3.2) | ms 14.20-14.34; gpu_wait 9.17-9.28; cpu 2.56-2.68; rows/s 203-207 | ms 16.27-17.31; gpu_wait 11.43-11.96; cpu 2.18-2.85; rows/s 164-185 | GPU-reach wait **+2.2-2.7 ms/window**; CPU term flat |
| per-request decode t/s (`N generated in M ms (T tok/s)`, gen≈1490 legs) | 219-250 (Nov-6), 222-236 (battery), 237-266 (Nov-7 12:17) | 191.0 / 195.7 / 195.7 | **−15-18%**, matches the operator's 175-198 standing readings |
| long solo legs (gen ≥5.6k) | 220.9 / 215.7 / 237-266 (Nov-7 am) | 173.8 (9661) / 198.0 (23003) / 182.5 (54006) | −15-20% |
| `decode expert cache hit rate` session finals (~810-870k lookups) | 96.6-98.0% | 96.3-97.9% | same pattern, −0.3 pp (noise) |
| `routed entries per window` shares | VRAM 93.0-97.3%, PCIe 0.3-1.5%, CPU 2.3-5.6% | VRAM 94.2-97.9%, PCIe 0.2-1.1%, CPU 1.8-4.6% | same shape; CPU share not worse |
| per layer-window `CPU experts` | 1.13-1.68 | 0.80-1.68 | equal or better on the 2 MiB complement |
| `resident RAM:` first-request sequence | 37.52-37.64 GiB; exchanges 125→298; **0** file reads | 37.86-37.91 GiB; exchanges 34→260; **0** file reads | identical cold-start path (the 1181-1749 file reads seen later on Nov-9 accumulate over 54k+token sessions the 2-minute bank legs never reached) |
| `decode waited` distribution | dominant `0.0 ms in 0 fetches` | dominant `0.0 ms in 0 fetches` | identical |
| GPU cache slots (`expert cache N slots`) | 10015-10062 (19.00-19.09 GiB) | 9863-9903 (18.71-18.79 GiB) | **−1.8% slots (−0.35 GiB VRAM)** — dropped at the Nov-8 reboot, before which Nov-7 was already 10031-10062; minor, not the deficit |
| lend region (`N of M lendable slots keep`) | 5183-5192 of M (full) | 5187-5189 of M (full) | identical arithmetic |
| adapt wait, sustained legs | 0.36-0.89 | 0.68-1.17 | +0.3 ms, admission-timing noise |

**Same patterns, different values:** the only structural value that moved is the window time (+2-2.5 ms) and it moved in a component the audited commits do not own (in-run launch/GPU-side pacing, visible as pair gpu_wait growth), and it moved **before** those commits existed.

## The named mechanism (what the deficit is NOT, and where to look)

- **Not the audited commits.** The step is present in the Nov-7 15:30 boot: conc ms/window 15.35 (in-run component 11.30) vs 12.16 (8.58) at 12:17 the same morning; solo legs fall 237-266 → 190-205 and long legs 216-221 → 140-153 — hours before 07530eb7 (20:10). The hugepage series' own leg verdicts confirm: leg-1 control read −12..−17% (8c912894), leg-3 pool complement solo +7.2% (9cedd332) — improvements measured against the already-degraded baseline.
- **Already hunted, code exonerated:** 0a45cf44 (Nov-7 15:38) closed the prior hunt — chart artifact, zram, and the v2 GPU-clock bounce (v3 shipped, `d6554f68`, 14:58 the same day — inside the degradation onset window) — with "code exonerated via the banked-era binary; the ~25% residual stands; memory fragmentation is the last unchecked variable". 0704fc1e measured GSQ 206.3/186.7/258.9 (−10..−19% vs banks) on pre-hugepage code, aged box, and named the free-2MiB supply collapse (18→1-2-5 GiB across the week).
- **The reboot did not restore the bank.** Post-reboot (Nov-8+) the capped complement masked recovery; e8484c36 un-masked it (Nov-9 legs 191-198 vs capped-era 164-191), but the in-run +2 ms component and pair gpu_wait persist across the reboot — so the residual is either still box/driver state (clock service posture, GSP, page-cache/memcg pressure under the 69.5 GiB pool draw) or a non-audited tree change from the same Nov-6→Nov-7 window (the doorbell flag-wait conversion in b3ce99f2 is the candidate that touches the launch/pacing path; it shipped Nov-6 19:39, inside the healthy band, so it does not explain the Nov-7 step either).
- **Next probes (law-17, same-hour, today's box):** (1) `STRATA_HUGEPAGE_ARENAS=0` one leg — prices the whole pool architecture against the residual on the current kernel state; (2) `STRATA_SHADOW_MEMFD=0` one leg — isolates the store; (3) verify the clock service v4 is live and the sustained boost holds ~2950 MHz under a lane (the v2 bounce produced exactly this anatomy shape: uniform inflation, gpu_wait growth, −15-30%); (4) recover the −0.35 GiB VRAM the reboot cost the GPU cache (10062 → 9872 slots).

## Boundary

This audit reads code and logs only; no boots were run. The gen≈1490 leg t/s and the anatomy rows/s are engine-side instruments, not the bench harness's bank instrument; cross-era comparisons here are same-instrument and config-matched by the anatomy line's own cadence fields. The bank mean 229.1/224.8/303.5 corresponds to the Nov-6 evening battery boots in this log; per-leg variance across that day (160-266 t/s solo legs) is itself visible in the bank era, so single-leg deltas under ~8% should not be read as mechanisms.
