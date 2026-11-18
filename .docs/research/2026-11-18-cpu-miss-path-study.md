---
type: research-study
date: 2026-11-18
subject: The GSQ decode-miss CPU lane — code path (shadow store arm ladder, pool dispatch, dequant, PCIe), the per-window cost model behind the 2.04 -> 4.2-4.9 ms/window doubling, the window-remainder accounting, and the memory-pressure guard that failed (NaN #879)
repo: /home/svg/LLM/strata-custom-2 (read-only study; HEAD 02c216e09 + uncommitted canonical-primary launcher/lib work e4dd2e3c4/c6bbc6ed5/89c00d737)
status: complete
---

# The decode-miss CPU lane: code path, cost model, doubling accounting, pressure guard

## 0. Verdict first

| Question | Answer |
| --- | --- |
| Where is `cpu-exp` measured? | `generate.cpp:12930-12944` — the batch-flush anatomy line. `CPU experts` = `(ver.ms_pool - bt_pool0)/w`; `ms_pool` accumulates per layer in `verify.cpp:3238` as the host wait from the GPU doorbell (`b`) through the full dispatch callback (`generate.cpp:1234` -> `expert_pool_dispatch_multi`) including `run_split_multi`'s join. Plan/actq sub-terms are inside it (`expert_source.cpp:5106-5109`). |
| What does one miss cost? | Byte-bound blob stream + dequant on one pool lane. Falsifier (SVG-CUSTOM F-359, `SVG-CUSTOM.md:11561`): IQ3_S 2,176,000 B L0 blob = 152.2 us/entry (14.3 GB/s); Q2_0 1,382,400 B = 75.1 us/entry (18.4 GB/s). Engine-effective (this log): ~43-46 us/entry shadow-armed, ~92-93 us/entry disarmed — ratio 2.0. |
| Reproduce the doubling? | 2.04 ms x (152.2/75.1) = 4.13 ms predicted; measured 4.23-4.93 (t66-warm r1-r3, `logs/custom-iq3_s.log:128123,129873,131623`). The band's +0.1-0.8 ms over prediction = environment residual (ntfs3 store mount, 16->8 visible physical cores, arena/file mix). |
| Window remainder (+3.4-4.1 ms window vs +2.3-2.4 cpu-exp) | GPU side is FLAT (gpu-wait 0.53-0.57, commit 0.36-0.37, emit 0.37-0.38; PCIe entries fell 16.4 -> 10.4/window; rows/accept flat). The remainder is the adaptive tier: `adapt wait 0.41 -> 1.16 ms/window` with swap rounds 53 -> 91 and swapped-in 3334 -> 6440 (`logs/custom-iq3_s.log:1037` vs `:128123`). No GPU dequant change exists: the GPU computes native IQ3_S in BOTH eras (`expert_source.cpp:4596-4601,4991` — the shadow only swaps the kind==-1 blob pointer). |
| Which guard failed (NaN #879)? | The arm ladder's reservation-failure branch does not refuse. `shadow_experts.cpp:161-228`: the hugetlb memfd `MAP_SHARED` mmap is the atomic pool check; on -ENOMEM it only sets `mfd_note` (`:205-215`, "the swappable tmpfs tier serves every miss") and falls through to a plain file mmap (`:225-228`) + best-effort mlock (`:234-240`). Live: all 39 armed banners in `logs/custom-iq3_s.log` read "the pool could not reserve the store (0.0 GiB free of 31.6 GiB needed)" (first `:1045`, last `:124808`) — every armed boot post-estate-move served misses from the swappable tier. Fix class shipped at the launcher (canonical-primary, tmpfs opt-in: `campaign/lib/launch-common.sh:73-91`); the engine-side refusal under pressure is still open. |

Primary sources: `logs/custom-iq3_s.log` (132,324 lines, 11-09 16:49 -> 11-18 18:12, 48 boots, 225 flush lines), `SVG-CUSTOM.md` F-359/F-360/F-365/F-367 (`:11561-11790`), campaign commit `385937a4d` (NaN census), sibling study `.docs/research/2026-11-18-cpu-exp-timeseries.md` (onset segmentation).

## 1. The accounting site

The `cpu-exp` number the probes quote is built here:

- `src/program/generate.cpp:12930-12944` — `"strata batch: %lld windows, avg %.2f rows, %.2f ms/window = run %.2f (CUDA0 GPU-reach wait %.2f + CPU experts %.2f) + commit %.2f + emit %.2f + adapt wait %.2f; ... per layer-window: CPU experts %.2f, VRAM hits %.2f, PCIe %.2f; routed entries per window: VRAM %.1f (%.1f%%), PCIe %.1f (%.1f%%), CPU %.1f (%.1f%%)"`.
  - `CPU experts` = `(ver.ms_pool - bt_pool0) / w` — per window.
  - `per layer-window CPU experts` = `(drive.d.multi_misses - bt_miss0) / (w * L)` — DISTINCT CPU experts per layer-window (`multi_misses += njobs`, `expert_source.cpp:5114`).
  - `CPU %` routed entries = `multi_entries` delta (per-pick, `expert_source.cpp:5031`).
  - `GPU-reach wait` = `ver.ms_wait`; the window total = run + commit + emit + adapt wait — so the doubling decomposes exactly across these printed terms.
- `src/core/verify.cpp:3238` — `ms_pool += ms_since(b)`; `b` is taken after the GPU doorbell ring (`:3185` region: spin on flag, `b = Clock::now()`), and the dispatch callback runs at `:3231-3233` (`pool(user, h_x_ + ..., h_ids_ + ..., n, ss.k, h_ymiss_ + ..., l)`). So `ms_pool` = plan + activation-quant + blob resolve + `run_split_multi` (with its worker join) + cold-read waits, per layer, summed over 48 layers. (`ms_pool` also accumulates at `verify.cpp:4043,4230,4328,4629` — the other verify shapes; this lane is the batch decode path.)
- The dispatch entry: `src/program/generate.cpp:1234` -> `strata::core::expert_pool_dispatch_multi(t->d, ...)` (`src/core/expert_source.cpp:4570`).
- Counters: `cache_hits` (`expert_source.cpp:4975`), `offload_entries`/`pcie_experts` (`:4976`, `:4839`), `multi_entries` (`:5031`), `multi_misses` (distinct jobs, `:5114`), `shadow_misses` (`:5005`).
- The round-level printer that names them: `generate.cpp:18272-18302` — "verify window ... pool ... CPU experts %.2f distinct / %.2f routed per layer", "shadow experts: %lld of %lld CPU misses computed from the shadow Q2_0 store", "resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the VRAM tier, %lld blob reads from the file".

## 2. The code path, end to end

### 2.1 The shadow store (`src/core/shadow_experts.cpp`, 271 lines)

- `arm(n_layers, n_expert)` (`:74`). Gate order:
  1. env `STRATA_SHADOW_EXPERTS` — unset or EMPTY returns false, one read, `st.tried` latched (`:76-80`). This is the rollback's byte-identity guarantee (`expert_source.cpp:4599-4601`: "Unset, `shadow` is false and nothing below takes a new branch").
  2. Hadamard-folded packs refused — the canonical Q2_0 form cannot carry rotated signs (`:84-85`).
  3. `experts.json` canonicality: `blob_bytes == BLOB` (`:100-102`), `n_layers` match (`:103-105`), `total_bytes == n_layers * n_expert * BLOB` (`:106-109`); per-layer offsets DERIVED arithmetically, not scanned (F-360's mis-key fix, `:110-114`).
  4. `experts.bin` size must equal `want` exactly (`:145-152`; Windows twin `:127-134`).
- Tier ladder (POSIX):
  1. **Hugetlb memfd tier** (`:161-224`, F4 2026-11-09): `memfd_create(..., MFD_HUGETLB | MFD_HUGE_2MB)` (`:173`), `ftruncate` to 2 MiB-rounded size (`:176-177` — does NOT reserve), then `mmap(dst, MAP_SHARED, mfd)` (`:180`) — the mapping's own reservation IS the pool check ("HugePages_Rsvd moved by exactly the store's page count, all-or-nothing, -ENOMEM", comment `:163-170`). `MADV_POPULATE_WRITE` + copy (`:185-197`), `mprotect(PROT_READ)` (`:199`).
  2. **On reservation failure: no refusal.** `dst == MAP_FAILED` only sets `mfd_note` — `"the pool could not reserve the store (%.1f GiB free of %.1f GiB needed) - the swappable tmpfs tier serves every miss"` (`:205-215`) — and falls through.
  3. **File-mapping fallback** (`:225-228`): plain `mmap(PROT_READ, MAP_SHARED, fd)` of `<dir>/experts.bin` — 4 KiB pages, page-cache/swap-eligible.
  4. **Best-effort mlock** (`:234-240`): on failure prints "mlock refused ... the store stays reclaimable" and continues. The comment records the measured failure mode: "a /dev/shm store mapping had 299 MiB in zram, and every miss-path read of a swapped page pays decompression" (`:229-233`).
  5. `st.armed = true` (`:244`), banner (`:249-254`).
- `blob(layer, expert)` (`:260-264`) = `base + layer_off[layer] + expert * BLOB`. `BLOB = 3 * (H*FF*BB/QK) = 1,382,400` (`include/strata/kernels/cpu/expert.hpp:44-51`; H=2560, FF=640, NE=512, QK=64, BB=18).

### 2.2 The miss-service path (`src/core/expert_source.cpp`, dispatch at `:4570`)

Per layer-window (n_tok rows x k picks):

1. **Plan** (`:4636-4849`): single pass builds distinct-expert groups in routing order (`:4679-4714`); each distinct expert classified by `host_res` (VRAM slot >= 0 -> kind 0, pointer `cache_base + slot*cache_blob`, `:4791-4795`), peer/helper (`:4796-4802`), Foresight slot (`:4800-4801`), else a miss. `pcie-frac` 0.35 / `pcie-mode dma` (config `config/custom-iq3_s-8081.json:28-31`) picks `m = d.pcie_model.pick(nmiss, (nmiss*d.pcie_num)>>8)` (`:4770-4771`); the LAST `m` missed experts in routing order go to the GPU **only if their native blob is pinned** (`:4804-4812`: `d.src->pinned(...) ? d.src->blob(...) : nullptr`, else stays a CPU miss). DMA source pointers published at `:4831-4832`, copy engine enqueued `:4849`. STRATA_ROUTE_TAIL_SKIP=7 removes rank>=7-only misses entirely (`:4731-4762`).
2. **Activation quant** (`:4893-4906`): shadow -> `act_quant_q8_1` (canonical Q8_1, `:4898-4899`); native IQ3_S -> `native_quant_act` into `nact_multi` (`:4902-4903`).
3. **Blob resolve for misses** (`:4985-5014`): `const uint8_t* b = shadow ? strata::core::shadow::blob(d.layers, e) : d.src->blob(d.layers, e)` (`:4990-4991`). This line is the whole shadow mechanism. Shadow skips the native prefetch entirely (`:4924-4926`). Native `blob()` service order (`:4352-4390`): RAM resident arena (`resident_blob`, `ram_reads_` counter) -> exchange override -> staged/IO-prefetch -> `mapped_blob` (the GGUF mmap; `file_read_bytes_` counter). `FileExpertSource::pinned` (`:4392-4403`) gates the PCIe share on the complement pin limit.
4. **Pool run** (`:5038-5039`, `:5083-5084`): `native && !shadow` -> `run_split_multi_native(lay.fmt, jobs, njobs)`; else -> `run_split_multi(jobs, njobs)`.

### 2.3 The two pool kernels (`src/kernels/cpu/pool.cpp`)

- `run_split_multi` (`:1064-1101`) — canonical Q2_0: phase 3 (gate+up, VNNI `vpdpbusd` + VBMI `vpmultishiftqb` bitplane unpack, `expert.hpp:9-22`), re-quant `act_quant_q8_1` per row (`:1087-1089`), phase 4 (down). Bytes ledger: `multi_bytes += n * BLOB` (`:1098`).
- `run_split_multi_native` (`:1103-1132`) — native IQ3_S: batches of `kMaxSplitMulti`, phases 5/6 with `NativeFmt` offsets, `act_quant_any`/`native_quant_h` re-quant (`:1116-1119`); ledger `multi_bytes += n * f.bytes` (`:1129`). IQK dequant is compute-heavier per byte than the Q2_0 bitplane path — F-359: Q2_0 is ">=32% faster after dequant" (`SVG-CUSTOM.md:11567`).
- Row-split parallelism: `phase_tasks` = 3 tasks/worker capped by rows (`:1055-1061`); a layer's ~0.8-0.9 distinct jobs split their rows across workers, so a single blob streams over a handful of cores — engine-effective per-entry lands ~1.5x the single-core falsifier rate in BOTH eras (bank Q2_0: 1.38 MB / ~44 us = 31 GB/s vs 18.4 single-core; today IQ3_S: 2.05 MB / ~93 us = 22 GB/s vs 14.3 single-core). The 16->8 visible-core change (below) therefore does NOT multiply per-entry time; it caps concurrency headroom.

## 3. The cost model

### 3.1 Constants

| Quantity | Value | Source |
| --- | --- | --- |
| Layers x experts | 48 x 512 | banner `logs/custom-iq3_s.log:1045` |
| Q2_0 blob (shadow) | 1,382,400 B | `expert.hpp:51` |
| IQ3_S blob (native) | L0 2,176,000 B; Oct-pack mean 2,046,400 B (1.48x Q2_0); current-pack figure 2.66 MB quoted by the timeseries study | `SVG-CUSTOM.md:11567-11569`; `.docs/research/2026-11-18-cpu-exp-timeseries.md` §0 |
| Single-core stream + dequant | IQ3_S 152.2 us/entry (14.3 GB/s); Q2_0 75.1 us/entry (18.4 GB/s) | F-359 falsifier, `SVG-CUSTOM.md:11561-11569` |
| Window shape (rows ~4.7-4.85) | ~1502 routed entries/window = rows(4.84) x k(~6.5) x 48; VRAM 96.2-96.3%, PCIe 0.5-0.7%, CPU 3.1-3.4% | `logs/custom-iq3_s.log:128123,131623` |
| CPU misses per window | 46.5-52.1 entries; 0.80-0.90 distinct/layer-window (38-43 distinct blobs) | same lines |
| pcie-frac / mode | 0.35 / dma | `config/custom-iq3_s-8081.json:28-31` |

### 3.2 Per-window arithmetic (rows 4.84, 48 layers, hit ~96.2%)

Misses/window = routed entries x CPU share = 1501.9 x 3.15% = **47.3** (matches the printed CPU 47.3). Distinct blobs = 0.84 x 48 = **40.3** (multi-entry jobs share one stream across their tokens).

**With shadow (Q2_0, bank era):**
```
cpu-exp = 47.3 entries x t_eff(Q2_0)
t_eff(Q2_0, engine) ≈ 43-46 us  (bank flushes: 2.00-2.48 ms / 46-64 entries)
=> 47.3 x 44 us ≈ 2.08 ms          measured: 2.00 (bank median), 2.04 (classic lines), 2.48 @ 64 entries
byte check: 40.3 blobs x 1.3824 MB = 55.7 MB/window at ~27 GB/s aggregate ≈ 2.0 ms
```

**Without shadow (IQ3_S, rollback era):**
```
cpu-exp = 47.3 entries x t_eff(IQ3_S)
t_eff(IQ3_S, engine) ≈ 92-93 us  (today: 4.38 ms / 47.3 = 92.6 us; 4.93/52.1 = 94.6; 4.23/46.5 = 91.0)
=> 47.3 x 92.6 us = 4.38 ms        measured: 4.23 / 4.38 / 4.93 (t66 r3/r1/r2)
byte check: 40.3 blobs x 2.0464 MB = 82.5 MB/window at ~19 GB/s aggregate ≈ 4.4 ms
```

**The doubling law:** per-entry ratio = bytes(1.48x) x per-byte dequant/stream penalty(18.4/14.3 = 1.29x) = **1.90x** on mean blobs, **2.03x** on the falsifier's per-entry figures (152.2/75.1). Applied to the bank 2.04: predicted **4.13 ms**; measured band **4.23-4.93** (within 2-19%; the excess is §4's environment residual). The launcher's own comment states the same law (`launch-custom-iq3s-8081.sh:78-80`).

### 3.3 Era table (flush lines, matched shape)

| Era | Boot / line | Shadow tier | workers | rows | window ms | gpu-wait | cpu-exp | CPU entries (share) | adapt wait | rows/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oct 17 native (pre-shadow reference) | old-logs/custom-iq3s-t17-dum48.log:1118 | none | 15 | 4.96 | 14.20 | 0.54 | 2.04 | 34.9 (2.3%) | 0.27 | 288.6 |
| Bank (shadow armed, /home q2_0 file mapping, pool 0.0 GiB) | logs/custom-iq3_s.log:1037 (11-10 era, 15 workers) | ext4 mmap + best-effort mlock | 15 | 4.85 | 14.41 | 0.57 | 2.48 | 64.0 (4.2%) | 0.41 | 272.8 |
| Bank median (timeseries §2) | 11-10 13:37 boot | same | 15 | 4.70 | 13.70 | 1.16 | 2.00 | ~44 (2.9%) | — | 217.6 |
| **Rollback (t66-warm r1)** | logs/custom-iq3_s.log:128123 | none (unset) | 7 | 4.84 | 17.76 | 0.53 | 4.38 | 47.3 (3.1%) | 1.16 | 209.1 |
| Rollback (r2) | :129873 | none | 7 | 4.87 | 17.98 | 0.50 | 4.93 | 52.1 (3.4%) | 1.27 | 225.5 |
| Rollback (r3) | :131623 | none | 7 | 4.79 | 17.05 | 0.67 | 4.23 | 46.5 (3.1%) | 1.00 | 228.8 |

Note the honest labels: the Oct 17 "2.04" lines are the NATIVE path on the old ext4 box (15 workers) — nearly the same cpu-exp as the shadow-armed bank. The clean same-box shadow A/B is F-360's ladder (`SVG-CUSTOM.md:11606-11615`: pair +7.7%, conc +8.4% end-to-end); the 2.04 -> 4.4 doubling prices the COMBINED quant swap + estate-move environment (§6).

### 3.4 The confounds that do NOT explain the doubling

- **Pool workers 16 -> 8 visible physical cores** (15 -> 7 workers; `logs/custom-iq3_s.log:104226` vs `:131665`, change lands between boots `:105991` and `:112633`, i.e. inside the Nov 10 -> Nov 18 gap): does not multiply per-entry time — the row-split factor is ~1.5x single-core in BOTH eras (§2.3). It caps the conc lane's headroom and shows in rows/s.
- **Hit rate / miss count**: constant (96.2-96.3% vs 96.2-96.8% bank; 46.5-52.1 vs ~44-64 entries — the bank line with MORE entries still ran 2.48).
- **PCIe bytes**: PCIe entries FELL bank -> rollback (16.4 -> 10.4/window; 33.6 -> 21.3 MB/window of 2.05 MB blobs) and the share reads the NATIVE pinned blobs in both eras anyway (`expert_source.cpp:4805`).

## 4. Window-remainder accounting (the "+4.4 window but only +2.4 cpu-exp" term)

Take the matched legs (bank :1037 -> rollback :128123; bank median -> t66-r1 gives the same picture):

```
Δwindow   = 17.76 - 14.41 = +3.35 ms   (bank-median pairing: 17.76 - 13.70 = +4.06)
Δrun      = 15.86 - 13.28 = +2.58
  Δcpu-exp = 4.38 - 2.48 = +1.90     (median pairing: +2.38)
  Δgpu-wait = 0.53 - 0.57 = -0.04    (median pairing: -0.27)
Δcommit   = 0.37 - 0.36 = +0.01
Δemit     = 0.38 - 0.37 = +0.01
Δadapt    = 1.16 - 0.41 = +0.75
sum ≈ +3.31 (vs +3.35 printed; remainder is leg variance)
```

- **GPU-side cost did not move.** The GPU dequantizes native IQ3_S for VRAM hits AND the PCIe share in BOTH eras — the shadow never touches the plan's kinds (`expert_source.cpp:4596-4601`, `:4991`); measured: gpu-wait 0.50-0.67 (today) vs 0.54-0.57 (bank matched legs), commit/emit +0.01 each, rows 4.84-4.87 vs 4.85-4.96, accept 64.3-65.7% vs 63.4-68.3%.
- **PCIe transfer**: FEWER bytes today (§3.4); `pcie-mode dma` reads the same pinned native staging. There is no "+bigger blob over PCIe" term — that hypothesis is dead on the counters.
- **The remainder is the adaptive tier**: `adapt wait 0.41 -> 1.16 ms/window`, rounds 53 -> 91, experts swapped in 3334 -> 6440 per flush window (:1037 vs :128123). Mechanism: the disarmed miss stream reads the native complement through the RAM arena + ntfs3 mmap (`blob()` order, `expert_source.cpp:4352-4390`), and the residency churn (usage/pred/heat, F-332 async scan) grows with the colder, slower-serving tier; the swap lane shares DRAM/page-cache bandwidth with the very misses that got slower. This is the same term the live-telemetry decomposition study flags (`pool_workers 15 -> 7`, `.docs/research/2026-11-18-live-telemetry-decomposition.md`).
- **Verify-window expert delivery**: unchanged — one multi-token dispatch serves the whole window's picks (n_tok x k, `verify.cpp:3231-3233`); rows/window and accept are flat, so the delivery shape contributes ~0 of the delta.

## 5. The pressure guard that failed (NaN #879 root-cause)

### 5.1 The census

- Commit `385937a4d` (2026-11-18 17:22): armed = every request dies at its first window (non-finite logits #879, 29/29); same engine + flags + exact 65-token prompt, shadow unset = clean 3/3. Store cmp-identical to the canonical pack (all four files). Root-cause "pending (the ActQ/run_split_multi audit)" at commit time.
- The launcher's settled ruling (working tree, `launch-custom-iq3s-8081.sh:73-87`): "**The NaN tripped on the tmpfs placement tier under host-memory pressure (M1 root-cause), not on the pack location**". The #879 guard itself: `generate.cpp:12559`, `:12818`, `:13066` ("non-finite logits (#879) in slot %d ... it ends").
- Precedent for this NaN CLASS on this lane: the G2-era boundary — batch x G2 x PCIe-share DMA device-alias/flagB corrupted, root cause not run to ground, banked as characterized (`SVG-CUSTOM.md:7741-7747`). This engine's non-finite failures are state/pressure-sensitive, not arithmetic-boundary failures.

### 5.2 The exact guard

The engine's `arm()` validates bytes and geometry but has **no memory-pressure guard**. What exists and what it misses:

| Check | Site | Behavior under pressure |
| --- | --- | --- |
| hugetlb reservation (the F4 "mapping's own reservation is the pool check") | `shadow_experts.cpp:180-182`, comment `:163-170` | All-or-nothing -ENOMEM -> **downgrades**, does not refuse |
| reservation-failure branch | `:205-215` | Sets `mfd_note` ("the swappable tmpfs tier serves every miss"), `view` stays `MAP_FAILED`, falls through |
| file-mapping fallback | `:225-228` | Arms on a 4 KiB swappable page-cache mapping |
| mlock | `:234-240` | Best-effort; refusal tolerated ("locked pages that later get reclaimed are worse than an honest 4K stall") — the store stays zram/swap-eligible exactly when the box is squeezed |
| `huge_pool_free_bytes()` / `huge_pool_covers()` | `include/strata/platform/hugepage.hpp:67-108` | EXIST, read `HugePages_Free - Rsvd` — after F4 they feed only the `mfd_note` string; no refusal consumes them |
| MemAvailable / pressure check | — | **Does not exist anywhere in `arm()`** |

Live proof of the failed state: all 39 armed banners in the log are reservation failures at "0.0 GiB free of 31.6 GiB needed" (`:1045` first, `:116100`, `:117847` (canon path), `:124394-124808` (the 11-18 census boots)). The F4 comment itself records the prior squeeze where the fallback "landed the store on the swappable 4 KiB tmpfs tier in the 11:05 squeeze (log:667335)" (`:165-170`) — the failure mode was observed, priced as a latency risk, and shipped anyway; the NaN census turned it into a correctness conviction.

Mechanism boundary (honest): identical bytes through a read-only mapping cannot become NaN by swapping; the swappable tier is the TRIPPED CONDITION (31.64 GiB of 4 KiB tmpfs + a failed 31.64 GiB mlock inside a squeezed box changes the memory state every other organ sees — the audit's named suspects are the armed path's own buffers: `ActQ` state and `run_split_multi`'s `split_multi_` SplitBufMulti array, `pool.cpp:769,966,1004,1089`), not the corruption mechanism itself. The deep audit stays open; the isolation (armed 29/29 vs unset 3/3) is what is proven.

### 5.3 Fix class — arm refuses under pressure

1. **Shipped (launcher/lib, the single owner)**: `campaign/lib/launch-common.sh:73-91` — canonical pack is THE method; tmpfs tier is explicit opt-in (`STRATA_SHADOW_TMPFS=1`); "the tier that corrupted under memory pressure ... is never automatic"; explicit env wins (including `""` rollback). Commits `e4dd2e3c4`, `c6bbc6ed5`, `89c00d737`; the dead 39.3 GiB /dev/shm copy deleted; hugepages tier "doctrine-off — the boot-armed pool was convicted 2026-11-09".
2. **Open (engine-side)**: in `arm()`, on `dst == MAP_FAILED` (reservation failure) do NOT fall through to the swappable tier — `return refuse(...)` so the lane takes the documented byte-identical native path. Concretely: consume `huge_pool_covers(want)` (`hugepage.hpp:85`) before ANY tier, and refuse when `MemAvailable - want` breaches a floor or when the mlock of the fallback mapping fails. One line of posture: **armed requires reserved-and-locked or nothing**. This restores the F4-removed pre-check as a refusal instead of a downgrade, which is exactly the fix class the census commit's rollback documents.

## 6. The boundary that would change the conclusion

- **Quant and environment landed together.** No GSQ battery exists between Nov 10 16:47 and Nov 18 13:48 (the timeseries study's gap table). The rollback (`385937a4d`, 17:22) and the box change (16 -> 8 visible cores, canonical store re-homed to read-only ntfs3, `945e747d0` 16:10) are coincident within hours. The 2.04 -> 4.4 doubling therefore attributes cleanly to the quant/tier law ONLY to the extent §3.2's arithmetic carries it (it does: 1.90-2.03x predicted vs 2.02x measured per-entry); the residual +0.1-0.8 ms and the adapt-wait +0.75 are the estate-move's share. The same-box shadow A/B (F-360: pair +7.7%, conc +8.4% end-to-end) prices the PURE shadow effect smaller than the headline doubling — the doubling is the rollback-priced TOTAL of leaving the shadow off ON TODAY'S BOX.
- The Oct 17 2.04 lines (native, 15 workers, ext4) sitting at the same cpu-exp as the Nov 10 bank (shadow, 15 workers, ext4) is visible in §3.3 and is the strongest single datum that the environment term is real: same native path, 58 -> 92 us/entry, entirely across the gap.
- If a future boot arms the shadow from the canonical pack on ntfs3 (the current lib default) and cpu-exp returns to ~2.0-2.5, the quant law is confirmed on today's box; if it stays >4, the environment (ntfs3 mmap service rate, 7 workers) dominates and the cost model's tier term needs re-fitting. The first canonical-armed boot has not flushed yet (no `/mnt/DATA` banner in the log as of this study).

## 7. File:line index

| Subject | Anchor |
| --- | --- |
| `cpu-exp` accounting (batch anatomy print) | `src/program/generate.cpp:12930-12944` |
| `ms_pool` accumulation | `src/core/verify.cpp:3238` (also 4043, 4230, 4328, 4629) |
| dispatch callback site | `src/core/verify.cpp:3231-3233`; `src/program/generate.cpp:1234` |
| dispatch + plan + counters | `src/core/expert_source.cpp:4570-5117` (plan 4636-4849; act quant 4893-4906; blob resolve 4985-5014; run 5038-5039, 5083-5084; misses 5114) |
| shadow blob pointer swap | `src/core/expert_source.cpp:4989-4991`; shadow gate 4601; prefetch skip 4924-4926 |
| native blob service order (arena -> override -> mmap) | `src/core/expert_source.cpp:4352-4390`; pinned gate 4392-4403 |
| pool kernels Q2_0 vs native | `src/kernels/cpu/pool.cpp:1064-1101` vs `:1103-1132`; buffers `:769,966,1004`; tasks `:1055-1061` |
| geometry constants (H/FF/NE/QK/BB/BLOB) | `include/strata/kernels/cpu/expert.hpp:44-51` |
| arm ladder + reservation failure + mlock | `src/core/shadow_experts.cpp:74-256` (gate 78-79; memfd 161-224; failure note 205-215; file mmap 225-228; mlock 234-240; banner 249-254) |
| pool free/cover probes | `include/strata/platform/hugepage.hpp:67-108` |
| #879 guard | `src/program/generate.cpp:12559, 12818, 13066` |
| round printer (shadow experts / resident RAM) | `src/program/generate.cpp:18272-18302` |
| falsifier cost law | `SVG-CUSTOM.md:11561-11569` (F-359); A/B ladder `:11589-11625` (F-360); minor-fault watch `:11727` (F-365); tmpfs tier `:11761` (F-367) |
| launcher cost law + census + rollback | `launch-custom-iq3s-8081.sh:69-90`; twin `launch-custom-ud-iq4xs-8081.sh:70-73` |
| lib ladder (canonical-primary, tmpfs opt-in) | `campaign/lib/launch-common.sh:73-91` |
| census commit / estate move / re-arm | `385937a4d`, `945e747d0`, `e4dd2e3c4`+`c6bbc6ed5`+`89c00d737` |
| era lines | bank `logs/custom-iq3_s.log:1037`; rollback `:128123`, `:129873`, `:131623`; reservation-failure banners `:1045` ... `:124808`; workers 15 -> 7 `:104226`/`:105991` vs `:112633`/`:131665` |
| sibling studies | `.docs/research/2026-11-18-cpu-exp-timeseries.md` (onset segmentation), `.docs/research/2026-11-18-live-telemetry-decomposition.md` (worker-count confound) |
