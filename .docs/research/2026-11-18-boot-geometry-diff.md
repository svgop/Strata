---
type: research-study
date: 2026-11-18
subject: GSQ lane boot-time geometry diff — bank era (2026-11-10, anchors 219.6/210.4/290.3) vs T66 warm recheck (2026-11-18, 169.1/169.0/225.2)
repo: /home/svg/LLM/strata-custom-2 (HEAD 1cdd2851b at write time; bank-era ref 8aa633710)
status: complete
---

# GSQ boot geometry diff: bank era (Nov 10) vs T66 (Nov 18)

## 0. Verdict first

Eight geometry fields changed between the bank era and T66; the rest of the armed geometry is byte-identical. The two load-bearing changes are (1) the **shadow store DISARMED** (`STRATA_SHADOW_EXPERTS=""`, the 2026-11-18 NaN rollback, commit 385937a4d — bank era: armed, 31.64 GiB, 48 layers x 512 experts x 1382400 B) and (2) the **CPU pin regime** (bank-era boots ran pool workers on logical 9-15 + host on 8 + aux on 24 under the isolation stack with `STRATA_ENGINE_CPUS=8-15,24-31`; T66 boots run workers on 1-7 + host on 0 + aux on 16, plain boot, no isolation cmdline). Secondary changes: expert cache 10142 -> 9914-9951 slots (-2.0%, -0.38 GiB VRAM), page-locked complement 37.40 -> 37.76-37.83 GiB (+0.38 GiB — same fixed ~55 GB split, cache/complement boundary moved), boot-time RAM +4.4-5.1 GiB more resident (MemAvailable 138.6-138.9 -> 133.6-133.9 GiB), the estate re-homed onto /mnt/DATA read-only NTFS, `--mtp-draft-vocab` added, and the prewarm sweep composition changed. KV, elastic arenas, slot sessions, PLE, cvec, spec windows, MTP drafters, prefill plan, conversation cache, PCIe frac, pool worker count (7), and engine version (0.1.46) are UNCHANGED.

The bank era's own banner proves the shadow tier was **never pool-reserved** in any recorded bank boot ("the pool could not reserve the store (0.0 GiB free of 31.6 GiB needed) — the swappable tmpfs tier serves every miss"): the bank's shadow advantage ran off tmpfs/pack placement, not hugetlb memfd. Losing it is a measured -3.5 to -5.5 pct per lane (commit 9e889bda3, F-367 adoption gains in reverse) — roughly a fifth to a quarter of the -19.7 to -23.0 pct lane regression T66 shows. The rest of the gap is not explained by boot geometry that the banner exposes; the T67 stock head-to-head (upstream v0.2.15 at 146.6/150.5/258.4 on this box the same evening) independently shows the box state itself is slower today than the bank era's.

## 1. Evidence set

Bank era (the newest iq3s-lane boots of 2026-11-10; no Nov 11-12 boots exist — nothing ran those days):

| Leg | .out | Engine boot (logs/custom-iq3_s.log) | Notes |
| --- | --- | --- | --- |
| refactor-warm-test | campaign/results/refactor-warm-test.out | 00:30:42, line 112606 | guard-era boot |
| iso-legA | campaign/results/iso-legA.out | 00:34:42, line 114352 | isolation-stack bench leg |
| iso-legB | campaign/results/iso-legB.out | (shares window; log line ~114352-116098) | bench leg |
| iso-legC | campaign/results/iso-legC.out | 00:38:04, line 116098 | bench leg |
| postreboot-warm-1 | campaign/results/postreboot-warm-1.out | 13:37:59, line 117845 | **newest bank-era boot**, single-owner shadow arm |

- The last bank-era commit is 8aa633710 ("GSQ post-reboot warm bench on the isolation stack, last-3 means 227.6/210.1/293.9"); the bank anchors 219.6/210.4/290.3 and record class 227.6/210.1/293.9 are carried in campaign/batteries/t66-warm-recheck.sh:4-9.
- t63-*/t42-*/t44-*/t49-* .out files are the iq3s lane but from Nov 2-3 (older geometry, pre-bank); t31-base ran 131072 ctx / 9396 experts; tqllm is Oct 16. All excluded by the "newest from that era" rule — the bank-era geometry is uniform across every Nov 9-10 boot checked (incl. v6 legs, Nov 9 19:05, log line 81200).
- probe-metrics.py of the bank era scraped only vLLM histograms/gauges (campaign/results/metrics.jsonl, 201 records back to Oct 21, zero engine-section fields) — no bank-era /metrics "engine" capture exists. Bank-era /metrics-equivalent values below are taken from the boot banner, which is richer.

Today (T66, fresh boots, same launcher launch-custom-iq3s-8081.sh):

| Leg | .out | Engine boot | Notes |
| --- | --- | --- | --- |
| t66-warm-r1 | campaign/results/t66-warm-r1.out | 17:49:30, line 126362 | |
| t66-warm-r2 | campaign/results/t66-warm-r2.out | 17:52:18, line 128140 | |
| t66-warm-r3 | campaign/results/t66-warm-r3.out | 17:54:57, line 129892 | |
| live | (running engine) | 18:12:48, line 131639 | /metrics engine section read live during this study |

Lane result context: bank anchors 219.6/210.4/290.3 vs T66 169.1/169.0/225.2 (M1 solo / M2 pair / M2 6-conc, 3-boot means) = -23.0 / -19.7 / -22.4 pct.

## 2. Field-by-field diff table

Bank column = range across bank-era boots (newest = postreboot-warm-1 where a single value is quoted). Today column = range across t66-warm-r1/r2/r3 (+ live /metrics where noted). "CHANGED" flags every field that moved.

| Field | Bank era (2026-11-10) | Today (2026-11-18 T66) | Delta | Changed |
| --- | --- | --- | --- | --- |
| **Shadow store (ARM state)** | **ARMED** — launcher prints `launcher: shadow store /dev/shm/strata-shadow-q2_0` (iso legs) / `.../packs/q2_0` (postreboot, single-owner arm); engine: `STRATA_SHADOW_EXPERTS=...: shadow Q2_0 expert store mapped — 48 layers x 512 experts x 1382400 B = 31.64 GiB` | **OFF** — no launcher line, no engine line; launcher ships `export STRATA_SHADOW_EXPERTS=""` (NaN rollback, 385937a4d) | disarmed | **CHANGED** |
| Shadow pool reservation | NOT reserved in every bank boot: "the pool could not reserve the store (0.0 GiB free of 31.6 GiB needed) — the swappable tmpfs tier serves every miss" | n/a (store gone) | tier removed | **CHANGED** |
| Expert cache slots (boot) | 10006 / 10109 / 10142 (newest 10142) | 9914 / 9951 / 9951 | -170 to -228 (-1.7 to -2.2%) | **CHANGED** |
| Expert cache VRAM | 18.99 / 19.18 / 19.24 GiB | 18.81 / 18.88 / 18.88 GiB | -0.18 to -0.43 GiB | **CHANGED** |
| Expert cache auto first-pass | 21.08-21.34 GiB free -> 7657-7761 slots | 20.90-20.97 GiB free -> 7586-7615 slots | -0.4 GiB free | **CHANGED** |
| Cache policy | PROFILE, ranked by routing frequency, no eviction; fill-ahead 256; pre-filled 10006-10142/10142 in 2.6 s (7955 MB/s) | identical policy; pre-filled 9914-9951 in 3.6 s (5539 MB/s) | fill 2.6 -> 3.6 s, 7955 -> 5539 MB/s (estate on NTFS) | **CHANGED** (rate only) |
| Pool lock state ("loading the experts ... and locking part of them") | Wrapper line identical both eras. Locked = the FULL cache complement, page-locked and mapped: resident 37.40-37.72 GiB == pinned 37.40-37.72 GiB (newest boot 37.40) | same semantics: resident 37.76-37.83 GiB == pinned 37.76-37.83 GiB | +0.36 to +0.43 GiB locked | **CHANGED** (size only; "part of them" = everything the GPU cache does not hold, both eras) |
| Complement copy time | 10 s copy / 14-15 s ready (postreboot+isoA); 17 s (v6 era) | 12 s copy / 20 s ready (all 3 legs) | +5 s page-lock | **CHANGED** (rate) |
| Pre-release free RAM ("really free" after file-page release) | 1.02 GiB (postreboot) / 28.80 GiB (isoA); 51.56-79.96 GiB post-release | 15.76 GiB; 67.07 GiB post-release | inside the bank-era range | unchanged |
| Lendable prompt-path slots | 5182 (postreboot) / 5185-5187 | 5187-5189 | +0 to +7 | unchanged (noise) |
| Prompt-path VRAM borrow | 5182-5186 slots = 9.80 GiB | 5187-5189 slots = 9.80 GiB | same bytes | unchanged |
| KV type / resident | int8, `--kv-resident 32768`; 32768 of 262144 cells per QSA layer in VRAM, K/V 3.09 GiB pinned RAM | identical | 0 | unchanged |
| KV host shadow (hugepages line) | absent in bank-era boots (present only in the Nov 9 16:49 boot, 0.26 GiB THP) | absent | 0 | unchanged |
| Elastic slot arenas | 6 x 0.97 GiB reserved, 2 mapped at boot, 4 registered-but-unmapped, 2.91 GiB freed in 9-10 ms, chunk 32 MiB, spare 31 chunks | identical (9 ms) | 0 | unchanged |
| Batch slot sessions | 6 x 0.95 GiB = 5.69 GiB on CUDA0; 20.03-20.09 GiB free at that point | 6 x 0.95 GiB = 5.69 GiB; 19.66-19.73 GiB free | -0.37 GiB free VRAM | **CHANGED** (free headroom) |
| MTP | draft layer 874 MiB VRAM (experts 675, dense 111); `--mtp-q4` 10 projections 37.3 MiB; 6 slot drafters x 51 MiB; `--spec 6`, spec-min-p 0.5 | identical (read rate 2178 -> 2642-2682 MiB/s off NVMe) | flags identical | unchanged |
| `--mtp-draft-vocab` | absent from the bank-era command line | `data/draft_vocab_en.bin` added (385937a4d) | new flag | **CHANGED** |
| Spec verify windows | `window up to 8 tokens, 128.2 MiB of device buffers`; 8-token capture at 1450-1490 MiB VRAM free | identical 8 tokens / 128.2 MiB; capture at 1485-1525 MiB free | +35 MiB free | unchanged (headroom noise) |
| PLE | on, table 320001536 rows | identical | 0 | unchanged |
| Control vector | project, per-layer, layers 1-47 (47 steered) | identical; live /metrics `cvec: "project:1-47", cvec_steered "47/48"` | 0 | unchanged |
| Prefill plan | chunk 32768, ring 96; file set 0.4 GiB; page cache 43.8 GiB (postreboot) / 74.3 GiB (isoA); drive **1.50 GB/s (assumed)**; PCIe 12.0 GB/s | chunk 32768, ring 96; file set 0.4 GiB; page cache 68.0-69.1 GiB; drive **2.50 GB/s (nvme2n1 is NVMe)**; PCIe 12.0 GB/s | drive now measured NVMe | **CHANGED** (detection, not geometry) |
| Conversation cache | 49152 MiB, 24 slots, min-tokens 200 | identical (live /metrics 49152/24/200) | 0 | unchanged |
| Conversation slot dir | `/home/svg/LLM/data/conversation-slots` | `/home/svg/LLM/data-rt/conversation-slots` (write path off RO mount) | re-homed | **CHANGED** |
| PCIe share | `--pcie-frac 0.35 --pcie-mode dma` | identical (live /metrics pcie_frac 0.35) | 0 | unchanged |
| VRAM reserve | `--vram-reserve-mib 2000` (+141 MiB draft head reserved in cache sizing) | identical | 0 | unchanged |
| CPU pool workers | **7 workers** (8 physical / 16 logical, bounded by every physical core but the host's); tasks/phase 24, participating 8 | **7 workers**; tasks/phase 24, participating 8 | count same | unchanged |
| Pool pin map | workers on logical **9,10,11,12,13,14,15**, host thread on **8**, aux CPU **24** — the isolation stack regime (`STRATA_ENGINE_CPUS=8-15,24-31` + isolcpus/nohz_full 8-15+24-31, irqaffinity 0-7+16-23, engine under swap.max=0 scope) | workers on logical **1,2,3,4,5,6,7**, host thread on **0**, aux CPU **16** — plain boot, pin env OFF, no isolation cmdline | **CHANGED** |
| Memory guard | limit 184.2-185.8 GiB; current 45.5-46.9 GiB; available 138.6-138.9 GiB | limit 184.2 GiB; current 50.3-50.6 GiB; available 133.6-133.9 GiB | +4.4-5.1 GiB resident at boot, -5 GiB available | **CHANGED** |
| Prewarm sweep | 17 files / 80.1 GiB (iso legs); **18 files / 111.7 GiB** (postreboot — includes the 31.6 GiB q2_0 shadow pack) | 18 files / 80.1 GiB | shadow pack out of the sweep; draft_vocab in | **CHANGED** (composition) |
| Model/pack paths | `/home/svg/LLM/data/...` (GSQ-RCO/Qwen3.8-Flash-Next/) | `/mnt/DATA/LLM/Models/GGUF/...` (GSQ-RCO/qwen3.8-flash-next/gsq-iq3s/), read-only NTFS | estate move 945e747d0 | **CHANGED** |
| Engine version | 0.1.46 | 0.1.46 | 0 | unchanged |
| Context | 262144 tokens, images on, API key required | identical | 0 | unchanged |

## 3. /metrics "engine" section — today's live values vs bank-era banner equivalents

probe-metrics.py did not scrape the engine section in the bank era (verified: metrics.jsonl carries histogram/gauge records only), so the bank column is the boot-banner equivalent, not a like-for-like endpoint read.

| Field | Today live (2026-11-18, task-provided / re-read this study) | Bank era equivalent (banner) | Same or changed |
| --- | --- | --- | --- |
| expert_slots | 9966 (task sample) / 9968 (re-read) | 10006-10142 boot slots | **changed** (-2.0 pct at boot; live value rides the adaptive tier +17) |
| expert_cache_mib | 19361 / 19364 | 19438-19702 (18.99-19.24 GiB) | **changed** (-0.4 GiB) |
| spec | 8 (verify window; STRATA_ADAPTIVE_SPEC=1) | 8 (`window up to 8 tokens`) | same |
| mtp_max | 6 | 6 (6 slot drafters, `--spec 6`) | same |
| lookup | 3 | no bank-era capture; flags + engine version identical -> presumed 3 | presumed same (unprovable from Nov-10 artifacts) |
| vram_free_mib | 1230 / 1232 | closest banner analog: 1450-1525 MiB free at the 8-token verify capture | different sampling point; not comparable |
| cvec | project:1-47 | project, layers 1-47, 47 steered | same |
| arena_mib | 38639 / 38637 (== the pinned complement, 37.73 GiB) | complement 37.40-37.72 GiB pinned (38298-38588 MiB) | **changed** (+~0.3 GiB, tracks the complement growth) |
| pool_workers | 7 | 7 | same (pin map changed, see section 2) |
| pcie_frac | 0.35 | 0.35 (`--pcie-frac 0.35`) | same |
| conversation_cache_mib | 49152 | 49152 | same |
| batch_slots | 6 | 6 (5.69 GiB) | same |
| prefill_cap | 32768 | 32768 (auto:32768, ring 96) | same |
| kv / kv_resident | int8 / 32768 | int8 / 32768 | same |

## 4. Reading the diff against the lane regression

1. Shadow store off: the only armed-vs-unarmed flip. Its measured cost is known and bounded (F-367 adoption read M1 +5.5 / pair +4.2 / conc +3.5): ~5 of the ~20-23 missing points. Also note the bank-era shadow ran pool-UNreserved (tmpfs tier) in every recorded boot, so the bank's advantage was already the weak form of the feature.
2. Pin regime: bank-era boots carried the full isolation stack (engine pinned 8-15+24-31, isolcpus/nohz_full, irqaffinity split, swap.max=0 scope) — the tracker's "structural quiet class" (GPU-reach wait 0.40-0.63 ms in all five bank probes). T66 boots run the plain map (workers 1-7, host 0). The v6 verdict (901f84095) priced taskset pinning neutral on a quiet box, but the bank boots were also booted INTO the isolation cmdline — that boot-state delta is not reproducible from the launcher alone.
3. Cache/complement boundary: the ~55 GB expert mass split moved by +-0.4 GiB toward the complement (-2 pct VRAM cache). Second-order; the adaptive tier and 98 pct hit rates are intact today.
4. Box state: boot-time RAM +5 GiB resident, MemAvailable -5 GiB; complement page-lock +5 s. T67 (stock v0.2.15, same box, same evening) reads 146.6/150.5/258.4 — an independent witness that today's box state is below the bank era's across engines.
5. The T66 judge band (solo >= 210, pair >= 200, conc >= 280) was NOT met: 169.1/169.0/225.2. Boot geometry alone does not close the gap; the named suspects outside the banner are the box boot state (no isolation cmdline, +5 GiB resident) and whatever remains of the NaN-rollback surface beyond the shadow flag.

## 5. Provenance

- Engine log boots: /home/svg/LLM/strata-custom-2/logs/custom-iq3_s.log lines 81200 (Nov 9 19:05), 112606/114352/116098 (Nov 10 00:30-00:38), 117845 (Nov 10 13:37), 126362/128140/129892 (Nov 18 17:49-17:55), 131639 (Nov 18 18:12, live).
- .out banners: campaign/results/{refactor-warm-test,iso-legA,iso-legB,iso-legC,postreboot-warm-1,t66-warm-r1,t66-warm-r2,t66-warm-r3}.out.
- Launcher diff: `git diff 8aa633710 HEAD -- launch-custom-iq3s-8081.sh` (3 hunks: shadow disarm, conv-slot re-home, path re-home). Config diff: `git diff 8aa633710 HEAD -- config/custom-iq3_s-8081.json` (paths + `--mtp-draft-vocab`).
- Live /metrics read: 2026-11-18 ~18:20 SAST, engine up (T67 boot), engine section quoted verbatim in section 3.
- Bank anchors and judge band: campaign/batteries/t66-warm-recheck.sh:4-9.
