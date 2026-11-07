# The no-reboot host-memory architecture

**Status:** design, research-validated 2026-11-07. Owner: this file + the move owners below.
Evidence: three digests at `.docs/research/digest/2026-11-07-hostmem-{inference-servers,kernel-state,no-reboot-precedents}.md`
(verified against upstream source and kernel docs; every claim carries a citation and confidence grade there).
Gates: law 17 on every move (before/after, same aged box, same-day interleave), law 13 (no lane worse),
law 16 (zero capability change: full 262144 ctx, packs, vision, spec, cache — untouched; this architecture
changes page size, not features).

## The problem, convicted

The CPU-expert lane is DRAM-bandwidth-bound and TLB-sensitive (`pinned.cu:262`: a 3 MB expert on 4 KB pages
costs ~750 TLB misses). After days of uptime the box cannot supply 2 MiB pages anymore, and every hot pool
silently lands on 4 KB. The engine's own boot log records the supply collapsing across the week
(free order-9 blocks for the arena): **18 → 16 → 9 → 0 GiB**, with `MAP_HUGETLB` dead since forever
(`vm.nr_hugepages=0`) and the #2126 guard correctly refusing to stall loads fighting a fragmented buddy
allocator — serving 4 KB instead. Kernel side: `compact_fail` 1.18M of 1.66M stalls (71%),
`AnonHugePages` 120 MB of 133 GiB resident, zram refilling within days of being cleared.

Three organs, one disease:

| organ | pack(s) | mass | backing today | why it degrades |
|---|---|---|---|---|
| expert arena | UD-IQ2_XS / UD-IQ4_XS | 39 / 56 GiB | anon, 2 MiB-aligned, `MADV_HUGEPAGE` only when free blocks ≥ half the arena (#2126) | guard skips on aged box → 4 KB |
| pinned cache complement | GSQ IQ3_S | 37.61 GiB page-locked | `cudaHostAlloc`-class pinned 4 KB | pinned pages are unmovable: no collapse, no compaction — 4 KB by construction, and each boot's pin/unpin churns ~10M pages through the buddy allocator |
| shadow store | all Qwen lanes | 38 GiB on `/dev/shm` | tmpfs shmem, `shmem_enabled=never` → zero huge folios ever (`SHADOW_STORE.md:34`'s "THP-eligible" is false on this box) | 4 KB shmem, reclaimable → also the zram-refill source |

A fresh boot wins by accident (pristine buddy lists). This architecture makes it win by construction.

## The research verdict (three digests, condensed)

1. **Reservation/pinning precedes fragmentation** (PostgreSQL `huge_pages`, Oracle SGA, DPDK): the pool is
   armed at boot and exempt from the kernel lifecycle — cannot swap, migrate, or fragment. Nobody re-acquires
   a fragmented pool online (`compact_memory` is a synchronous stall; PG docs: "repeat the command or reboot").
2. **Engineered huge pages on the named pool, never global policy**: vLLM (HugeTLB + `MADV_HUGEPAGE`) and
   SGLang (`SGLANG_HUGEPAGE_SIZE`, THP-aligned prefault) built exactly this; global `THP=always` is banned by
   the DB class and ranked last for us too (only touches the small anon side, imports stalls onto a zram-full box).
3. **A silent 4 KB fallback is the bug that ships** (PG fail-fast, Oracle alerts, DPDK fatal-at-init). Our
   #2126 fallback is logged but silent in effect — it becomes loud (banner + metric + strict mode).
4. **Online repair is real but glue, not architecture**: `MADV_COLLAPSE` (kernel 6.1+, shmem + anon, PMD-size,
   ignores enabled settings) + `compact_memory` + `vm.compaction_proactiveness` restore supply; and every
   engine restart re-faults the whole store — with huge folios enabled on the mount, **the daily restart cycle
   becomes a defrag event instead of a fragmentation source**.
5. **Nobody mature churns multi-GiB pools per load** (TRT-LLM: allocate once at boot; ollama keep_alive;
   SGLang wrote its own pinned allocator to avoid power-of-two rounding). Our shared-arena machinery
   (`pinned.cu:46-57`, pack-hash-keyed `MAP_SHARED`) is built for this and has zero callers.
6. Upstream llama.cpp has **no** hugepage work on x86 and its issue #22360 reports restart-to-fix slowdowns
   with the root cause unconfirmed — the exact disease here, diagnosed and fixed ahead of them.

## The architecture — arm what the tree already built, plus glue

Ordered by leverage-over-effort; every move independently reversible and law-17 gated.

**M1 — the shadow store on huge folios (config-level, no engine change).**
A dedicated tmpfs mounted `-o size=48G,huge=within_size` (unit: `campaign/strata-huge-tmpfs.service`, mount
`/mnt/strata-huge`; `within_size` suits the written-once store). Owner: `campaign/shm-shadow.sh` (DST retarget)
+ `launch-custom-iq3s-8081.sh` (+ UD launchers) tier-0 resolution + `STRATA_SHADOW_EXPERTS` env. Fallback tiers
unchanged. Witness: `ShmemPmdMapped` ≈ 38 GiB in /proc/meminfo after the copy; store miss-path µs.

**M2 — supply-side sysctls (one line, root).** `vm.compaction_proactiveness=20→40` via
`/etc/sysctl.d/90-strata-vm.conf` (digest-3 range 40-50; watch kcompactd CPU). This raises the free-2-MiB-block
supply the #2126 guard checks, so fresh boots and restarts pass it more often. Witness: the guard's
"NOT requested" banner disappearing from boots over time.

**M3 — huge-page the two engine organs (env + small code).**
- UD arena: with M1+M2 raising supply, the existing `MADV_HUGEPAGE` path engages at fault time.
- GSQ complement: arm `STRATA_HUGEPAGE_ARENAS=1` — the T35 C2 branch (`expert_source.cpp:3541-3721`,
  `layer.cpp:596`) already switches the complement and KV shadow onto the `pinned.cu` reserve() path
  (hugetlb → THP → guarded 4 KB). Locked pages cannot be collapsed later, so the huge pages must be
  faulted in **before** `cudaHostRegister` — reserve() does exactly this order. Named risk, empirically
  decided by the leg: whether pinning splits THP-backed regions (witness: `AnonHugePages` before/after
  register, lanes).
- Loudness: the #2126 4 KB fallback gets a standing banner (once per boot, not a log line) and
  `STRATA_ARENA_PAGES=strict` for benches (exit non-zero when the hot pool is not 2 MiB-backed) — default
  stays serve-degraded (law 3: no pretend, but no refusal either).
- Post-load `MADV_COLLAPSE` pass over the store mapping (`shadow_experts.cpp` arm()) — converts "fresh-boot
  class" into "every-boot class" for the shmem organ.

**M4 — cgroup v2 protection for the engine (launcher change).** Launchers exec the engine under
`systemd-run --scope` with `MemoryLow=` (stop kswapd/direct-reclaim scan pressure — 5.3M pgscan events on this
box) and `MemorySwapMax=0` (hard-opt the engine out of zram while zram keeps serving the desktop). Kills the
zram-refill incident class at the source. Starting value `MemoryLow=120G`, tuned down if desktop pressure
appears. Owner: the three launchers.

**M5 — the endgame, decision-gated: a boot-reserved hugetlb core.** The engine's `MAP_HUGETLB` path is already
written and engages alone once `nr_hugepages` covers the arena; `/dev/hugepages` is already mounted. Static
cost (the standing lane's arena ~38-56 GiB + store 38 GiB ≈ the 56320-page class the DS-era sizing note
already carried), armed at boot (never runtime on a fragmented box — silently partial per kernel docs, never
surplus overcommit). Adopted **only if M1-M4 leave a measured residual** in the aging battery. Under the same
gate: arm-or-delete the dead shared-arena machinery, and the mmap-not-anon direction for staging
(digest-2's ollama invariant: file-backed beats anon for CPU offload).

## Sizing (184 GiB box)

| mass | GiB | notes |
|---|---|---|
| GSQ pinned complement | 37.6 | + 5184 prompt-path slots |
| UD arena (IQ2_XS / IQ4_XS) | 39 / 56 | largest standing lane governs M5 sizing |
| shadow store | 38 | on the huge= mount (M1) |
| KV/slot arenas | ~3-6 | ELASTIC_SESSIONS, unchanged |
| page cache + OS + desktop | remainder | packs stream from disk; zram stays for non-engine |

## The battery (law 17, aged box, no reboot needed for the verdict)

- **Leg 0 (before):** GSQ + UD-IQ2_XS current-config legs, same-day, on this 6-day box.
- **Leg 1:** M1+M2 armed (operator root block) → interleave the same two packs. Acceptance: lanes recover
  toward the fresh-boot class (mechanism witnesses: `ShmemPmdMapped` ≈ 38 GiB, guard banner gone, zram flat,
  `compact_fail` delta per restart shrinking).
- **Leg 2:** M3 (env arms + collapse pass) → interleave. Leg 3: M4 → interleave. Leg 4: M5 only if residual.
- **Exit ratchet (the aging battery):** N boot/stop cycles, lanes at cycle 1 vs cycle N, within noise.
  "No reboot needed" is claimed only after this ratchet holds; the chart gets the aged-box retention numbers.

## Non-goals

Global `THP=always` (last place in every digest). Runtime `nr_hugepages` growth on a fragmented box.
Removing the #2126 guard (it is correct engineering — the *supply* changes, not the guard). Any capability
trade (law 16). Deleting working machinery without a verdict (the shared arena gets arm-or-delete, not drift).
