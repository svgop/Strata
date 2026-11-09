---
type: digest
id: 2026-11-09-refactor-docs-drift
status: complete
created: 2026-11-09
owner: docs drift audit (read-only sweep; fixes are line-level edits listed per file)
scope: README.md + 6 translations, AGENTS.md, SVG-CUSTOM{,-FEATURES,-TODO}.md, docs/*.md (every file swept by grep for hazard classes + full read of primary docs), campaign/*.md, campaign/tracker/README.md, campaign/ROADMAP-*.md, docs/media/, .docs/
ground-truth: engine/snapshots ABSENT, campaign/91-strata-hugepages.conf + 90-strata-vm.conf ABSENT (git 09899c72), campaign/results/ds-pool-topup-block.sh PRESENT, strata-gpu-lock.service the only enabled unit, DEFAULT_PORT=8080 (strata_cli/launch/settings.py:30) / estate launchers 8081 (config/*.json "port": 8081), spec controller WIRED (src/program/generate.cpp:86 includes strata/spec/controller.hpp)
---

# Docs drift audit — 2026-11-09 (post standing-hugepages closure)

Doctrine reference (the two docs updated today, both consistent with the tree):
`docs/HUGEPAGES.md` § Arming and `docs/PERFORMANCE-SETUP.md` §4. Everything below is measured
against them + the live tree. Every fix is a doc-only edit = SAFE-NOW unless marked otherwise.
Known system-state landmine (NOT doc drift, named in commit 09899c72): the RUNNING boot's
`/proc/cmdline` still carries `hugepages=54272`; the strip command is already documented in both
updated docs.

## H1 — LIVE HAZARD: surviving boot-arming instructions

### F1. campaign/tracker/README.md:27 — teaches boot-time hugepage reservation as a queued lever
Offending (line 27, item 4 of "The next lever queue"):
> `4. Boot-time hugepages (vm.nr_hugepages≈56320) for the 109.5 GB store — needs a reboot to reserve.`

Corrected:
> `4. RETIRED 2026-11-09: standing/boot-time hugepages were convicted (−12–15% here, −20–22% stock; the conf is deleted). The lever survives per-session only: sudo bash campaign/results/ds-pool-topup-block.sh (runtime arm → READ BACK → serve → release), see docs/HUGEPAGES.md § Arming.`

Hazard class: this file is the tracker's protocol owner and is read before every lane point;
a future session could re-install a standing pool from this line. (The number is also wrong:
the deleted conf carried 54272 pages = 106 GiB, sized to the 110.85 GB DS draw + 2 GiB slack.)

### F2. AGENTS.md:432-434 — names "hugepages" as a DS lever without the per-session qualifier
Offending:
> `the DeepSeek lever queue (T53+: DS4_CACHE_THREADS, CACHE_MARGIN, dspark-conf, hugepages) is maintained in the tracker README`

Corrected:
> `the DeepSeek lever queue (T53+: DS4_CACHE_THREADS, CACHE_MARGIN, dspark-conf, the per-session DS pool top-up) is maintained in the tracker README`

## H2 — Internal contradictions BETWEEN docs

### F3. docs/SHADOW_STORE.md vs docs/HUGEPAGES.md + the engine (worst class: two backings taught)
The engine's shadow store is now **memfd-first** (`src/core/shadow_experts.cpp:163-173`,
"F4 (2026-11-09 bootstorm): ATTEMPT the memfd arm first", `memfd_create(MFD_HUGETLB)`,
`STRATA_SHADOW_MLOCK` at line 230ff; HUGEPAGES.md organ table: "hugetlb-backed memfd ...
falls back to the tmpfs/disk tiers"). SHADOW_STORE.md (dated 2026-10-27/29) teaches tmpfs
placement as THE architecture and carries a claim HOSTMEM-NO-REBOOT.md:29 already convicted:

- Offending (line 3): `**Status:** architecture, shipped and standing on the Qwen lane.`
  Corrected: `**Status:** shipped; backing updated 2026-11-09 — hugetlb memfd first (pool-covered boots), /dev/shm placement and canonical file as fallback tiers; owner src/core/shadow_experts.cpp. Pre-memfd text below is the 2026-10-27 adoption record.`
- Offending (line 34): `RAM-resident pages are THP-eligible and never evicted, so the pool pays no page faults (F-365: ...)`
  Corrected: `the tmpfs tier's pages were believed THP-eligible (F-365); measured later on this box: shmem huge folios are dead here (shmem_enabled=never — zero promotions, HOSTMEM-NO-REBOOT.md organ table). The memfd tier is the fix: 2 MiB pool pages by construction.`
- Offending (lines 36-38): ``/dev/shm` is tmpfs and is **wiped on every reboot**. `campaign/shm-shadow.sh` rebuilds the placement cache (12 s for 37 GB) and is a no-op when the copy is intact; `launch-custom-iq3s-8081.sh` calls it before every start`
  Corrected: `the launcher's tier-0 calls `campaign/shm-shadow.sh` only when `HugePages_Free` < 17000 (launch-custom-iq3s-8081.sh:114) — i.e. when the pool cannot cover the memfd arm; with the pool present the memfd is the store and the copy is not needed.`
- Missing: no mention of `STRATA_SHADOW_MLOCK` (mlock of the mapping; PERFORMANCE-SETUP.md documents it). Add one line under Rollback: `STRATA_SHADOW_MLOCK=0 — unmap-lock disabled`.

### F4. AGENTS.md:274-278 vs campaign/README.md:35-38 (the authoritative map) and the tree
Offending (AGENTS.md):
> `The standing boot pair (`launch-custom-iq3s-8081.sh`, `custom-iq3_s-8081.json`, `refusal-r.gguf`) stays at REPO ROOT — the launcher resolves `.venv` and the config relative to itself (LAUNCH LAW + F-249).`

Ground truth: `custom-iq3_s-8081.json` lives at `config/custom-iq3_s-8081.json` (launcher line
123: `--config "$PWD/config/custom-iq3_s-8081.json"`); `refusal-r.gguf` exists NOWHERE in-tree
(F-212 purged it; the vector is at `/home/svg/LLM/data/vectors/qwen3.8-flash-next-refusal-r.gguf`).
campaign/README.md:35-38 already teaches the correct form — AGENTS.md contradicts it.
Corrected (AGENTS.md):
> `The standing boot triple (`launch-custom-iq3s-8081.sh` at REPO ROOT by LAUNCH LAW; its config at `config/custom-iq3_s-8081.json`; the steering vector in the estate at `/home/svg/LLM/data/vectors/qwen3.8-flash-next-refusal-r.gguf`) — the launcher resolves `.venv`, the config path and the vector relative to itself (LAUNCH LAW + F-249; F-212 purged the root copy of the vector).`

### F5. campaign/tracker/README.md:28-29 vs SVG-CUSTOM-FEATURES.md §6 — W1 execution state
Offending (tracker README, item 5):
> `5. Qwen side: the W-queue (campaign/ROADMAP-perf-8moves.md) — W1 adaptive spec and W9 S26 expert envs are the unexecuted high-ROI rows.`

Ground truth: W1 ADOPTED 2026-11-02 (SVG-CUSTOM-FEATURES.md:181-187 "The adaptive-speculation
controller (W1, adopted 2026-11-02)... now sizes the serial verify window";
`src/program/generate.cpp:86` includes `strata/spec/controller.hpp`).
Corrected:
> `5. Qwen side: the W-queue (campaign/ROADMAP-perf-8moves.md) — W1 adaptive spec is EXECUTED (adopted 2026-11-02, FEATURES §6); W9 S26 expert envs remain the unexecuted high-ROI row.`

### F6. campaign/ROADMAP-perf-8moves.md:56-62 — W1 premise paragraph now false
Offending:
> `A COMPLETE cost-model controller is dead code: `Controller::choose`/`CostModel::step_ms` (src/spec/controller.cpp:40–79) referenced only by controller_test.cpp.`

Corrected (insert a verdict banner under the W1 heading, keep the original premise as history):
> `**VERDICT 2026-11-02: ADOPTED.** The "dead code" premise below was the trigger — the controller is now the live chooser (`src/program/generate.cpp` includes `strata/spec/controller.hpp`; cut-only after min_p, before the keep rule; battery-neutral on the probe, the +35-38% class is live deep-context traffic). Premise text preserved below as the 2026-10-27 read.`

### F7. campaign/HOSTMEM-NO-REBOOT.md:60-64 (M1) vs docs/HUGEPAGES.md ("no mounts, no services")
M2 and M5 got tombstones in 09899c72; M1's hugetlbfs-mount route did not, and it asserts a
service that exists neither in tree nor systemd:
> Offending (line 61): `The store therefore lives on **hugetlbfs** (`campaign/strata-huge-store.service`, `/mnt/strata-huge`, uid/gid=svg, `size=40G`): 2 MiB pages **by construction**`

Ground truth: `campaign/strata-huge-store.service` ABSENT; unit not installed; the shipped
backing is the in-process memfd (HUGEPAGES.md: "The store lives as a hugetlb-backed memfd owned
by the process itself — no mounts, no services").
Corrected: `*(M1 superseded before install: the shipped form is the in-process hugetlb memfd — no mount, no service; see `docs/HUGEPAGES.md`.)* ~~The store therefore lives on **hugetlbfs** (`campaign/strata-huge-store.service`, `/mnt/strata-huge`, uid/gid=svg, `size=40G`)~~: 2 MiB pages **by construction**`

## H3 — Dead artifact references in live docs

### F8. campaign/README.md:22 — batteries/ lists a harness deleted 2026-11-04
Offending:
> `- `abab-run.sh`            — the generic ABAB interleave battery runner`

Ground truth: deleted in 776e5739 ("three dead harnesses removed": abab-run.sh,
t43/t46-ternary-battery.sh; it called campaign/battery-boot.sh, itself deleted in F-268).
The live ABAB runners are `campaign/batteries/t38-abab-b.sh` + `era-interleave.sh` (in the dir).
Corrected:
> `- `t38-abab-b.sh`, `era-interleave.sh` — the ABAB interleave battery runners (`abab-run.sh` deleted 2026-11-04, 776e5739; git history is the archive)`

### F9. SVG-CUSTOM-FEATURES.md:21-22 — prewarm2 in present tense (curated live doc)
Offending:
> `lives as a page-locked, device-mapped host complement (~40–42 GiB) built at boot and re-warmed post-pin (prewarm2).`

prewarm2.sh is deleted; the warm-in is the server's own `Prewarm` lifecycle (AGENTS.md:268-270).
Corrected:
> `lives as a page-locked, device-mapped host complement (~40–42 GiB) built at boot (the prewarm2 post-pin warm-in of the founding era is retired — the server's own `prewarm` lifecycle owns warm-in today).`

### F10. SVG-CUSTOM-TODO.md:537, 1272, 1766 — deleted scripts in the live-queue doc
- Line 537: `EXECUTED (no sudo needed): campaign/prewarm.sh — full sequential read of pack+shards into the page cache before every battery boot ... wired into battery-boot and battery-config.` — prewarm.sh AND battery-boot.sh/battery-config.sh all deleted (36555d55). Append one line: `(2026-11-09 note: prewarm.sh and the battery-boot/-config harnesses are deleted — F-268; warm-in is the server's prewarm lifecycle now.)`
- Line 1272: `prewarm2.sh (post-pin dd repopulate + anti-prefix warm requests) BANKED, untested vs true cold.` — script deleted; "BANKED" = git history only. Append: `(script deleted since; the bank is git history 36555d55-era.)`
- Line 1766 (fast-warm package row): `surviving package = campaign/prewarm2.sh post-boot, zero code, +3-5% five-boot means (est.)` — dead path presented as a live option. Append: `(path deleted 2026-11: closed — the server lifecycle replaced the side script.)`

### F11. SVG-CUSTOM.md:11998 — engine/snapshots/ in the F-268 journal entry (ARCHIVAL — do not edit the entry)
> `- **`engine/snapshots/`** — the 22 binary archives (the live `engine/strata` stays).`

The dir was removed today (estate prune, 9bbdfb75). Policy: the journal is the index of what ran;
git history is the archive — historical entries stay. NO line edit. Optional one-line tombstone
in the F-268 entry header if a future session misreads it as current layout: `(2026-11-09: engine/snapshots/ later pruned — 9bbdfb75.)`
No live doc or script references engine/snapshots (verified: zero hits outside this journal entry
and .docs/research digests).

## H4 — README front page (highest-visibility stale claim)

### F12. README.md:21-29 — the hugepages differentiator teaches the boot-armed standing pool
Offending:
> `- **The reserved-page memory architecture** (`docs/HUGEPAGES.md`) — every large streaming organ (expert arenas, the pinned complement, the shadow store) draws from a boot-armed pool of 2 MiB pages: pool memory cannot fragment, swap, or age, engine restarts are a perfect re-arm, and the week-long lane decay that once demanded reboots is gone by construction.`

plus the armed-era claim `measured +5–7% solo on DRAM-ceiling lanes from the page class alone`.
The standing boot-armed pool is the thing convicted and deleted TODAY; doctrine is released
standing + per-session runtime arm for DS (docs/HUGEPAGES.md § Arming).
Corrected:
> `- **The reserved-page memory architecture** (`docs/HUGEPAGES.md`) — every large streaming organ (expert arenas, the pinned complement, the shadow store) can draw from a reserved pool of 2 MiB pages: pool memory cannot fragment, swap, or age, engine restarts are a perfect re-arm, and the week-long lane decay that once demanded reboots is gone by construction. The pool arms per session, only when a model's measured worst draw needs it (DeepSeek), and the box runs released otherwise — a standing boot-armed pool measured as a −12–15% tax on every workload and was retired 2026-11-09.`

(and in the same bullet: `measured +5–7% solo on DRAM-ceiling lanes from the page class alone` →
`with the pool armed, measured +5–7% solo on DRAM-ceiling lanes from the page class alone`).

## H5 — Translations (known-stale, confirmed shape)

### F13. All six translations lack the fork-differentiators era entirely
README.zh-CN/ja/de/fr/es/pt-BR.md have 11 `##` sections each, none named "svg custom
line"/differentiators; README.md carries that section as its opening. The hugepages
"boot-armed" claim is NOT present in any translation (they predate the era — no hazard), and
their 8080 teaching matches the code default (6 hits each, fine). Fix class: full re-sync from
current README.md, one language per pass — NOT line edits. Owner: README.md language row.

## H6 — Project-record gap (named, not created — task was read-only)

### F14. .docs/index.md, .docs/technical_summary.md, .docs/log.txt, .docs/workspace.json — ABSENT
Only `.docs/research/` exists. Per the workspace contract these are required records; creating
them needs a task that authorizes record repair.

## H7 — Media orphans

### F15. docs/media/runpagoda.png — referenced NOWHERE (0 hits across md/html/ts/py)
Every other asset is referenced: app-chat.png (7), guess-and-check.svg (9), how-it-works.svg (9),
pagoda-preview.webp (8), terminal-monitor.png (7); make_figures.py is the figure generator
(IMAGES.md should ideally name it, minor). SAFE-NOW: `git rm docs/media/runpagoda.png`.

## Verified CLEAN (no action)

- docs/HUGEPAGES.md, docs/PERFORMANCE-SETUP.md — the doctrine pair; every env var they cite is
  live in source: STRATA_HUGEPAGE_ARENAS (layer.cpp, expert_source.cpp), STRATA_SHADOW_MEMFD,
  STRATA_NO_LARGEPAGES, STRATA_POOL_SMALL_ARENAS, STRATA_ARENA_PAGES (pinned.cu),
  STRATA_SHADOW_MLOCK (shadow_experts.cpp), DS4_COPY_HUGETLB (ds4_model.cu),
  STRATA_SWAP_GUARD + STRATA_ENGINE_CPUS (all five launch-*8081.sh).
- Service inventory: strata-gpu-lock.service is the only enabled unit; PERFORMANCE-SETUP.md §1
  is the only doc teaching a service install, and it teaches exactly that one. Single-service
  doctrine holds.
- Ports: docs teaching 8080 are code-true (`strata_cli/launch/settings.py:30 DEFAULT_PORT=8080`,
  `launch/adopt.py:97` writes `"port": 8080`); estate configs all bind 8081; `serve/server.py`
  bare default 8095 never reaches a user path. No drift.
- README.md:112 / AGENTS.md:5 "Version 0.2.1 · engine 0.1.46" — TRUE for release artifacts
  (docs/release-notes-0.2.1.md:1: "engine packages are version 0.1.46"). Local BUILD.json says
  0.1.43.1 = the box's build; not a doc defect.
- campaign/ROADMAP-architecture.md and campaign/ROADMAP-dsv41.md — `status: active` is honest
  (open rows with owners: ram_avail fold, CK family, GGUF readers fold, D3-blocked verdict).
  Neither is a concluded roadmap wearing a live badge. ROADMAP-perf-8moves.md stays active
  too (W9-19 flagged rows) but needs the F5/F6 corrections; a one-line progress header
  (W1 adopted, W4 superseded, W9c/W12/X1/X2 falsified) would make its state legible.
- campaign/HOSTMEM-NO-REBOOT.md SUPERSEDED header + M2/M5 tombstones — landed today, correct.
- docs/TROUBLESHOOTING.md hugepage mention = STRATA_NO_ARENA_THP (live env, fine).
- Journal mentions of prewarm2/snapshots inside SVG-CUSTOM.md F-entries — archival by policy,
  no edits (campaign/README.md:3-4).

## Fix order (all SAFE-NOW)

1. F1 + F2 (arming hazard in tracker README + AGENTS.md pointer)
2. F12 (README front page era claim)
3. F3 (SHADOW_STORE backing rewrite), F7 (HOSTMEM M1 tombstone)
4. F4, F5, F6, F8 (contradictions + dead harness in the authoritative map)
5. F9, F10 (prewarm tombstones in FEATURES/TODO)
6. F15 (orphan image), optional F11 tombstone
7. F13 (translation re-sync — multi-pass, separate task); F14 (record repair — needs authorization)
