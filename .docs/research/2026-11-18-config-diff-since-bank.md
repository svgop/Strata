---
type: research
id: config-diff-since-bank
date: 2026-11-18
lane: qwen (GSQ-RCO iq3_s, port 8081)
bank-era: cc71fe7ee (2026-11-10 00:00:11 +0200) / 712f19810 (2026-11-10 00:08:19 +0200)
scope: config/custom-iq3_s-8081.json, launch-custom-iq3s-8081.sh, campaign/lib/launch-common.sh, serve layer (config/serve/ has no commits — the dir does not exist; the EFFORT tables live in serve/frontend.py)
---

# Config diff since the bank era — qwen lane

## Baseline verification

| Claim | Proof | Result |
|---|---|---|
| Same binary as bank era | `sha256sum engine/strata` = `53dc4e05b27d65b9b9a5f202552035d4cfe4119f07d543b4c6c2daff4122fd0a` | MATCH |
| Engine unchanged in git | `git diff 712f19810 HEAD --stat -- engine/` empty | MATCH |
| BUILD.json vs git | `engine/BUILD.json` is NOT git-tracked (no log entry). version 0.1.46, src `c35b1f24d72bbcf3`, archs [120]. Cannot diff vs git — named boundary. Binary hash is the binding proof. | BOUNDARY |
| No qwen-lane engine code change since bank | `git diff --name-only 712f19810 HEAD -- src/` = all `src/mm3/*` plus ONE doc file: `src/dsv41/vendor/README.md` (80fc63602). Zero qwen-lane (dsv41 code) changes. | CONFIRMED |
| Working tree clean for the four paths | `git status --short` empty | CONFIRMED |

Commits touching the lane's runtime configuration since 2026-11-10: exactly two.

- `945e747d0` 2026-11-18 16:10:37 +0200 — estate move (24 files: 10 configs, 6 launchers, shm-shadow.sh, 7 tools)
- `385937a4d` 2026-11-18 17:22:44 +0200 — shadow NaN rollback (config mtp re-point + launcher disarm + lib canon path)

Adjacent serve-layer commits (not in the four paths, but they change the serving runtime): `c6336d6c6` (Nov 15 13:03) and `2fa29b771` (Nov 18 17:28), both `serve/frontend.py`.

## What did NOT change

Every behavioral argument in `config/custom-iq3_s-8081.json` is byte-identical to `712f19810`: `--spec 6`, `--max-context 262144`, `--kv int8`, `--kv-resident 32768`, `--spec-min-p 0.5`, `--prefill auto:32768`, `--mtp-q4 all`, `--mtp-hnorm stream`, `--pcie-frac 0.35`, `--pcie-mode dma`, `--vram-reserve-mib 2000`, `--conversation-cache-mib 49152`, `--conversation-cache-slots 24`, `--conversation-cache-min-tokens 200`, cvec scale `1.0` + layer range `1 47` + `--cvec-mode project` + `--cvec-dir per-layer`, `--ple-io mmap`, `--adapt-decay 0.85`, expert-profile trio (`data/expert-profile.bin` / `-learned.bin` / save-every `30`), `--resident-experts`, `parallel 6`, port 8081, `"draft_vocab": "en"`, api_key, host. The 41 pre-existing `export STRATA_*` arms in the launcher are unchanged in value (41 → 42 export lines; the +1 is the disarm export below).

## Change table

| # | Change | Commit | Date | Perf class | Evidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | `STRATA_SHADOW_EXPERTS=""` — the shadow store (31.6 GiB hugetlb memfd of q2_0, 75 us/entry CPU-miss reads) disarmed; misses now read the canonical pack / page cache (~152 us/entry class) | 385937a4d | 11-18 17:22 | PERF-AFFECTING (deliberate rollback) | Launcher line 78; lib `-` export preserves explicit empty (launch-common.sh:81-82); 29/29 NaN deaths armed vs 3/3 clean unset; post-rollback warm recheck 169.1/169.0/225.2 (1cdd2851b); t67 head-to-head 142.7/150.8/262.6 (02c216e09) | CONTRIBUTING — the only confirmed perf delta vs bank era; banked numbers were taken shadow-ARMED |
| 2 | All read paths remapped to `/mnt/DATA` (nvme2n1p2, ntfs3, rw): `--pack`, `--native`, `--ple-gguf`, `--control-vector-scaled`, tokenizer, vision mmproj/model. Model dir respelled `Qwen3.8-Flash-Next/` → `qwen3.8-flash-next/gsq-iq3s/` | 945e747d0 | 11-18 16:10 | PERF-AFFECTING CANDIDATE, magnitude UNPROVEN | All artifacts verified present at new paths with preserved mtimes (pack Oct 5, shards Sep 26, cvec Oct 16, mmproj Nov 3). `--ple-io mmap` + pack reads now traverse the in-kernel ntfs3 driver. Mount flags show `rw` — the commit prose's "READ-ONLY" is procedural, not a mount flag. No isolated ntfs-vs-ext4 A/B exists; post-move benches confound this with change #1 | UNKNOWN — artifacts byte-source identical, transport changed; not isolated |
| 3 | `--mtp`: `/home/svg/LLM/data/mtp-en/rt` → `/mnt/DATA/.../strata/data/mtp/rt` + NEW arg `--mtp-draft-vocab data/draft_vocab_en.bin` | 945e747d0 (path remap) + 385937a4d (dir swap + flag) | 11-18 | PERF-NEUTRAL by artifact identity | See mtp section below — same drafter weights, same EN vocab bytes. NOT mentioned in 385937a4d's commit message (the message only names the tee wrapper + shadow rollback) — undocumented commit delta, documented here | NEUTRAL (artifact-identical); protocol-level A/B of flag-vs-default-read not run |
| 4 | `STRATA_CONV_SLOT_DIR` default: `/home/svg/LLM/data/conversation-slots` → `/home/svg/LLM/data-rt/conversation-slots` | 945e747d0 | 11-18 16:10 | NEUTRAL | Same env var, same engine drain/restore path. New home is on the SAME filesystem as the old one (`/home` = nvme1n1p3, 952G — the "806G volume"); only the directory changed because the old parent tree moved to the ntfs mount. Live slots present (slot-14..18.bin written 17:57 today) | NEUTRAL — same semantics, same disk |
| 5 | `_SHADOW_CANON` in launcher + `launch::shadow_export` canon default in lib → `/mnt/DATA/.../packs/q2_0`; `campaign/shm-shadow.sh` SRC remapped | 945e747d0 (launcher comment+var) + 385937a4d (lib line 68) | 11-18 | NEUTRAL today | The tier is disarmed by change #1, so these paths are dormant; they define the correct re-arm home | NEUTRAL (dormant); correctness-required for future re-arm |
| 6 | EFFORT tables, `serve/frontend.py`: c6336d6c6 (Nov 15) introduced per-family tables whose default maps xhigh/max/maximum → `'max'`, which the qwen pack's pre-F-283 template REJECTS → every top-tier request on this lane 400'd Nov 15→Nov 18; 2fa29b771 (Nov 18 17:28) fixed `_EFFORT_QWEN` top tier → `'high'` (template alias renders the xhigh text, claimed byte-identical to bank-era render) | c6336d6c6 + 2fa29b771 | 11-15 / 11-18 | COSMETIC/functional (request mapping; no engine path) | `git show 2fa29b771 -- serve/frontend.py`; in-process tested all spellings × 3 families; `bin/gen-tee.sh` arg bug fixed in the same commit (`exec "${@:2}"`) | NET-NEUTRAL vs bank era (bank-era behavior restored); regression window Nov 15 13:03 → Nov 18 17:28 |
| 7 | `config/serve/` | — | — | — | No commits since Nov 10; the directory does not exist. The task's "serve/frontend EFFORT tables" live in `serve/frontend.py` (row 6) | N/A |

## MTP drafter: bank era vs today (item 3)

Bank era effective drafter (`--mtp /home/svg/LLM/data/mtp-en/rt`, no draft-vocab flag → engine default reads `<mtp>/draft_vocab.bin`):

- `draft_vocab.bin` — REAL file, EN subset, 162,100 B
- `dense.bin`, `dense.txt`, `experts.bin` — SYMLINKS into the full drafter `/home/svg/LLM/data/mtp/rt/`

Today (`--mtp /mnt/DATA/LLM/Models/GGUF/strata/data/mtp/rt` + `--mtp-draft-vocab data/draft_vocab_en.bin`):

| Artifact | Bytes | sha256 | Role |
|---|---|---|---|
| `data/draft_vocab_en.bin` (repo) | 162,100 | `369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25` | the flag's EN subset |
| `.../mtp-en/rt/draft_vocab.bin` (bank-era EN vocab, moved) | 162,100 | `369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25` | IDENTICAL BYTES to the repo file |
| `.../mtp/rt/draft_vocab.bin` (full) | 425,196 | `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` | full vocab — NOT used, the flag overrides it |
| `.../mtp/rt/dense.bin` | 116,099,072 | — | drafter dense weights (mtime Oct 13 03:02, preserved by the move) |
| `.../mtp/rt/experts.bin` | 707,788,800 | — | drafter expert weights (mtime Oct 13 03:02) |

Chain of identity: the bank-era `mtp-en/rt` symlinks pointed at `/home/svg/LLM/data/mtp/rt/*` — the same full-drafter files that now live at `/mnt/DATA/.../mtp/rt/*` (the estate moved the tree wholesale; mtimes preserved; old path gone). The repo's `--mtp-draft-vocab` file is sha256-identical to the bank-era EN `draft_vocab.bin`. So today's drafter = same weights + same EN token subset as the bank era. The flag itself is not new-engine: `engine/strata --help` (bank binary) documents `--mtp-draft-vocab FILE  opt-in: the draft head's token subset (default <mtp>/draft_vocab.bin)`.

Why the re-point happened: the estate move left `mtp-en/rt/dense.bin`, `dense.txt`, `experts.bin` as DANGLING symlinks (verified: targets under `/home/svg/LLM/data/mtp/rt/` no longer resolve; only `draft_vocab.bin` is a real file there). The bank-era config path was broken by the move; `385937a4d` repaired it by pointing at the real full dir and re-supplying the EN subset explicitly. Geometry (dense/experts shapes) and acceptance vocabulary are unchanged → draft acceptance behavior should be bank-era-identical.

## Conversation-cache slot dirs (item 4)

Same semantics: same `STRATA_CONV_SLOT_DIR` env var, same engine drain-at-exit / restore-at-boot path, same slot-`<serial>.bin` format. Old home `/home/svg/LLM/data/conversation-slots` no longer exists (tree moved). New home `/home/svg/LLM/data-rt/conversation-slots` is on the same filesystem as the old one (`/home`, nvme1n1p3) — the move exits the ntfs model-store mount, where writes belonged procedurally even though the mount is technically `rw`. Perf: neutral (boot/exit I/O only, same disk).

## Launcher env arms (item 5)

41 → 42 `^export STRATA` lines. No existing arm gained, lost, or changed a value except `STRATA_CONV_SLOT_DIR`'s default path (row 4). The single added export is `STRATA_SHADOW_EXPERTS=""` (row 1) — a DISARM of a feature that was ARMED at the bank era via the lib ladder (`launch::shadow_export`: hugepages ≥ min → canon memfd; elif tmpfs; elif canon page-cached). Full launcher diff `712f19810..HEAD` contains exactly three hunks: the CONV_SLOT_DIR line, the shadow comment/`_SHADOW_CANON` block, and the disarm export.

## Bottom line

One confirmed perf-relevant delta vs the bank era: the shadow store disarm (row 1) — deliberate, correctness-motivated, and the banked numbers predate it. One unproven candidate: the ntfs3 transport under the mmap'd pack (row 2). Everything else is artifact-identical remapping (rows 3-5) or restored behavior (row 6). Any post-bank benchmark comparison must be read against these two deltas, and row 1 dominates.
