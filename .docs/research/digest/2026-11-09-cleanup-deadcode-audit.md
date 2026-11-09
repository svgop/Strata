---
type: research-digest
date: 2026-11-09
subject: dead-code and comment-rot audit, post-campaign cleanup triage
scope: src/, include/, launch-custom-*.sh (READ-ONLY audit; no code touched)
method: exhaustive grep of the 11 campaign symbols + 6 rot paths; caller tracing
  through the public headers to every construction/open site; pairwise diff of
  all five launchers; CMake registration checks; file-existence proof for every
  referenced script.
verdict: ZERO hard-dead code found in the campaign machinery. Every env gate
  grepped has at least one live in-tree consumer. One redundant specialization,
  three comment-rot sites, one launcher drift block. The prior "shared arena =
  zero callers" verdict is CORRECTED below (CLI-reachable + built selftest).
---

# 2026-11-09 cleanup dead-code audit

## 1. Dead-code classification (campaign symbols)

### 1.1 SharedArenaHeader + shared_file arena path — CORRECTED: not zero-caller

Definition: `src/core/pinned.cu:49` (`struct SharedArenaHeader`), size guard
`:57`, header write/check `:180-208`, map `:234`. Reached only when
`reserve()` gets a non-empty `shared_file` (`src/core/pinned.cu:92,98,149`).

Call graph (production):

```
generate.cpp:2870  "--shared-expert-arena" -> o.shared_expert_arena   (parse)
generate.cpp:6062  arena_src.open(o.pack, ..., pin_limit, o.shared_expert_arena)
expert_source.cpp:5588  ArenaExpertSource::open(..., shared_arena_file)
expert_source.cpp:5776/5779  PinnedArena(..., shared_arena_file, pack_hash)
pinned.cu:450/570  reserve(bytes, backing, note, shared_file, ...)
pinned.cu:180/202  SharedArenaHeader write / verify
```

Call graph (selftest — built and registered on Linux, `CMakeLists.txt:1387-1391`):

```
pinned_shared_test.cpp:25,32,41,47  PinnedArena(bytes, no_bounds, 0, path, pack_hash)
```

Classification: **DORMANT-BUT-REACHABLE, not dead**. The `--shared-expert-arena`
flag is parsed and forwarded, but NO launcher, config file, serve script, or doc
sets it (grep over `launch-custom-*.sh`, `config/`, `serve/`, `docs/`: zero
hits). The selftest keeps the header/format contract exercised at test time.
Removing the machinery requires removing the flag (`generate.cpp:613,2870`,
header decls `include/strata/core/pinned.hpp:63-79`,
`include/strata/core/expert_source.hpp:1254`) AND the selftest as one set.
Risk: MEDIUM. Recommendation: KEEP (see 4.2).

### 1.2 STRATA_ARENA_MMAP — LIVE (env-gated AMD arm)

Readers: `src/core/expert_source.cpp:5636` (gate), `:5659,5714,5846,5860`;
`src/program/generate.cpp:258` (`pin_blobs_locks`), `:2470` (ROCclr threshold),
comment anchors `:247,2465,7595`; storage `include/strata/core/expert_source.hpp:1304`.
KEEP.

### 1.3 STRATA_ARENA_LOCK — LIVE (A/B arm)

Readers: `src/core/pinned.cu:462,470,523,550-559,638-643,685` (lock/disable
notes on all three reserve paths). KEEP.

### 1.4 hugepage_report — LIVE (page-class banner)

Definition: `include/strata/platform/hugepage.hpp:191` (Linux) / `:213` (stub).
Callers: `src/core/layer.cpp:603` ("the KV host shadow"),
`src/core/expert_source.cpp:3758` ("the expert complement"). Both under the
`hugepage_arenas_enabled()` gate. KEEP.

### 1.5 kPlanPartialSkip — LIVE

Definition: `include/strata/kernels/verify_kernels.hpp:166`. Consumers:
`src/core/verify.cpp:2108` (armed by `STRATA_PLAN_PARTIAL`, read at `:2102`),
`src/kernels/dm_plan_parity.cpp:338,385,395,739` (parity harness),
`src/kernels/cuda/verify_kernels.cu:1442,1694,1701,1712` (device wait/zero
paths). `STRATA_PLAN_PARTIAL` also gates `expert_source.cpp:4768,4770`.
KEEP.

### 1.6 kSplit<6> — ONE REDUNDANT LINE + parked traits

Dispatch law: `src/kernels/cuda/iq_kernels.cu:3168,3870,3913,3938` —
`if constexpr (!kSplit<TY>)` routes to the per-entry kernel; the decode-once
multi kernels (`using S = Split<TY>`, `:1438-1638`) instantiate only for types
with `kSplit<TY> = true`.

- `template<> inline constexpr bool kSplit<6> = false;` (`iq_kernels.cu:1134`)
  is a **no-op**: the primary template `:814` already yields `false` for every
  unspecialized TY. Zero references to `kSplit<6>` anywhere else in the tree.
- `struct Split<6>` traits (`iq_kernels.cu:1135-1177`) are **never
  instantiated** (kSplit<6> false closes the only instantiation path; no
  external `Split<6>` reference exists). Deliberately parked per the verdict
  comment `:1127-1133` ("The traits stay for the follow-up: a per-NC dispatch").

Classification: line 1134 SAFE-TO-REMOVE (trivial); traits 1135-1177
KEEP-WITH-REASON (parked experiment, traits kept per campaign law).

### 1.7 il_q8_on — LIVE (parked IlQ8 arm, default off)

Definition: `src/kernels/cuda/native_mmvq.cu:3142` (`STRATA_MMVQ_IL_Q8`).
Gates: `native_mmvq.cu:3195,3202,3278`; parity consumer
`src/kernels/mmvq_il_parity.cpp:316`. Default off = Q8_0 keeps the native
kernels. KEEP-WITH-REASON.

### 1.8 STRATA_SHADOW_MEMFD — LIVE (default ON)

`src/core/shadow_experts.cpp:162` (read), `:171-223` (memfd arm: create
`:173`, ftruncate `:177`, populate `:185-197`, mprotect `:199`), fallback
`:225-228`. `=0` keeps the plain file mapping. KEEP.

### 1.9 STRATA_SHADOW_MLOCK — LIVE (default ON)

`src/core/shadow_experts.cpp:235-240` (read + `mlock`, refusal is non-fatal
`:237`). KEEP.

### 1.10 STRATA_SWAP_GUARD — LIVE (all five launchers)

`launch-custom-dsv41-8081.sh:23`, `launch-custom-dsv41-4e-8081.sh:23`,
`launch-custom-ud-iq2xs-8081.sh:125`, `launch-custom-iq3s-8081.sh:133`,
`launch-custom-ud-iq4xs-8081.sh:125`. Textually identical across all five;
only the config filename differs. KEEP.

### 1.11 COND_RING — correctly absent

Zero hits in `src/` + `include/`. Sole artifact:
`campaign/results/condring-parked.patch`. Correct parked state; nothing to do.

### 1.12 STRATA_HUGEPAGE_ARENAS — LIVE gate, no launcher arms it

Readers: `include/strata/platform/hugepage.hpp:31-37` (gate, default off),
pool primitives `:47-119`; consumers `src/core/layer.cpp:599,608,616`,
`src/core/expert_source.cpp:3401-3402,3556-3559,3598`. Commit 260520cf removed
the export from the GSQ launcher; the gate ships released for the Qwen rotation
and arms per-session for DS (commit ea56db09 doctrine). KEEP-WITH-REASON.

Supporting primitives confirmed live: `huge_pool_free_bytes()`
(`shadow_experts.cpp:212`, `expert_source.cpp:3402`), `huge_pool_covers()`
(`expert_source.cpp:3557`; Windows stub `hugepage.hpp:84` needed for the
non-Linux build).

## 2. Comment rot

### 2.1 Launchers cite a branch that was NOT removed, at a file:line that moved

`launch-custom-iq3s-8081.sh:108-110`, `launch-custom-ud-iq2xs-8081.sh:108-110`,
`launch-custom-ud-iq4xs-8081.sh:108-110` (identical text in all three):

```
# (STRATA_ADAPT_OVERLAP / _RESIDENT removed F-187: the engine refuses them while DMA_BATCH=1 —
#  "is off: not with STRATA_DMA_BATCH" boot line; KEEP_MAPPING removed: dead branch on Linux,
#  expert_source.cpp:510-520. F-185 rejected the overlap trade anyway.)
```

Two false claims: (a) the KEEP_MAPPING branch was **not** removed — it is live
at `src/program/generate.cpp:8504-8509` (`std::getenv("STRATA_KEEP_MAPPING")`
gates `src.drop_mapping()`); (b) `expert_source.cpp:510-520` today holds the
Windows RAM report + `close_direct()` + `TierProbeResult` — the cited lines
have no KEEP_MAPPING content. Action: rewrite lines 108-110 (drop the
KEEP_MAPPING clause or correct it to `generate.cpp:8504`). TRIVIAL risk.

### 2.2 ds4_pool.hpp cites a deleted warming script

`src/dsv41/ds4_pool.hpp:204`:

```
// box whose cache holds the pack's shards (prewarm-ds4.sh warms them), the SSD leaves the answer path.
```

`prewarm-ds4.sh` does not exist anywhere in the tree (find: zero files; only
stale log artifacts `logs/prewarm2-20261027-1017.log`,
`logs/prewarm2-f367.log` survive). The env gate the comment documents
(`DS4_STORE_BUFFERED`, read at `:207-211`) is live. Action: reword the
parenthetical (name the actual warming procedure or drop it). TRIVIAL risk.

### 2.3 No rot for the hugepage-mount family in live surfaces

`/mnt/strata-huge`, `strata-huge-store`, `strata-huge-tmpfs`, `gpu-lock v2/v3`:
zero hits in `src/` and `launch-custom-*.sh`. They appear only in
`campaign/` and `.docs/research/digest/` (archives — correct home, not rot).
The launcher hugetlb comments (`:62-64` in the three quant launchers) match the
current engine behavior (memfd arm first, `shadow_experts.cpp:153-171`).

### 2.4 Stale "pool exists by default" claims: none found

Grep for default-on pool claims in `src/`, `include/`, `docs/`: zero hits.
`hugepage.hpp:1` states "default off" and the code agrees. Shadow memfd/mlock
default-on claims match the code. No contradictions.

### 2.5 Minor: detached measurement comment (all three quant launchers, :83-85)

The orphaned fragment `store (31.64 GiB mapped, 1.38 MB blobs vs IQ3_S's...`
hangs after the `echo` line at `:82` with no export line above it. Identical in
all three launchers (not drift). Cosmetic; fold into the block comment above
or delete during the 2.1 edit.

## 3. Launcher consistency drift (launch-custom-*.sh)

Pairwise diff results:

| Block | dsv41 | dsv41-4e | iq3s | ud-iq2xs | ud-iq4xs |
|---|---|---|---|---|---|
| Swap guard (STRATA_SWAP_GUARD + MemorySwapMax=0 scope) | :23 | :23 | :133 | :125 | :125 |
| Shadow tier ladder (canon/tmpfs/disk + STRATA_SHADOW_EXPERTS) | absent | absent | :65-85 | :65-85 | :65-85 |
| Single-owner rule (HugePages_Free >= 17000 -> engine memfd) | absent | absent | :67-70 | :67-70 | :67-70 |
| shm-shadow re-establish call | absent | absent | :118 | :118 | :118 |
| CPU-pinning block (STRATA_ENGINE_CPUS + taskset exec) | absent | absent | **:121-128** | absent | absent |

Exact divergences:

1. **Swap guard: NO drift.** Byte-identical text in all five; only the
   `--config` filename differs.
2. **THE DRIFT: the taskset pinning block exists ONLY in iq3s**
   (`launch-custom-iq3s-8081.sh:121-128`):

   ```bash
   _CPUS="${STRATA_ENGINE_CPUS-}"   # default OFF (measured neutral on a quiet box; set 8-15,24-31 for contended hours)
   if [ -n "$_CPUS" ]; then
     exec taskset -c "$_CPUS" .venv/bin/python serve/server.py ... 
   fi
   ```

   The campaign commit (a01f4ca-class, "launcher execs taskset on the frequency
   CCD by default (STRATA_ENGINE_CPUS env, empty=off)") implies all launchers;
   four of five lack it. Default-off makes the behavioral difference zero
   today, but the arm is unreachable on the two UD lanes and both DS lanes.
   Action: propagate the 8-line block to the other four launchers (or strip it
   from iq3s until the A/B concludes) — operator decision, not a dead-code
   removal.
3. **Shadow ladder + single-owner rule + shm-shadow call: absent from both
   dsv41 launchers BY DESIGN** — the DS4 subsystem (`src/dsv41`) has no Q2_0
   shadow store; its own budget path carries the pool correction
   (`ds4_os.hpp` per `hugepage.hpp:54-56`). Not drift.
4. dsv41 vs dsv41-4e and ud-iq2xs vs ud-iq4xs: differ only in the config
   filename. Clean.
5. The rot lines 108-110 (section 2.1) are identical in the three quant
   launchers — one edit applied three times.

## 4. Ranked cleanup list

### SAFE-TO-REMOVE (ranked by value, proof attached)

| # | Item | Proof of no effect | Action | Risk |
|---|---|---|---|---|
| R1 | Launcher comment rot, lines 108-110 in `launch-custom-iq3s-8081.sh`, `launch-custom-ud-iq2xs-8081.sh`, `launch-custom-ud-iq4xs-8081.sh` | The KEEP_MAPPING branch is LIVE at `generate.cpp:8504-8509`; `expert_source.cpp:510-520` holds unrelated code. The comment lies about the tree. | Rewrite/delete lines 108-110 in all three files (keep the F-187 clause if wanted, corrected). | TRIVIAL (comments only) |
| R2 | `src/dsv41/ds4_pool.hpp:204` parenthetical "(prewarm-ds4.sh warms them)" | `prewarm-ds4.sh` absent from the tree (find: zero hits). | Reword line 204 to drop the dead script reference. | TRIVIAL (comment only) |
| R3 | `template<> inline constexpr bool kSplit<6> = false;` — `src/kernels/cuda/iq_kernels.cu:1134` | Primary template `iq_kernels.cu:814` already yields false; zero references to `kSplit<6>` tree-wide; dispatch (`:3168,3870,3913,3938`) is unaffected. | Delete line 1134. KEEP the verdict comment 1127-1133 (it records the law-13 decision). | TRIVIAL (semantic no-op; rebuild to confirm identical fatbin) |
| R4 | Stale log artifacts `logs/prewarm2-20261027-1017.log`, `logs/prewarm2-f367.log` | Their producing script `prewarm2.sh` no longer exists; logs are orphaned output. | Archive to `campaign/` or delete per the storage canon (ledger the deletion first). | LOW (out of src/; pure artifacts) |
| R5 | Detached measurement fragment, quant launchers :83-85 | Comment text with no owning export line; identical in all three. | Fold into the block comment at :51-64 or delete during the R1 edit. | TRIVIAL |

Not removal candidates despite suspicion: `STRATA_KEEP_MAPPING`
(`generate.cpp:8505` — live A/B, and the R1 comment edit must not trigger its
deletion), `huge_pool_covers` (`expert_source.cpp:3557` calls it), the Windows
stub `hugepage.hpp:84` (needed off-Linux).

### KEEP-WITH-REASON

| # | Item | Reason it stays |
|---|---|---|
| K1 | Shared-arena machinery: `SharedArenaHeader` + `shared_file` params (`pinned.cu:46-234`, decls `pinned.hpp:63-79`, `expert_source.hpp:1254`) + `--shared-expert-arena` (`generate.cpp:613,2870,6063`) + selftest (`pinned_shared_test.cpp`, `CMakeLists.txt:1387-1391`) | CLI-reachable and test-covered; the "zero callers" prior verdict holds only at launcher/config level. Corrects the record: one production caller chain (off by default, never armed by any shipped surface) + a registered Linux ctest. Remove only as one set (flag + params + machinery + selftest) if the multi-process arena is abandoned — MEDIUM risk, operator call. |
| K2 | Pool-gated hugepage machinery: `STRATA_HUGEPAGE_ARENAS` gate + `huge_pool_*` + `hugepage_report` (`hugepage.hpp`, `layer.cpp:596-616`, `expert_source.cpp:3401,3556,3598,3758`) | Ships released for the Qwen rotation, armed per-session for DS (ea56db09 doctrine); the memfd shadow store (`shadow_experts.cpp:153-243`, `STRATA_SHADOW_MEMFD`/`STRATA_SHADOW_MLOCK` default ON) is the live single-owner path the launchers' HugePages_Free guard selects. |
| K3 | Parked experiment traits kept by law: `Split<6>` (`iq_kernels.cu:1135-1177`) + `il_q8_on`/`STRATA_MMVQ_IL_Q8` (`native_mmvq.cu:3142,3195,3202,3278`) + `kPlanPartialSkip`/`STRATA_PLAN_PARTIAL` (`verify.cpp:2102-2108`, `expert_source.cpp:4768-4770`) | Campaign decision: traits stay for the named follow-ups (per-NC dispatch; IlQ8 re-price; T1.1 partial mode). All three have live parity harnesses (`mmvq_il_parity.cpp:316`, `dm_plan_parity.cpp:338-739`). COND_RING stays parked as the patch file (`campaign/results/condring-parked.patch`) — correct state, tree already clean. |

## 5. Boundary

- No build executed (READ-ONLY audit): R3's "identical fatbin" claim rests on
  C++ semantics (an explicit variable-template specialization duplicating the
  primary's value), not on a compiler run. Rebuild + byte-compare after any
  code deletion.
- `STRATA_KEEP_MAPPING`'s Linux reach (`generate.cpp:8505` is platform-generic
  text; the NTFS rationale is Windows-flavored) was not behaviorally tested —
  out of scope for a static audit. The R1 edit fixes the comment only.
