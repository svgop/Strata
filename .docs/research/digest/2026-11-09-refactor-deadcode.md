---
type: research-digest
date: 2026-11-09
subject: verified dead-code and retired-knob sweep (env vars, zero-call-site functions, post-campaign leftovers)
scope: src/, include/, serve/, strata_control/, strata_cli/ plus cross-reference over launchers, config/,
  docs, batteries, tools, tests, bench, sycl/, metal/, campaign/ living scripts (READ-ONLY; this digest is the only file written)
method: quoted-literal capture of every STRATA_*/DS4_* name in code trees (getenv-adjacent AND helper-wrapped
  reads: env_flag, off(), env_on) crossed against a hardened set/documentation scan (VAR=, VAR:, environ[],
  setdefault, putenv, JSON/YAML keys, launchers, batteries, docs) over the living tree with
  campaign/results, campaign/research, .docs, logs excluded as archives; whole-corpus word-count
  reference counting for C++/Python function definitions with per-candidate manual vetting (virtual
  overrides, CUDA launch sites, template instantiation, framework dispatch); tools/lint_tree.py executed (0 findings).
verdict: NO env var in the production read set is provably dead — every read is a reachable operator
  knob (law 16 holds; nothing here justifies capability shrink). 11 C++ functions and 15 Python
  functions have provably zero call sites (each vetted against virtual dispatch, kernel launches, and
  dynamic dispatch). COND_RING is correctly absent from the tree (parked patch + out-of-tree binary);
  STRATA_WINDOW_GRAPH and ar_on() are LIVE. STRATA_HUGEPAGE_ARENAS remains a live, meaningful gate.
  The lint tripwire is clean and no deletion candidate can regress it (deletions only shrink counts).
---

# 2026-11-09 refactor dead-code / retired-knob sweep

## 1. Env-var inventory and classification

### 1.1 Numbers

| Metric | Count |
|---|---|
| Distinct `STRATA_*`/`DS4_*` names read (quoted literal) in code trees | 1057 |
| LIVE — set or documented by a living launcher/config/doc/battery/py-tool/code-set | 650 |
| ORPHAN — read in code, nothing in the living tree sets or documents it | 407 |
| — of which have at least one production-path read site (non test/parity/bench) | 397 |
| — test/bench-file-only orphans | 10 |

LIVE context overlap (a var can appear in several): doc 559-era names documented, bench-scripts 236,
py-tools 153, launchers 48, batteries 58, config JSON/YAML 7, code-set (`nv_defaults.cpp`/`arch_defaults.cpp`
`setenv` defaults, `apply_nv_defaults`) — the mechanism-set names (STRATA_ADAPT_NOWAIT, STRATA_ADAPT_LAG,
STRATA_SIDE_MIXER, STRATA_ATTN_DAILY, STRATA_GR_UP75, STRATA_GU_SWIGLU, STRATA_QSA_ROWS, STRATA_B6_MMVQ_ROWS,
STRATA_FUSE_WAIT, STRATA_FUSE_DECODE, STRATA_CPU_IQ_ALL_EXACT1, ...) are set by the engine itself per-GPU-class.

### 1.2 Classification rule

- **LIVE** — a living surface sets or documents it. Includes battery-armed A/B legs
  (`campaign/batteries/*.sh`), doc-documented knobs (`docs/*.md`, `SVG-CUSTOM*.md`), and
  code-set defaults (`nv_defaults.cpp:52-65`, `arch_defaults.cpp`).
- **ORPHAN** — read in code, no setter/documenter anywhere in the living tree. These are *reachable
  operator knobs with default-off semantics*, NOT dead code: the gate is compiled into the served
  engine and the knob arms it. Deleting one deletes a capability branch (law 16 blocks removal without
  a proof the branch is unreachable — the branch is reachable by definition of being env-gated).
  The full list is Appendix A; the doc-or-delete decision belongs to the operator.
- **DEAD-CLASSIC** — read in code whose consuming path is itself provably dead. **Zero found.**
  Every getenv read site sits inside a function with live callers or inside a test/parity/bench
  harness that is built and registered (`STRATA_BUILD_TESTS` targets, CMakeLists.txt:280+).

### 1.3 The named suspects

- **`STRATA_HUGEPAGE_ARENAS` — LIVE, gate still meaningful.** Gate: `include/strata/platform/hugepage.hpp:31-37`
  (default off). Three live consumers, none removed by the released-only standing-pool doctrine:
  (1) RAM-budget correction — `src/core/expert_source.cpp:3400-3404` adds `huge_pool_free_bytes()` to the
  complement budget when armed (without it a pool-equipped box reads RAM-starved and caps the complement);
  (2) the paced-pool flip — `src/core/expert_source.cpp:3557-3563`: armed AND `huge_pool_covers(bytes)`
  routes the complement to the paced 2 MiB path instead of one-shot `cudaHostAlloc` (the HOSTMEM-NO-REBOOT
  composition; comment records the law-13 rationale); (3) the paced allocator — `:3598` `huge_alloc`;
  (4) the KV host shadow — `src/core/layer.cpp:596-616`. Armed per-session by
  `campaign/batteries/hostmem-leg2.sh:34-36`, rollback documented `docs/HUGEPAGES.md:109`, and a boot TODAY
  ran with it (`campaign/results/vram-deficit-boot-2026-11-09.out:8`). KEEP-WITH-REASON.
- **COND_RING — correctly absent from the tree.** Zero readers of `STRATA_COND_RING` in src/include/serve
  (grep over all code suffixes). The sole consumer is the OUT-OF-TREE parked binary
  (`/home/svg/LLM/condring-build/build/strata`, built from `campaign/results/condring-parked.patch`, 60 KB,
  present). The battery `campaign/batteries/gsq-condring-ab.sh` was updated 2026-11-09 with the regen
  recipe in its header. The design-digest C6 increment (`wrap = g_window_graph && (ar_on() || cond_ring_on())`,
  `2026-11-09-one-doorbell-conditional.md:42`) never landed: in-tree the wrap is
  `src/core/verify.cpp:2554` `const bool wrap = g_window_graph && ar_on();` — no `cond_ring_on` symbol exists.
  Nothing to delete in-tree; the parked state is coherent.
- **`STRATA_WINDOW_GRAPH` — LIVE opt-in.** Read `src/core/verify.cpp:253-258` (CUDA-only; constexpr false on
  HIP), consumed at `:2552-2554`, tail kernel `win_cond_tail_kernel` defined+launched
  `src/kernels/cuda/verify_kernels.cu:1924,1930`, header `include/strata/kernels/verify_kernels.hpp:92`,
  falsifier bench registered `bench/micro/cond_node.cu` (CMakeLists.txt:1433).
- **`ar_on()` — LIVE.** `include/strata/core/verify.hpp:504`, ~20 call sites in `src/core/verify.cpp`
  (1231, 1340, 2041, 2093, 2236, 2554, 3009-3285, 3963-3966, 4190-4282, 4550, 4568) plus
  `ar_live()` (`verify.hpp:189`, called `generate.cpp:15524`). No retired window-graph gate remains.
- **F-187 comment rot (prior audit R1, still open).** `STRATA_ADAPT_OVERLAP` / `_RESIDENT` are still READ
  (`src/program/generate.cpp:1617`, `:10404`) — the launcher comments at
  `launch-custom-iq3s-8081.sh:108-110` (+2 siblings) claiming "removed" are wrong on both clauses; the
  engine honors/refuses them at runtime. The comment rewrite remains the only action.

### 1.4 ORPHAN shape (who reads them)

Production-path orphans by reading tree (Appendix A carries the names): `src/dsv41` 124 (the Project-Maya
DS4 era tuning knobs — the `strata-dsv41` target is `STRATA_BUILD_DSV41`, OFF by default, CMakeLists.txt:2312),
`sycl/` 33 + shared-with-src ~70 (the `STRATA_ENABLE_SYCL` EXPERIMENTAL target, OFF by default, CMakeLists.txt:14),
`metal/` 23 (Apple platform tree), `src/kernels` 46, `src/program` 57, `src/prefill` 41, `src/core` 49,
`src/qwen35` 9, `include/` 5, `tools/` 11. All are default-off debug/test/A/B knobs (STRATA_TEST_*,
STRATA_DBG_*, STRATA_DUMP_*, probe dumps, one-off A/B arms). Recommendation: decide doc-vs-delete per knob;
none meets the law-16 bar for deletion because each gate is compiled into a built target and reachable.

## 2. Zero-call-site functions

Method: project-wide word-count per identifier (all code, scripts, configs, docs, living campaigns);
a function is a candidate when its name's total textual occurrences equal the number of its
definition statements. Manual vetting per candidate: virtual override check (base-interface dispatch would
hide the derived name — none of the candidates is virtual or overrides a virtual), CUDA launch-by-name check
(`name<<<` counts as an occurrence — zero), template instantiation (dependent calls still textually contain
the name — zero), getattr/string dispatch (ref corpus includes strings — zero), framework hook conventions
(Textual `on_*`/`action_*` excluded from Python claims).

### 2.1 C++ — proven dead (13)

| # | Symbol | Site | Proof | Removal touches |
|---|---|---|---|---|
| D1 | `load_read_seconds()` | `include/strata/core/expert_source.hpp:1299` | 1 occurrence corpus-wide (the accessor). Sibling `load_seconds()` (:1298) is live; only other hit for these names is the basalt snapshot copy under `campaign/headport/` (not compiled). | strata + strata-dsv41 rebuild (header included by src/core, src/program) |
| D2 | `load_copy_seconds()` | `include/strata/core/expert_source.hpp:1300` | same proof | same |
| D3 | `draft_stats()` | `include/strata/core/mtp.hpp:146` | 1 occurrence; zero callers incl. tests/bench. Backing field `h_stat_` is written elsewhere — accessor only. | full engine rebuild |
| D4 | `draft_abort_armed()` | `include/strata/core/mtp.hpp:147` | 1 occurrence; field `abort_stats_` untouched by removal | full engine rebuild |
| D5 | `entry_at(size_t)` | `include/strata/core/conversation_cache.hpp:511` | 1 occurrence (sibling `serial_at` :512 IS live — do not touch) | engine + conversation tests rebuild |
| D6 | `pin_sim_peak()` | `include/strata/core/pin_budget.hpp:180` | 1 occurrence; `pin_sim_report` (:182) is live — keep | engine rebuild |
| D7 | `arena_lo()` | `src/dsv41/ds4_store.hpp:323` | 1 occurrence (strata-dsv41 only) | strata-dsv41 target rebuild (default OFF) |
| D8 | `arena_hi()` | `src/dsv41/ds4_store.hpp:324` | 1 occurrence | same |
| D9 | `identity_mapped()` | `src/dsv41/ds4_store.hpp:325-330` | 1 occurrence | same |
| D10 | `bf2f_()` | `src/dsv41/ds4_attn_split.cuh:23` | 1 occurrence; `__device__` helper, never called from any kernel | strata-dsv41 rebuild |
| D11 | `w2_h2()` | `src/kernels/cuda/iq_kernels.cu:5204` | 1 occurrence; inverse `w2_u32` (:5205) is live — keep | strata kernels rebuild |
| D12 | `computed_code()` | `src/dsv41/ds4_tq.cu:43` | 1 occurrence; constexpr predicate never evaluated | strata-dsv41 rebuild |
| D13 | `k_gate_rows` | `src/qwen35/q36_engine.cu:233-238` | 1 occurrence; `__global__` kernel NEVER launched (no `k_gate_rows<<<` anywhere; no string reference) | q36 engine rebuild |

### 2.2 Python — proven dead (14 of 15 non-framework candidates; 1 withdrawn)

Free functions/methods whose name occurs exactly once corpus-wide (the `def` itself), no decorators, no
dunder, no base-class override, no string/getattr dispatch:

| Symbol | Site | Withdrawn/kept |
|---|---|---|
| `has_configs` | `strata_control/discovery.py:121` | dead (`_has_configs` models.py:54 is a separate dead twin) |
| `_has_configs` | `strata_control/models.py:54` | dead |
| `median_ratio` | `strata_control/autotune.py:152` | dead |
| `answer_class` | `strata_control/tune_knobs.py:335` | dead |
| `requires_flags` | `strata_control/tune_knobs.py:74` | dead (no base class declares it; zero refs) |
| `verify_file` | `strata_control/components.py:86` | dead |
| `parent_of` | `strata_control/procs.py:103` | dead |
| `exe_path` | `strata_control/procs.py:118` | dead |
| `console_columns` | `strata_cli/consoleinfo.py:38` | dead |
| `flag_list` | `strata_cli/launch/params.py:198` | dead |
| `option_for` | `strata_cli/launch/access.py:266` | dead |
| `cmdline_args` | `tools/strata-doctor.py:71` | dead |
| `eval_acceptance` | `tools/mtp_train.py:115` (nested) | dead |
| `as_out_major` | `tools/mtp_phaseb_adapter.py:70` | dead |
| `drain_available` | `strata_control/tune_engine.py:263` | withdrawn — method on a worker object; could be invoked via interface-style duck typing by future callers; grep shows zero refs today, but keep until the tune-engine surface is next touched (operator call, not a proof of deadness failure) |

All 77 other scan hits are Textual TUI framework callbacks (`on_mount`, `on_key`, `on_resize`,
`action_*` — dispatched by the framework by name) — NOT dead, do not remove.

## 3. Deletion candidates ranked (highest confidence first)

1. **D1+D2** `load_read_seconds`/`load_copy_seconds` — public-API-shaped accessors with a live sibling;
   provably uncalled; cleanest diff.
2. **D3+D4** mtp.hpp accessors — same shape.
3. **D5, D6** single accessors with live siblings adjacent (edit precision required).
4. **D10, D11, D12** dead device/constexpr helpers (dsv41 + iq_kernels).
5. **D7+D8+D9** the ds4_store arena-range trio (one hunk, :322-330 including the doc comment).
6. **D13** dead CUDA kernel `k_gate_rows` (+ its comment :232).
7. **Python 14** — control-plane/CLI tools; each is a single-function removal.

None of these changes any lint finding: `python tools/lint_tree.py` reports **0 finding(s)** today, the
exit rule is a ratchet over per-file baselines (`tools/lint_tree.py:76-91`), and deletions only reduce
counts — no new site, no new file. `campaign/batteries/gsq-condring-ab.sh` stays (it tests the parked
out-of-tree binary; its pkill sites are baseline-owned in `tools/lint_exceptions.json`).

## 4. Boundary

- No build was executed (READ-ONLY audit). Every "removal touches" cell is the CMake target set that
  includes the file; rebuild + bench lane comparison (law 13) is required before landing any deletion.
- Dynamic-dispatch residual risk: C++ candidates were checked against launch-by-name, virtual tables,
  templates, and macro indirection; Python candidates against getattr-string dispatch and framework
  conventions. A `dlsym`-style or data-driven symbol lookup would evade this net — none exists in this
  tree (grep for `dlsym|GetProcAddress` in src/serve: engine-local uses only, none reference these names).
- The 397 production ORPHAN knobs are classified, not condemned: law 16 keeps them until the operator
  rules doc-or-delete per knob (Appendix A).
- The set-scan cannot see sets constructed dynamically (`f"STRATA_{x}"`-style). Grep found none for the
  STRATA_/DS4_ namespaces (`src/core/nv_defaults.cpp` / `arch_defaults.cpp` table-driven sets are captured
  via their literal tables).

## Appendix A — ORPHAN inventory (read in code; nothing in the living tree sets or documents it)

See the machine-generated appendix appended below (397 production-path names bucketed by reading tree;
plus the 10 test/bench-file-only names). Counts and provenance were computed from the living tree with
`campaign/results`, `campaign/research`, `.docs/`, `logs/` excluded as archives.

### A.1 Production-path orphans (397) by reading tree

**[include] — 4 var(s)**
STRATA_CONV_COMPRESS, STRATA_EMULATE_CC, STRATA_SPEC_PROB_DT, STRATA_SPEC_PROB_GATE

**[include + src/prefill + sycl] — 1 var(s)**
STRATA_TEST_PAGEABLE

**[metal] — 23 var(s)**
STRATA_METAL_ALLOC_LOG, STRATA_METAL_ARENA_CHUNK_GB, STRATA_METAL_ARENA_STEP_GB, STRATA_METAL_CLASS_WASTE, STRATA_METAL_CTAB, STRATA_METAL_DRAFT_VOCAB, STRATA_METAL_ENGINE_KERNELS, STRATA_METAL_EXPERT_PROFILE, STRATA_METAL_GEMV_NR, STRATA_METAL_IO_THREADS, STRATA_METAL_LOGPOS, STRATA_METAL_MTP_GGUF, STRATA_METAL_PREFETCH, STRATA_METAL_PREFETCH2, STRATA_METAL_PREFETCH_MRU, STRATA_METAL_PROMPT_GEMM, STRATA_METAL_PROMPT_GEMM_MIN, STRATA_METAL_PROMPT_MIN, STRATA_METAL_PROMPT_QSA, STRATA_METAL_READ_SPLIT, STRATA_METAL_SKIP, STRATA_METAL_STREAM_FAKE, STRATA_METAL_STUBS

**[src/core] — 21 var(s)**
STRATA_COMMIT_TIMING, STRATA_COPY_HIP_PRIO, STRATA_DF_PLE, STRATA_DF_QB_Q4, STRATA_FS_ADMIT, STRATA_FS_BUDGET, STRATA_FS_DEPTH, STRATA_IO_PF_BUDGET_FRAC, STRATA_IO_PF_BUDGET_MB, STRATA_IO_PF_GATE_HI, STRATA_IO_PF_MINCONF, STRATA_IO_PF_SCHED, STRATA_IO_PF_TARGET, STRATA_MTP_HIP_PRIO, STRATA_MTP_PROFILE, STRATA_PEER_DIRECT, STRATA_QFUSE_HC, STRATA_RAM_ADAPT_QUEUE, STRATA_TEST_SLOW_COPY_US, STRATA_TEST_UNMAP_GGUF, STRATA_UMA_REGISTER

**[src/core + src/program] — 3 var(s)**
STRATA_ROUTE_RESIDENT_RANKS, STRATA_STARTUP_UNBUFFERED, STRATA_VERIFY_COMPLEMENT

**[src/core + src/program + sycl] — 1 var(s)**
STRATA_MTP_FULL_HEAD

**[src/core + sycl] — 23 var(s)**
STRATA_ARENA_LOCK, STRATA_DEFERRED_REGISTER, STRATA_FETCH_PVM, STRATA_FUSE_HEAD_GR, STRATA_HC_Q8_INJECT, STRATA_HEAD_MIX_MULTI, STRATA_KV_MAIN_OFF, STRATA_KV_RING_OFF, STRATA_LFUSE_GATE, STRATA_LFUSE_PAIR, STRATA_MTP_PREFILL_SYNC, STRATA_MTP_PRIORITY, STRATA_MTP_SHARED_BRANCH, STRATA_NO_BATCH_KV_STEP, STRATA_ONE_TOKEN_COMMIT, STRATA_POOL_TRACE, STRATA_REMOTE_SPIN, STRATA_REMOTE_ZEROCOPY, STRATA_ROUTE_PROJ_PER_TOKEN, STRATA_SH_FORK_LATE, STRATA_TEST_VERIFY_STALL, STRATA_TG_FLUSH_US, STRATA_VERIFY_ARENA

**[src/dsv41] — 124 var(s)**
DS4_AHEAD_EXTRA, DS4_AHEAD_ORDER, DS4_AHEAD_WARM_THREADS, DS4_AR_F16, DS4_ASYNC_T, DS4_ATTN_SLAB, DS4_CACHE_BONUS, DS4_CACHE_DECAY, DS4_CACHE_INFLIGHT, DS4_CHECK_SWAPS, DS4_CHUNK, DS4_CHUNK_MIN, DS4_COMMIT_CAP_GB, DS4_COPY_ALL, DS4_COPY_GB, DS4_COPY_NUMA, DS4_COPY_PIN, DS4_COPY_PIN_RO, DS4_CPU_SMT, DS4_DBG_GG, DS4_DEFAULTS, DS4_DQ_OLD, DS4_DSPARK_CONF, DS4_DSPARK_DUMP, DS4_DSPARK_MAX, DS4_DSPARK_QTEMP, DS4_DSPARK_SAMPLE, DS4_DUMP_H, DS4_DUMP_L, DS4_DUMP_R, DS4_DUMP_Y, DS4_EG_CHECK, DS4_EG_DIRECT, DS4_ENGRAM_CACHE_MB, DS4_ENGRAM_THREADS, DS4_FAST_ATTN, DS4_FEEDLOG, DS4_FLOW, DS4_FLOW_PUMP_WAIT, DS4_FLOW_STATS, DS4_FLOW_TP, DS4_GG_IQ256, DS4_GG_IQ512, DS4_GG_VERBOSE, DS4_HEAD_PER, DS4_IQ_AVX2, DS4_IQ_CTA_PER_SM, DS4_IQ_GATHER, DS4_IQ_MAX_SMS, DS4_IQ_OWN, DS4_IQ_PF_COLD, DS4_IQ_VERBOSE, DS4_LAST_WINDOW, DS4_LAYER_CACHE_MB, DS4_MK_SPLIT, DS4_MX_PF, DS4_MX_TIER, DS4_NO_EOS, DS4_ONE_DRAIN, DS4_PARTIAL_COMBINE, DS4_PATH_PICK, DS4_PF_ALL, DS4_PICK, DS4_PICK_MIN, DS4_PREDICT_CREDIT, DS4_PRED_DUMP, DS4_QUANT_NODEDUP, DS4_QUANT_SINGLE, DS4_READAHEAD_MB, DS4_READERS, DS4_RESIDENCY_COVER, DS4_RING_MB, DS4_ROUND_LOG, DS4_SCORE_PREFIX, DS4_SETTLE_STORE, DS4_SPEC_Q, DS4_SPEC_WASTE, DS4_SPLIT, DS4_STORE_BG, DS4_STORE_BUFFERED, DS4_STORE_CHUNK_SLOTS, DS4_STORE_FLOOR_GB, DS4_STORE_GB, DS4_STORE_POLICY, DS4_STORE_PREGROW, DS4_STORE_PRELOAD, DS4_STORE_PUT, DS4_STORE_RESERVE, DS4_STORE_SPARE, DS4_STORE_THREADS, DS4_STORE_ZC_FRAC, DS4_TEST_IDLE_MS, DS4_TIMER_MS, DS4_TOKLOG, DS4_TOPK, DS4_TP_ALIGN, DS4_TRACE, DS4_TRACE_AHEAD, DS4_UMA_GPU_RESERVE_GB, DS4_WARM, DS4_WINDOWS, DS4_WIN_DIRECT, DS4_WIN_SUBMIT, DS4_XIN_DUMP, DS4_XIN_N, DS4_YOCO, DS4_YOCO_KEEP, DS4_ZC_COARSE, STRATA_DENSE_PFGEMM, STRATA_DS4_A_MB, STRATA_DS4_DISK_SKIP_FROM, STRATA_DS4_DISK_SKIP_W, STRATA_DS4_DSPARK_CUM, STRATA_DS4_LANE_FREE, STRATA_DS4_NO_LIGHT, STRATA_DS4_NO_TC, STRATA_DS4_PROMPT_SKIP, STRATA_DS4_RAM_TIER, STRATA_DS4_REUSE, STRATA_DS4_SLOT_CAP, STRATA_DS4_SLOT_DAYS, STRATA_DS4_SLOT_MIN, STRATA_DS4_TQ_WORDS, STRATA_GLM_EXL3_DECODE

**[src/kernels] — 27 var(s)**
STRATA_CCD_SPLIT, STRATA_CPU_YMM, STRATA_DBG_GDN, STRATA_EXPDEC_DRB, STRATA_GDN_PRESCALE, STRATA_GFX906_ATTN_REDUCE12_TRACE, STRATA_GR_UP_WIDE, STRATA_HIP_WMMA_MIN_CAP, STRATA_MMVQ_IL_Q3K, STRATA_MMVQ_IL_T1, STRATA_NO_IQ4NL, STRATA_NO_IQ512, STRATA_NO_Q8K_AVX1, STRATA_NO_Q8K_AVX2, STRATA_NVFP4_PREFETCH, STRATA_PA_RPF, STRATA_PA_TRIM, STRATA_POOL_UMWAIT, STRATA_POOL_UMWAIT_C02, STRATA_POOL_UMWAIT_ESC, STRATA_PROMPT_ATTN_Q4_TC, STRATA_Q8_SM60_BPS, STRATA_QFUSE_HC_LOCAL, STRATA_SCORES_TC, STRATA_SIMT2_TILE, STRATA_SM60_TABLE, STRATA_SMALL_UNR

**[src/kernels + src/program] — 1 var(s)**
STRATA_NO_IQ128

**[src/kernels + sycl] — 18 var(s)**
STRATA_FUSED_SWIGLU_Q81, STRATA_GROUPED_PAIR_MIN_HITS, STRATA_GROUPED_V1, STRATA_IQ_STAGE_GRID_MMVQ, STRATA_MULTI_BLOCK_ARGMAX, STRATA_NORM_ROPE, STRATA_NO_MMVQ_ROWS2, STRATA_NO_NORM_ROPE, STRATA_OLD_GROUPED, STRATA_PROMPT_ATTN_Q4, STRATA_PROMPT_ATTN_V1, STRATA_Q6P_GRID, STRATA_Q6P_SELFTEST, STRATA_QSA_EARLY_EXIT, STRATA_ROPE_TABLE, STRATA_ROUTER_BLOCK, STRATA_S2_SWIGLU_Q8, STRATA_TOPK_CAPACITY_GUARD

**[src/prefill] — 20 var(s)**
STRATA_DBG_CPU_GATE, STRATA_DBG_FEAT, STRATA_DBG_FEAT_LAYERS, STRATA_DBG_ROUTE_OVERRIDE, STRATA_DUMP_MOE_LAYER, STRATA_GDN_CHUNK_PROF, STRATA_HANDOFF_PIPE_WORKER, STRATA_MMQ_RESIDENT_SORT_NE, STRATA_PF_BK64, STRATA_PF_F16_BN, STRATA_PF_OCC_W, STRATA_PF_PEER_F16, STRATA_PF_PEER_GATHER, STRATA_PF_PEER_HOST, STRATA_PF_PEER_SUMS, STRATA_PF_PEER_XSTREAM, STRATA_PREFILL_STREAM_AHEAD, STRATA_QSA_DUMP_LAYER, STRATA_QSA_DUMP_STEPS, STRATA_STAGER_BATCH

**[src/prefill + sycl] — 21 var(s)**
STRATA_CVEC_FUSE_CHECK, STRATA_DBG_FORCE_SLOW_LAYER, STRATA_GDN_CONV_SERIAL, STRATA_GDN_PIPELINE, STRATA_GROUP_COPY, STRATA_GR_UNFUSED, STRATA_HC_UPMIX_CHECK, STRATA_IDX_FP16_CHECK, STRATA_INDEXER_PER_TOKEN, STRATA_KV_STAGE_OWN, STRATA_MTP_BATCH_F16, STRATA_PF_HCDOWN, STRATA_PF_OCC, STRATA_PF_PEER_COMPACT, STRATA_PF_PEER_OUT_PIPE, STRATA_PF_PEER_RING, STRATA_PF_PEER_STREAM, STRATA_QSA_DUMP_LAST, STRATA_SELECT_OLD, STRATA_SEL_OVERLAP, STRATA_WMMA_BF16

**[src/program] — 27 var(s)**
STRATA_ADAPTIVE_WARM, STRATA_ADAPT_OVERLAP_SLOW_US, STRATA_ELASTIC_DEMOTE_S, STRATA_KV_GROW_HOLD, STRATA_LAYER_SPLIT_GATE, STRATA_LAYER_SPLIT_PROBE, STRATA_PCIE_K_RULE, STRATA_PCIE_SHARE_MAX, STRATA_PIN_BY_NEED_SKIP, STRATA_PIPED_CONCURRENT_UNPIN, STRATA_PIPELINE_AUTO_MIN_STAGES, STRATA_PIPELINE_SPEC_DEPTH, STRATA_PREFILL_HELPER_SOURCE, STRATA_RAM_ADAPT2, STRATA_RAM_ADAPT_CACHED_W, STRATA_RAM_ADAPT_PER_LAYER, STRATA_RAM_ADAPT_PROMPT_W, STRATA_RAM_ADAPT_SKIP_TOP, STRATA_SPEC_ENTABORT, STRATA_SPEC_KEEP_STATS, STRATA_SPEC_LOGISTIC, STRATA_SPEC_SURVTOPK, STRATA_SPLIT_LINK_ABS, STRATA_SPLIT_NEAR_TIE_PCT, STRATA_TEST_CONC_ON_HOST_CPU, STRATA_TEST_PINNED_SHARE, STRATA_WATCHDOG_IO_S

**[src/program + sycl] — 30 var(s)**
STRATA_CKPT_REREAD, STRATA_EARLY_REMOTE_CONTEXTS, STRATA_KV_GROW_FLOOR, STRATA_KV_GROW_INIT, STRATA_KV_GROW_STEP, STRATA_LOGPOS_EXTRA, STRATA_LOOKUP_CHAIN_FIXED, STRATA_PEER_HOT, STRATA_PEER_HOT_AT, STRATA_PIPELINE_AGREE, STRATA_PIPELINE_DOOM_SKIP, STRATA_PIPELINE_LOG, STRATA_PIPELINE_LOOKUP_PON, STRATA_PIPELINE_PRESTAGE, STRATA_PIPELINE_SNAP_OVERLAP, STRATA_PIPELINE_SWITCH, STRATA_PIPELINE_THETA, STRATA_PIPELINE_TRACE, STRATA_PLE_PROBE, STRATA_PREFILL_LEND_PCT, STRATA_REFILL_SERIAL, STRATA_RSS_TRACE, STRATA_SPLIT_COVER_B, STRATA_SPLIT_OWN_BUFFERS, STRATA_SPLIT_OWN_PLACE, STRATA_SPLIT_SMALL_MAX, STRATA_SPLIT_SMALL_OWN, STRATA_SSD_KEEPALIVE_WINDOW, STRATA_TEST_CACHE_FAIL, STRATA_TRACE_ADAPT

**[src/qwen35] — 9 var(s)**
STRATA_Q36_BLAS_F32, STRATA_Q36_DRAFT_VOCAB, STRATA_Q36_KEEP_PACKED, STRATA_Q36_NO_EXT, STRATA_Q36_NO_MMQ, STRATA_Q36_NO_MTP_CPU, STRATA_Q36_PLACE_DEBUG, STRATA_Q36_STREAM_ALL, STRATA_Q36_VDEBUG

**[sycl] — 33 var(s)**
STRATA_CHECK_SWAPS, STRATA_CHUNKED_FILL, STRATA_DBG_DRAFT, STRATA_DOORBELL_PLAIN, STRATA_FOLLOW_LOGPOS, STRATA_GGUF_PLAN_READ, STRATA_GR_LARGE_GRF, STRATA_GR_MORE_GROUPS, STRATA_GR_NORM_SPLIT, STRATA_GR_UP_COLS, STRATA_GR_UP_ILP, STRATA_HOST_UNCACHED, STRATA_KQ_GU_LANES, STRATA_KV_WIDE_HOST, STRATA_MMVQ_PF, STRATA_MMVQ_RPW, STRATA_MMVQ_WIDE, STRATA_MMVQ_WIDE_K_RPW, STRATA_MMVQ_WIDE_MAXC, STRATA_MMVQ_WIDE_MIN, STRATA_MMVQ_WIDE_RPW, STRATA_PF_DQ_FUSED, STRATA_PF_XMX_GU_MAX, STRATA_PF_XMX_ROWS, STRATA_PIPE_HELPER_SIBLING, STRATA_PROFILE_PRIOR_ENTRIES, STRATA_RING_TIMEOUT_S, STRATA_SYCL_GEMM_NO_STAGE, STRATA_VERIFY_HOST_UNCACHED, STRATA_VERIFY_PIPE, STRATA_VERIFY_SPLIT, STRATA_VERIFY_STEPPED, STRATA_WINDOW_SYNC

**[tools] — 11 var(s)**
STRATA_DUMP_CTX, STRATA_DUMP_NGL, STRATA_QI_EXACT_ATTN, STRATA_QI_FAST_LN, STRATA_QI_GFX12_ATTN, STRATA_QI_GGML_ATTN, STRATA_QI_NO_QKV_FUSE, STRATA_QI_OLD_QKV, STRATA_QI_QSPLIT, STRATA_QI_TIMING, STRATA_VISION_ARCH_CHECK

### A.2 Test/bench-file-only orphans (10)
DS4_BENCH_ACTIVE, DS4_BENCH_WIN, STRATA_BENCH_PLANAR, STRATA_BENCH_QMIXED, STRATA_FAST_GDN, STRATA_PARITY_DUMP, STRATA_QSA_PROBE_OUT, STRATA_SESSION_GOLDEN_PRINT, STRATA_TEST_VERBOSE, STRATA_WMMA_PARITY_EXACT
