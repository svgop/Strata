---
type: digest
id: refactor-serve-layer
created: 2026-11-09
status: evidence
scope: serve-layer structure study — serve/server.py + serve/ inventory, fault-prone seams, behavior-preserving decomposition, test-coverage reality
method: full AST inventory (python3 -I, ast.parse) of serve/server.py and serve/frontend.py; cross-reference of every slot/lock attribute write site; git history (919b4db3 NameError fix); import-graph census of all 88 importer files; read of boot path, request path, release path, Prewarm, Service.__init__. READ-ONLY — no behavior-change proposal lacks a named validation gate.
---

# Serve-layer structure study (2026-11-09)

## 1. serve/server.py — inventory

9,060 lines, 135 top-level defs/classes, 67 module constants, 187 `[strata]` witness-print sites.

Classes (line spans):

| Class | Lines | Size | Shape |
|---|---|---|---|
| `Engine` (protocol) | 358-360 | 3 | abstract |
| `MockEngine` | 363-395 | 33 | 2 methods |
| exception family (`EngineDied`, `EngineStarting`, `ClientGone`, `ModelBusy`, `QueueWaitExpired`, `QueueCancelled`, `EngineSilent`, `EngineNonFinite`, `SessionRefused`, `EngineStuck`, `GpuBusy`) | 398-466 | 68 | pure data |
| `ConvCacheLog` | 763-856 | 94 | 3 methods |
| `StrataEngine` | 1011-2684 | 1,674 | **37 methods** |
| `Vision` | 2723-3019 | 297 | 17 methods |
| `ByteTokenizer` / `Detokenizer` / `StopMatcher` | 3532-3655 | 96 | small, clean |
| `Prewarm` | 3715-4018 | 304 | **12 methods — already a class** |
| `Service` | 4097-5944 | 1,848 | **58 methods** |
| `BadBody` / `Server` | 6404-6409, 7753-7772 | 26 | trivial |
| `make_handler` (contains nested `Handler`) | 6412-7750 | **1,339** | one function |

Top functions by size (all >150-line functions flagged):

| Function | Lines | Span | Role |
|---|---|---|---|
| `make_handler` | 1,339 | 6412-7750 | request path: nested `Handler` class + route dispatch, all closures over `svc` |
| `main` | 639 | 8418-9056 | boot path: argparse + config merge + engine/vision/prewarm construction + shutdown |
| `Service._run` | 535 | 5410-5944 | one request's generation loop (budget, parsing, stops, retries, timings) |
| `StrataEngine.generate_batched` | 405 | 1899-2303 | batched admission + decode + release |
| `Handler.do_GET` | 343 | 6685-7027 | 15+ routes in one elif chain |
| `StrataEngine.__init__` | 216 | 1044-1259 | spawn + slot tables + pump thread |
| `Handler._post` | 187 | 7035-7221 | POST routes |
| `Service.prepare` | 122 | 5140-5261 | prompt assembly |
| `StrataEngine.generate` | 122 | 2369-2490 | single-slot path |
| `Service.__init__` | 119 | 4098-4216 | **83 `self.` assignments** — the mutable-state catalog |
| `Service.ensure_loaded` | 109 | 4549-4657 | lazy load + before_load hook + VRAM gate + preboot sweep |

**Overly-long (>150 lines):** `make_handler`, `main`, `_run`, `generate_batched`, `do_GET`. Borderline: both `__init__`s and `_post`.

## 2. Other serve/ files

- `serve/frontend.py` (1,484 lines): 19-method `OutputParser` (900-1484) plus ~30 small pure functions. Largest function 70 lines (`anthropic_to_messages` 612-681). Fine-grained already; **no refactor target**.
- Everything else in `serve/` ≤ 690 lines (`responses.py` 690, `mcp.py` 675, `telemetry.py` 657). `test_server.py` (6,134 lines) is a test file, not serving code.
- `strata_cli/cli.py` 2,118 / `bench.py` 1,307 / `strata_control/runtime.py` 1,265 — out of this lane's scope; same size class, same future treatment.

## 3. Duplicated blocks

1. **Boot-arg parsing vs config derivation — two derivations of the same facts.**
   - Pure derivators exist and are good: `engine_args` (server.py:3121-3161), `parallel_args` (:3167-3185), `learned_profile_args` (:3203-3223), `layer_split_value` (:3082-3112), `effort_end_args` (:3046-3068), `sampling_defaults_from_config` (:8206-8273).
   - `main` then re-derives in-line: parallel re-scanned from raw `cfg["args"]` for `--batch`/`--slots` at :8769-8778 (duplicating `parallel_args`'s detection at :3169), **62 `cfg.get(` calls** and **49 `svc.X = ...` wirings** across :8753-8932; argparse defaults (:8436-8506) vs config fallbacks (:8560-8608, :8847-8857) state the same defaults twice. One new config key = edits in 3-4 places.
2. **Admit-order ceremony repeated 3x** — `queued += 1` → `with self.fifo` → `queued -= 1` → `status.update(busy=True...)` → `finally busy=False`: `slot_action` :4363-4398, `load` :4708-4730, `_run` :5528/:5566/:5582/:5924. The boolean `waiting` + double-finally dance at :4365-4398 is hand-rolled in each site.
3. **Shutdown closer list built twice** — identical four closers (`httpd.shutdown`, `engine.close`, `vision.shutdown`, `hub.close`) at :8987-8990 and :9019-9020; the two lists can drift. `cache_drop.drop_after_exit` wired at three sites (:9030-9033 kill path, :9044-9048 graceful path, `ensure_loaded` :4626-4630).
4. **Probe/health status surface reads from 4 endpoints** — `/health` :6948-6956, `/status` :6969, `/v1/status` → `v1_status` :5002-5050, `/metrics` :4906-4964, `/metrics/prometheus` :4966 — each re-assembles prewarm state, context, load state independently (prewarm snapshot read at :6953-6955 and again inside `v1_status`).
5. **Witness logging**: 187 bare `print(f"[strata] ...", flush=True)` sites; the same "thinking limited to N of max_tokens" sentence appears 3x in `_run` (:5428, :5438, :5445).

## 4. Tangled state

### 4a. The slot family (known) — confined to `StrataEngine`, spread over 13 methods

Attributes: `slot_q`, `slot_busy`, `slot_claims`, `slot_releasing`, `slot_held`, `slot_used`, `slot_order`, `_orphan_seen`, plus the admission twins `waiting`/`wait_lens`/`ctl_ticket` and locks `slot_cv` (:1239), `ctl` (:1247), `wlock` (:1257).

Writers per method (AST cross-reference): `__init__` :1221-1239, `_pump` :1263-1313, `_bdone_after_eos` :1672, `_release_slot_when_done` :1691-1730 (+ its `wait` closure :1705-1729), `_park_slot` :1735-1751 (+ `free` timer closure), `_reap_orphan_slots` :1755-1776, `_take_control` :1780-1853, `_may_go_solo` :1870, `_shorter_waiting` :1880, `_cadence_open` :1885, `generate_batched` :1899-2303, `pick_slot` :2309-2352, `slots_view` :2354.

`slot_busy[slot]` is written from **five execution contexts**: the request thread (`generate_batched` :1969, :2055, :2236, :2281, :2291, :2301), the release-wait daemon thread (:1727), the park `threading.Timer` (:1747), the reaper (caller holds `slot_cv`, :1767), and whole-list replacement in `restart` (:1403 re-calls `self.__init__`, discarding the old lists). There is **no single owner** of the slot state machine; `slot_cv` is the only guard, and the alias-join pattern at :1701 (`q, busy, held, releasing = self.slot_q[slot], ...` "this process's: restart() replaces them") is the patch over the identity-replacement hazard.

### 4b. `Service.status` + `status_lock`

`status_lock` (:4175) is touched by ~25 methods (reserve_inference :4261 through `Handler._captured` :7418); `admission_changed` is a `Condition(status_lock)` (:4176) so drain (:4311, :4318) and release (:4273) interleave with every metrics/status read. `self.status` dict mutated at 9 sites in 4 methods (:4364-4398, :4708-4730, :5528-5924, :5905). `last_request_at` written in 4 methods (:4172, `_run`, `load`, `slot_action`).

### 4c. Half-built `Service`

`Service.__init__` sets 83 attributes (:4098-4216); `main` then assigns **49 more from outside** (:8753-8932: `svc.offline`, `svc.max_queue`, `svc.api_key`, `svc.idle_unload_s`, `svc.gpu_indices`, ...). The object's real contract lives in two files-worth of one function; tests that build a `Service` bare (:4110 in `test_server.py` setUpClass) get only the `__init__` half.

### 4d. Module-global mutation

`global ENGINE_READY_S` reassigned inside `main` from config (:8542-8544); `OPENS_BROWSER` :8511. Frontend flag set by import-side-effect: `frontend.ENV_TO_USER = etu` :8879-8880.

## 5. Fault-prone seams

1. **The NameError class lived in closure-over-mutable-state + rename.** Commit 919b4db3 (2026-11-09): `_release_slot_when_done`'s `wait()` closure (:1705) referenced bare `releasing` after the rename to `self.slot_releasing`; the failure fired *after* `busy[slot]=False` but *before* `notify_all` (:1726-1729) — waiters slept their full timeout on every stop/disconnect. Live in every boot since the #1603 rename; found by a log audit, **not by any test** (§7). The structural cause: a 9k-line module where renaming an attribute cannot be verified by tests because the failing path is a daemon thread's closure. Every closure in `_release_slot_when_done.wait`, `_park_slot.free`, `_take_control`, `generate_batched.give_way`/`others` (:1958-1994) is the same class of hazard.
2. **Lock inventory** (9 in server.py): `slot_cv` Condition :1239, `ctl` :1247, `wlock` :1257, `Vision.lock` :2758, `Prewarm.stop`/`lock` :3757-3758, `Service.fifo` :4115, `config_lock` :4124, `status_lock` :4175, `admission_changed` :4176. **Nested acquisition exists**: `status_lock` inside `fifo` in `slot_action` :4367-4369 and `_run` :5514-5519 (inside `queue_started` called from engine-thread context); `slot_cv` held across `ctl` acquisitions in `_take_control` :1785-1853. Ordering is currently implicit and nowhere written down.
3. **Transition ownership is scattered** (§4a): five writers of one flag. The mitigations (`_reap_orphan_slots` for #1603 leaks, alias-join for restart identity) are patches bolted at the seam, each carrying its own issue tag (#1603, #9, #1012, #1527) — the module has already paid for this shape repeatedly.
4. **`restart()` replaces state by re-calling `__init__`** :1403 — the only object in the file whose lifecycle is "re-run your own constructor". Any attribute added to `__init__` silently becomes restart-replaced state; anything cached in a closure at request start goes stale (the :1701 comment exists to warn exactly this).

## 6. Decomposition proposal (behavior-preserving)

Convention: SAFE-NOW = pure move, import-graph preserved via re-export, no lock/timing semantics touched. NEEDS-BENCH = touches lock ordering, wake timing, or restart identity; requires the bench A/B (law-13) before merge.

| # | Extraction | Contents | Class | Evidence |
|---|---|---|---|---|
| D1 | **`serve/boot_contract.py`** | All pure config/arg functions: `engine_args`, `parallel_args`, `learned_profile_args`, `profile_shape`, `layer_split_value/of`, `effort_end_args`, `engine_silence_s`, `engine_binary`, `gpu_list`, `child_env`, `hip_*` family, `vision_env` (:3022-3529) + `clean_shared_defaults`, `split_joined_args`, `read_config`, `parse_config_text`, `lift_reasoning_effort_arg`, `apply_saved_tuning`, `sampling_defaults_from_config` (:8052-8273) | **SAFE-NOW** | module-level pure functions, zero `Service`/`Engine` coupling; 34 import sites across 88 files hit exactly these names (§7 census: `child_env` 19, `engine_args` 10, `parallel_args` 2). Re-export from `server.py` keeps every importer green; `tools/test_user_args.py`, `tools/test_effort_end_args.py` already characterize them |
| D2 | **`serve/errors.py`** | `Engine` protocol, `MockEngine`, the exception family :358-466 | **SAFE-NOW** | pure data/protocol; `EngineDied` imported by 9 files, `MockEngine` 29, `Service` 38 — all through names, re-export preserves |
| D3 | **`log()` witness helper** | One `log(msg)` wrapper over the 187 print sites | **SAFE-NOW** (mechanical, byte-identical output) | §3.5; a rename-proof seam — the 919b4db3 fix lands in one place next time |
| D4 | **`BootContract.apply(svc, cfg, args)`** | The 49 `svc.X = ...` wirings + their validation prints :8753-8932, in exact order | **SAFE-NOW** as pure move; **NEEDS-BENCH** only if combined with dedup (§3.1) | line order preserved = print order preserved = boot behavior preserved; the duplicated parallel re-derivation :8769-8778 vs `parallel_args` :3169 should *not* be merged in the same commit |
| D5 | **`SlotManager`** (slot tables + `slot_cv` + `ctl` + `waiting`/`wait_lens`/`ctl_ticket` + `_release_slot_when_done`, `_park_slot`, `_reap_orphan_slots`, `_take_control`, `_may_go_solo`, `_cadence_open`, `_shorter_waiting`, `pick_slot`, `slots_view`) | **NEEDS-BENCH** | `generate_batched` interleaves `slot_cv`/`ctl`/`fifo` hand-in-hand (:1936-2095); `_take_control` holds `slot_cv` across `ctl` :1785-1853; `restart` replaces the tables by re-`__init__` :1403. Ownership consolidation is the *right* end state (five writers → one) but every wake order changes unless the moves keep lock scope byte-identical. Gate: characterization tests (§7) + the standing battery lanes before/after |
| D6 | **`make_handler` → route modules** | Split the 1,339-line function into per-route handlers | Helpers (`_body`, `_read_chunked`, `_watch_client`, `_captured`) = **SAFE-NOW**; the `do_GET`/`_post` elif-chain rewrite = **NEEDS-BENCH** (dispatch order is behavior: first-match routes, `_own_page`/`_foreign_page` gates :7040-7054) | health-repair wave 3 (5710d0e4) fixed ten unhandled-exception seams here — the route bodies carry exception contracts that a rewrite must not reorder |
| D7 | **Admit-order ceremony → one `admitted()` context manager** | The 3x `queued/fifo/busy` pattern (§3.2) | **NEEDS-BENCH** | the `finally`-chain at :4395-4398 encodes "release the queue count if we never got the FIFO" — folding it into a CM changes exception flow ordering; low risk but it is the request-admission path, so bench |
| D8 | **Prewarm** | **Already exists** as `Prewarm` :3715-4018 (12 methods). Lifecycle wiring is the remaining scatter: preboot sweep `main` :8701-8706, `Prewarm(...).start(svc)` :8941-8949, `warm_before_load` call in `ensure_loaded` :4639-4646, restart re-arm + geometry re-gate via `_run` polling :3850-3868 | wiring consolidation **SAFE-NOW** (move the three call sites into `Service.attach_prewarm()`); the polling loop itself **NEEDS-BENCH** (2s cadence, `last_done` 30s floor) | |

Sequencing: D1+D2 first (shrinks server.py by ~700 lines, zero risk), then D3, then D4; D5 last and only after §7's characterization net exists. After D1-D4 server.py drops to roughly 8,300 lines with the boot contract testable standalone.

## 7. Test coverage reality

**Import census:** 88 files import `serve.server`. Top symbols: `Service` (38 files), `ByteTokenizer` (37), `MockEngine` (29), `StrataEngine` (22), `child_env` (19), `engine_args` (10), `Vision` (9), `EngineDied` (9). Anything renamed inside server.py can break 88 files — this is the blast radius that made the #1603 rename land un-exercised.

**What exists today:**
- `serve/test_server.py` (6,134 lines): **the request-path characterization harness already exists** — `MockEngine` + `Service` + `serve(svc, port=0)` + urllib against the real `make_handler` (:25-58 shows the pattern). Covers max_tokens budget, both APIs, sampling, security headers.
- `serve/test_slots.py` (449): `/slots` save/restore with a scripted `FakeProc` — exercises `session_file`, not the release thread.
- `serve/test_slot_alloc.py` (108): `pick_slot` via `object.__new__(StrataEngine)` — slot tables without a process.
- `serve/test_admit_order.py` (119): hand-builds admission state (`slot_claims`/`slot_releasing`/`slot_cv`/`ctl`) and drives `_take_control` — the closest thing to a slot-state-machine test.
- `serve/test_parallel_yield.py` (384): covers `_reap_orphan_slots` (:334-338). `serve/test_parallel.py` (812): batched paths.
- `tools/test_geometry_gate.py`: **imports server.py by `exec`-compiling its source into a namespace dict** (:9-13) instead of importing — works, but it is a second, divergent loader path; any refactor that splits the module breaks it first.
- `tests/terminal/` (`test_drain.py`, `test_ready.py`, `lifecycle_matrix.py`, `test_component_start.py`): subprocess-level lifecycle.

**The gap that let 919b4db3 ship:** no test drives `_release_slot_when_done` or `_park_slot` (grep across serve/test_*, tools/, tests/ finds zero direct hits). The release contract — after BSTOP/BDONE: `busy[slot] is False`, `slot not in slot_releasing`, `notify_all` fired, `slot_held[slot]` holds the tail — is asserted nowhere.

**Characterization test to write before touching the request path (or D5):**
1. Slot-release contract, mock-engine style like `test_slot_alloc.py`: `StrataEngine("missing-executable", ["--batch","2"], lazy=True)` + hand-built tables (the `test_admit_order.py:20-30` fixture), a fake `slot_q` fed `BT ...`/`BDONE ...` lines; call `_release_slot_when_done(0, stream=[...])`; join the daemon thread; assert busy/claims/releasing/held end-state and that a `slot_cv.wait(0.1)` wakes. Same shape for `_park_slot` (after `SLOT_PARK_S` the flag clears; a re-claim inside the window survives).
2. Restart-identity contract: alias-join semantics — a release `wait()` closure holding pre-restart lists must not clear the *new* engine's flags (the :1701 invariant, currently comment-only).
3. HTTP golden-stream: extend the existing `test_server.py` pattern — one POST per route family asserting the chunk sequence shape (role chunks, `[DONE]`, `finish_reason`), plus the status invariants after each (`status["queued"] == 0`, `status["busy"] is False`, `history` grew by 1). This pins `make_handler`'s dispatch order for D6.
4. Boot contract: `main(argv)` called with `--engine mock` inside a subprocess-free harness asserting the exit paths (bad key :8591-8603, listen-in-use :8604-8607) — pins D4's reordering safety.

Items 1-2 are the net that makes D5 (SlotManager) executable at all; without them D5 is a NEEDS-BENCH-only change and stays parked.
