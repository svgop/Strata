# T26-4 — SERVING CONCURRENCY MECHANICS AT SOURCE: PIPED_CONCURRENT vs dma, prompt-read-beside-windows, elastic KV + batch slots, admission overlap in the cohort reactor, the 320-380 ladder

Date: 2026-10-21. Scope: line-level read of our tree (strata-custom-2 @ F-252 standing: merged v0.2.1 line,
COHORTS=2, conc 315.7 five-boot mean) and the read-only references (basalt fb0944c7, project-maya 35bb2b2).
Read-only session: no builds, boots, GPU work, or edits outside campaign/research/. READ-FIRST set consumed:
T25-3-serving-scheduling.md, F-201-scheduling-agent.md, F-201-kv-agent.md, R2-S3-two-cohort-impl-plan.md,
F-249/T21 (SVG-CUSTOM.md:8458). Everything below was re-verified at source this session; anchors are
file:line in OUR tree unless prefixed "basalt "/"maya ".

---

## 0. EXECUTIVE SUMMARY

1. **The dma refusal is real and fully mechanical — but it is not the binding constraint.** The gate is
   `batch_read.hpp:48` (`if (s.pcie_dma) return "--pcie-mode dma";`), fed by `generate.cpp:13838`. The
   conflict: under PIPED_CONCURRENT the slots' doorbells are served on a dedicated thread whose contract is
   **"makes no CUDA call"** (generate.cpp:13909-13912, hazard #31); under `--pcie-mode dma`, serving one
   doorbell calls the sink fetch, which issues `cudaMemcpyAsync` + `cudaLaunchHostFunc` on that thread
   (`expert_source.cpp:4776` → `verify.cpp:3166-3192`, `fetch_dma`). A driver-lock hold there stalls exactly
   the window spinning on the flag the call was supposed to raise (our own comment: generate.cpp:9154-9157,
   copied from basalt generate.cpp:5222-5224). In kernel/direct mode the same call passes `n=0`
   (`expert_source.cpp:4758,4776`) and the doorbell service is volatile writes + atomics only
   (`verify.cpp:3169,3969-3979`) — legal. basalt's answer was to never ship dma (auto→kernel,
   basalt generate.cpp:5222-5225) and run the read beside windows unconditionally.
2. **T25-3's "one env boot" falsifier is dead on our lane — the overlap branch is unreachable without code.**
   The PIPED read runs only `if (piped && pr_conc ...)` (`generate.cpp:13850`), and
   `piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1` (`generate.cpp:12025`) with
   `n_pipe = stages.size() + 1` (`generate.cpp:12024`) — on our single-GPU COHORTS lane n_pipe==1, so piped
   is false ALWAYS (the pump/group machinery at generate.cpp:12049-12196 is the multi-GPU pipeline path; our
   lane is driven by `batch_step`, generate.cpp:12231-12236). Today's arm would print the dma refusal and
   then silently do nothing (only side effect: window stream priority, verify.cpp:921-932). **AMENDED
   verdict; the port is the cohort-lane reader thread, cost M, not env-S.**
3. **Admission overlap concretely** = today, during any prompt read, the cohort reactor stops; between
   pieces the slots decode for `STRATA_BATCH_DECODE_SHARE × piece_ms` (default 0.5,
   generate.cpp:13696-13704, 14185-14191), and those budget windows run against a cache shrunken by the
   prompt path's borrow (boot log: "the prompt path borrows 6647 CUDA0 cache slots (12.55 GiB)" of 9469 —
   borrowed experts are computed on CPU, generate.cpp:13816-13818). The overlap port = reader thread +
   the existing Doorbells thread iterated over `{ver, ver_b}` + main thread launching cohort rounds and
   running the serial draft/output steps between rounds (basalt's mail-box shape, basalt
   generate.cpp:8012-8107). Insertion site: `read_part` (generate.cpp:13715+, piped branch :13850).
4. **Elastic KV confirmed in basalt (floor 32768 + 16384-cell take-back steps, VMM tail segments,
   between-windows only), but it is NOT our mechanism**: basalt's elastic is explicitly incompatible with
   KV streaming (basalt generate.cpp:2962-2963) and we RUN `--kv-resident 32768` streaming (boot log line
   8; `kv_stream.cu`). Our `--kv-grow` hard-excludes `o.batch == 0` (generate.cpp:5194-5198). Standalone
   conc gain at our probe ≈ 0-2% (KV-stream VRAM hit 97.1-99.4%, log lines 94/103 — mid-depth streaming is
   ~1% of block reads). Its real value is **funding parallel-8** (4+4 cohorts fits `kVerifyMaxT=8` rows
   exactly, verify_kernels.hpp:21) — the only +15-25%-class lever on the board.
5. **The 320-380 ladder, ranked by confidence:** (1) `STRATA_BATCH_DECODE_SHARE` raise (env, S, +1-4%
   est.) → (2) PIPED-for-cohorts port (M, +3-6% net est.) → (3) `--adapt-every 1` (env, S, +0-2%) →
   (4) parallel-8 funded by elastic-KV/vram-reserve (M-L, +15-25% class, reaches 380-420). COHORTS=3 stays
   dead at 6 clients (F-249); maya fair-slice and ADMIT_SHORT_FIRST move TTFT, not conc.

---

## 1. DONOR MECHANISM MAP (basalt @ fb0944c7, maya @ 35bb2b2 — read this session)

### 1.1 basalt prompt-read-beside-windows (always-on) + stream priority

- The overlap loop: basalt `generate.cpp:8012-8107` (`read_part`). Gate `o.batch <= 0 || piped ||
  (!stages.empty() && !batch_on())` → plain read; else the WHOLE range is read on a side thread
  (`std::thread th` at :8047 running `sp.run`), while the MAIN thread loops: run queued mail jobs (draft
  layer / output steps, the `Mail` box :8019-8038) and `batch_step()` (:8053-8056) — the windows keep
  launching. Log line "basalt batch: overlapped prompt read: N windows ... beside M chunk jobs" (:8107).
  Commit 81ab8aef removed the overlap_on gate (always-on with --batch); 95f3f535 added the priority.
- Stream priority: `ver.set_main_priority(o.batch > 0)` (basalt generate.cpp:5190; `main_high_` at
  basalt include/basalt/core/verify.hpp:118) → the windows' main stream `cs_` is created at the greatest
  stream priority (basalt verify.cpp:1044-1045), plus side branches at that priority (:1053-1063); batch
  graphs instantiate with `cudaGraphInstantiateFlagUseNodePriority` (basalt verify.cpp:3284-3285).
- BYIELD cooperates with the overlap: `sp.should_stop`/`abort_read` (:8043, :8046) stop the read at a
  chunk; the yield handling is :8109-8131.
- **No doorbell thread exists in basalt** (no `set_external_serve`/`batch_serve` anywhere in their tree —
  grepped): their doorbells are served inside `batch_step` on the main thread. Their conflict resolution
  for the CUDA-call hazard is mode choice, not thread design: basalt generate.cpp:5222-5224 — "auto: the
  copy kernel for every pack. DMA has the host call cudaMemcpyAsync + cudaLaunchHostFunc inside a verify
  window while the GPU spins on the flag they raise, and the host can get stuck in that cudaMemcpyAsync on
  a driver lock for good." `auto` resolves to kernel (mode 2) at :5225.
- Their prompt path lends cache slots too; a read's borrowed slots are seen missing by the windows (CPU
  computes them) — same economics as ours, i.e. their shipped win is net of that cost.

### 1.2 basalt elastic KV + batch slots

- Floor: `kKvElasticFloor = 32768` (basalt generate.cpp:2964); armed when serve, one GPU, no
  `--kv-resident`, no `--vram-elastic`, `max_context > 32768` (:2965-2967). **"Not with KV streaming (its
  pools are a fixed set of VRAM pages)"** (:2962-2963) — the pools are VMM-mapped per session on demand.
- Tail accounting: what the sessions' unbacked cells would take is measured over the solo session + all
  `bslot_ss[0]` slot sessions (:3466-3471) and given to the expert cache: `xcache.set_elastic_tail(...)`
  (:3472); the tail slots are "the ones the KV may take back" (:3547-3551, boot line "elastic KV: N cells
  backed per session, the other X GiB ... the expert cache's last M slots").
- Take-back: `kv_fit(s, cells, e, shrink)` (basalt generate.cpp:6309-6341): `kv_drain` (wait_commit +
  adaptive drain + `cudaDeviceSynchronize`, :6239-6250 — between windows, never inside one, :6235-6237),
  then whole VMM tail segments released (`xcache.shrink`, evicted experts → `vram_evicted`, `host_res =
  kNotResident`, :6327-6340), then `session_kv_map(s, cells)` (map + zero, :6341). A request backs its
  prompt + `kKvGrowCells = 16384` (:6235) before its prompt is read, and +16384 more whenever decode runs
  past (call sites :6578, :6681, :7586, :8448). Evicted experts become CPU misses read from their RAM
  copies (running.md:19-26).
- Give-back: `kv_release` (:6343-6357) = `session_kv_unmap` + `cache_regrow` (:6253-6305: map the tail
  back as far as free VRAM above the reserve allows, refill returned slots from RAM, most-routed-first).
- Cost per move: a cache shrink (miss-rate bump for the following windows), a device sync, VMM map ops —
  every move logged with ms (basalt generate.cpp:6329, 6346).
- Draft K/V is elastic too (:3140-3141).

### 1.3 maya fair-slice (7-12-client lane)

- Server: `FairLock` (maya serve/server.py:141-235: ordered deque of [token, units], `wait_turn` holds the
  place), `fair_slice_seconds()` = `STRATA_FAIR_SLICE_S` (:230-235), `_sliced` (:2123-2260): once the
  answer has decoded ≥ slice seconds AND someone waits, send engine `PAUSE <rid> keep` (:991-993), requeue
  at units=1, continue later from prompt+produced (only the delta is re-read).
- Engine: `PAUSE`/`PAUSE keep` control verb (maya generate.cpp:1131-1191: `ctl` map rid→{STOP, PAUSE,
  PAUSE keep}; `pause_snapshot` :1836-1851 moves the state to the last fed token). Measured anchor:
  CHANGELOG #93 — first word 10 s vs 60 s during a 700-token answer (2x V100). Fairness only; 0% conc.

## 2. OUR EQUIVALENTS (file:line, this session)

| donor piece | ours | state |
|---|---|---|
| overlap loop | `read_part` generate.cpp:13715-14203; piped branch :13850 (`piped && pr_conc`); `piped` def :12025; pump :12071-12222 (batch_launch at :12184 — single-window-per-stage, NOT the cohort loop); Doorbells thread :13913-13971 (iterates `stage_verifier(k)`, k<n_pipe, calls `batch_serve`) | present, **piped-gated: unreachable on single GPU** |
| stream priority | verify.cpp:920-934: `cs_` created at greatest priority when `STRATA_PIPED_CONCURRENT=1` and `STRATA_PIPED_CONCURRENT_PRIO!=0` (default on) | present, env-gated (arms even when the overlap silently no-ops — a confound) |
| refusal gate | batch_read.hpp:31-54 (`ConcurrentReadSetup`, `concurrent_read_off`), evaluated at generate.cpp:13827-13849; dma flag set at :13838 | present |
| doorbell service | `batch_serve` verify.cpp:3935-3985 (ext_serve path; pool call :3962-3964; empty-plan fallback :3969-3978; flag raises :3977-3979); `batch_poll` ext_serve branch verify.cpp:3994-4017 (cudaStreamQuery ≤1/ms, :3998-4002); dma fetch: expert_source.cpp:4776 → verify.cpp:3166-3192 (`cudaMemcpyAsync` :3178, batched :3180, `cudaLaunchHostFunc` :3191); non-dma fetch passes n=0 (expert_source.cpp:4758, :4776) → flag only (:3169) | present |
| piecewise read (what actually runs today) | pieces = 1 chunk each beside slots (batch_read.hpp:22-28); between pieces `batch_step()` for `decode_share × piece_ms` (generate.cpp:14185-14191; default 0.5 at :13701-13704); reactor idle during the piece itself | active on our lane |
| BYIELD give-way | engine side generate.cpp:13710-13713, 13801-13810; server side server.py:1645 (`YIELDS_MAX=2`), :1744-1747 (`_shorter_waiting`), :12904 | active |
| admission order | F-14 FIFO tickets server.py:1141-1143, 1655-1656, 1674-1678; short-first merged but dormant: server.py:310-317, 1670-1688, `_admit_next` :1719-1729; `QueueWaitExpired` :391, :1663 | present, NOT armed |
| cohort reactor | init generate.cpp:9108-9133 (`ver_b`, slots [0,3)/[3,6), own pcie_mode :9116); round generate.cpp:11503-11602: `collect` :11522-11543 (rotates within span, MTP 2 rows/slot, cap `kVerifyMaxT` :11524), stage both :11574-11581, `batch_go(err,false)` both :11582-11589, ONE reactor loop polling A then B :11591-11602, per-cohort tails `finish` :11620+ | adopted (F-233) |
| kv primitives | `batch_stage_rows`/`batch_go` verify.cpp:3896-3927 (stage: :3896; go: graph launch + b_running :3904-3927); commit from cohort tail, `with_commit=false` comment :3909-3911 | — |
| elastic KV | `--kv-grow` gate generate.cpp:5187-5205: requires `!multi_gpu && kv_resident<=0 && profile && !resident_cpu && expert_cache && !remote && vmm_available() && o.batch == 0 && !vram_elastic && peer<0` (:5194-5198) — **the batch carve excludes our lane**; KV streaming is our live mechanism (`--kv-resident`, generate.cpp:562, 2509, 3202-3206; kernels/cuda/kv_stream.cu) | kv-grow dormant; streaming active |
| PAUSE | absent (control lines are BSTOP/BYIELD only, generate.cpp:12258-12263, :12284+) | missing |

Boot-log facts for the mapping (custom-iq3s-cohorts.log, F-233 boot): "KV streaming: 32768 of 131072 cells
per QSA layer in VRAM, the K/V in 1.55 GiB of pinned RAM" (line 8); "6 slot sessions on CUDA0 (0.73 GiB
each)" (line 10); slot sessions take 4.37 GiB the cache would hold (line 11); "expert cache 9469 slots,
17.96 GiB" (line 29); prompt path borrows 6647 slots / 12.55 GiB (line 52); KV-stream VRAM hit 97.14%→99.41%
(lines 94, 103). The mission's "8090676 blocks" appears nowhere in tree or logs (searched) — closest real
quantities are 9469 expert-cache slots and the per-layer 131072-cell K/V; treat that number as
unverifiable.

## 3. DELTAS + IMPLEMENTATION SPECS (guards: single-stream protection)

### 3.1 PIPED-for-cohorts (the admission-overlap port; the mission's named lever)

- Delta vs today: the reactor loop (generate.cpp:11591-11602) and the piecewise decode-share budgets
  (:14185-14191) are mutually exclusive with the read; the port makes them concurrent.
- Spec (reuses existing parts):
  1. New gate beside :13850: `cohorts == 2 && stages.empty() && pr_conc && req_imgs.empty()` (keep the
     pr_conc exclusion list; it is exactly the doorbell-thread contract check).
  2. Doorbells: the existing thread (:13934-13971) iterates `{ver, ver_b}` instead of `stage_verifier(k)`,
     k<n_pipe — a 3-line change; `set_external_serve(true)` on BOTH (:13924) — batch_serve/batch_poll
     ext_serve already exist per verifier (verify.cpp:3935, :3994).
  3. Main thread: launches cohort rounds (`batch_stage_rows`/`batch_go`, :11574-11589) whenever any slot
     can step, runs the serial draft/emit/output jobs (the `finish` tail :11620+) between rounds via a
     mail box (basalt generate.cpp:8019-8038 shape), and pumps BYIELD/BSTOP (:14024-14035).
  4. Reader thread: `sp.run` (:13986-14001 verbatim) with `stop_at_chunk` for BYIELD.
  5. Guards (single-stream/M1/pair protection): gate on `n_active > o.batch/2` (the reactor's own
     condition, :11503) — solo/pair lanes never see the new path; the F-233 ramp fence (:11502-11510)
     unchanged; adaptive tier stands still during the read (:11616 `!in_prompt_read` preserved); the
     reader's slots are pre-lent and refilled after (:13816-13818 semantics unchanged).
- dma question, resolved: keep `--pcie-mode dma` ONLY if the doorbell thread never serves a dma fetch — it
  does (expert_source.cpp:4776), so the port REQUIRES `--pcie-mode kernel` (or direct). Price:
  upstream-measured ~1-3% decode (45.3→44.8 tok/s at 8 requests, generate.cpp:9157 comment); at our
  2.0-2.3% miss rate the PCIe share is small, so estimate sub-1% on conc (ESTIMATE, not measured).
  `STRATA_DMA_BATCH=0` does NOT clear the gate (it selects the submission form inside fetch_dma,
  verify.cpp:3176-3187; the gate reads `o.pcie_mode`, generate.cpp:13838) — T25-3's alternative leg is
  wrong on this point.
- Gain: T17B candidate-4 pricing is admissions ≈ 14% of probe wall (1.6 s of 11.56 s) → +6-8% gross;
  net of the dma→kernel tax and the un-removable borrow-missing windows (§2), call it **+3-6%
  (ESTIMATE)**. Plausible: yes — two trees ship the mechanism; but note T17B's 14% was measured pre-cohort;
  cohorts shrank window time, so the admission SHARE is likely higher now (the one-boot log will re-price).

### 3.2 prompt-read-beside-windows falsifier without code changes

Not possible on the scoreboard lane: with `piped==false` the overlap branch is unreachable
(generate.cpp:12025, :13850). The env-only boot (`STRATA_PIPED_CONCURRENT=1`) changes exactly one real
thing on our lane — window stream priority (verify.cpp:921-932) — and prints the dma refusal. A/Bing that
would measure priority alone, not overlap. The zero-code DIAGNOSTIC that is worth one boot: the refusal
line confirms the gate wiring, and the boot log's admission share (BGEN→first-BT gaps vs window wall)
re-prices §3.1's numerator. Expected effect of the true port: §3.1's +3-6%.

### 3.3 Elastic KV + batch slots, mapped onto our KV layout

- Exact change if ported: our slot sessions are `--kv-resident 32768` streaming pools (fixed VRAM window +
  pinned-RAM spill, kv_stream.cu) — basalt's elastic replaces exactly this layer with VMM per-session
  pools + cache-tail lending; porting means building the VMM pool for `bslot_ss` sessions and the
  `kv_fit`/`kv_release` pair at admission/decode boundaries, plus unwiring `o.batch == 0` in the kv-grow
  gate (generate.cpp:5197-5198) and `set_vmm(on)` (generate.cpp:5204). M-L.
- What it buys HERE: (a) standalone ≈ 0-2% conc (ESTIMATE) — the probe's KV-stream misses are already
  ~0.6-2.9% of block reads (log lines 94/103), so resident-32k costs ~nothing at probe depths; (b) the
  real prize: shrink per-slot backing to fund 2 more slots (parallel-8: +1.46 GiB by the 0.73 GiB/slot
  figure) or grow the expert cache — but funding FROM the cache is self-defeating at the measured slope
  (F-183: ±513 slots = ±4.9% conc; 1.46 GiB ≈ 770 slots ≈ −7% class); the honest sources of the 1.46 GiB
  are elastic backing or `--vram-reserve-mib` 2000→~1000 (json:33-34; fragmentation risk, untested).
- Risk to the single-stream path: the solo session shares the kv-resident machinery (generate.cpp:5214-5218
  reports it per primary session); a slot-only VMM pool leaves the solo path byte-identical if gated to
  `o.batch > 0`. Standing laws engaged: kvg_ensure OFF (T15B), VMM availability check
  (`vmm_available()`, gate :5196), F-183 slope, F-230 untouched (no prefill bits).
- Cheapest falsifier (zero-code, already answerable): the KV-stream hit-rate line IS the instrument —
  97-99% VRAM hits on the probe say no session lives deep enough for elastic to pay on the scoreboard;
  run it per-request (one log line, T25-3 §2.2's instrumentation) only if parallel-8 is chartered.

### 3.4 ADMIT_SHORT_FIRST — where it slots in

Env-only, zero code: arm `STRATA_ADMIT_SHORT_FIRST=<tokens>` (+ optional `STRATA_ADMIT_MAX_WAIT_S`) in the
launcher; it reorders ONLY first admissions inside `_take_control` (server.py:1670-1688 via `_admit_next`
:1719-1729, entries flagged `new` :1651), under the F-14 ticket law (tickets are handed out in arrival
order :1655-1656; short-first only chooses WHO may take the control lines next). It does not touch
verify.cpp or the reactor. Effect at exactly 6 fixed clients: none (order among simultaneous admissions);
at 7+: queued-TTFT shaping. The reactor-side complement (none exists): a BYIELD is the only in-flight
give-way today (YIELDS_MAX=2, server.py:1645).

### 3.5 maya fair-slice requirements (7-12 clients)

1. Engine `PAUSE [rid] keep` verb: new control-line parse beside BSTOP/BYIELD (generate.cpp:12258-12263),
   a pause flag consulted at the NEXT WINDOW BOUNDARY (our slots commit per window already — the cohort
   tail's `keep[]`/commit at generate.cpp:11620+ — so "suspend at boundary" = mark slot inactive-but-warm,
   emit a SUSPENDED line; resume = BGEN continues from the slot, which the slot cache already supports).
2. Server: FairLock + `_sliced` port (maya server.py:141-235, :2123-2260) onto our `_take_control`
   structure; the F-14 epoch machinery for reordering already exists (`ctl_epoch`/`after_epoch`,
   server.py:1671, 1678) and must be extended to slices (a sliced request re-enters at units=1).
3. Heartbeats while requeued (maya :2150+ pattern). Cost M-L. Scoreboard conc: 0 by construction
   (6-client probe has no queue); conflict: slot-set stability within a cohort round (a PAUSE must land on
   a window boundary — the same rule R2-S3 §3 set for staging).

## 4. GAIN / COST / FALSIFIER (each row one mechanism)

| mechanism | gain (est. marked) | cost | falsifier |
|---|---|---|---|
| PIPED-for-cohorts port (§3.1) | +3-6% conc (ESTIMATE; gross +6-8% per T17B #4) | M (one gated branch + 3-line bells change; reuses batch_serve) | one 3v3 boot with the port + `--pcie-mode kernel`; gates: token-equality per slot vs standing, conc delta, dma→kernel tax read from the same boot |
| decode-share raise (NEW, env) | +1-4% conc (ESTIMATE: hides 20-35% more of the read wall; admissions ~14% of wall) | S (one env: STRATA_BATCH_DECODE_SHARE 0.5→0.7-0.85, generate.cpp:13701) | one 3v3 boot; watch TTFT (reads slow) and M1/pair no-regress |
| `--adapt-every 1` (T25-3 #3) | +0-2% conc on fixed probe prompts; +2-4% class on switchy traffic (their measurement) | S (env) | one 3v3 boot |
| ADMIT_SHORT_FIRST (§3.4) | 0% conc at 6; TTFT only at 7+ | S (env) | one 7-client lane, queued-TTFT readout |
| maya fair-slice (§3.5) | TTFT 60→10 s class (their #93); 0% conc | M-L (PAUSE verb + slice server) | only if the 7-12 lane is chartered; first step is the env above |
| elastic KV under slots (§3.3) | ~0-2% standalone (ESTIMATE); enabler of parallel-8 (+15-25% class, T17B #1) | M-L | the KV-stream hit-rate lines (already 97-99%) + per-request depth log if parallel-8 chartered |
| parallel-8 (4+4 cohorts) | +15-25% class (T17B), fits kVerifyMaxT=8 exactly (4 slots × 2 MTP rows) | M-L (KV funding + slot drafts + F-183 guard) | boot with parallel 8 on today's KV (0.73×8=5.86 GiB vs 4.37 — the 1.46 GiB gap must come from reserve/cache first) |
| COHORTS=3 | measured FLAT (F-249: M1 −1.6%, pair −2.3%, conc +0.3%, all inside noise) | — | closed at 6 clients; re-open at ≥9 clients or with cross-cohort tail reuse (F-249's own condition) |

## 5. PLAN IMPACT VERDICTS

| item | verdict | evidence |
|---|---|---|
| PIPED_CONCURRENT vs dma | **AMENDED**: dma refusal CONFIRMED mechanical (batch_read.hpp:48 ← generate.cpp:13838; doorbell CUDA calls expert_source.cpp:4776 → verify.cpp:3166-3192; #31 deadlock class documented generate.cpp:9154-9157); but (i) STRATA_DMA_BATCH=0 does not clear the gate — only `--pcie-mode kernel/direct/auto`; (ii) the overlap branch is piped-gated and unreachable on the single-GPU lane (generate.cpp:12024-12025, :13850) — the "one env boot" falsifier boots into a silent no-op + priority side-effect (verify.cpp:921-932). Port M, not env-S | §1.1, §2, §3.1-3.2 |
| prompt-read-beside-windows | **AMENDED**: falsifier NOT runnable without code changes on our lane; basalt's always-on shape (basalt generate.cpp:8012-8107, no gate, no exclusions) is the port template; priority machinery already in-tree (verify.cpp:920-934) | §1.1, §3.2 |
| elastic KV + batch slots | **AMENDED**: donor confirmed (basalt generate.cpp:2964-2967, 3466-3472, 6233-6357; floor 32768, +16384 steps, VMM tail, between-windows only) but incompatible with our LIVE kv-resident streaming by their own constraint (basalt generate.cpp:2962-2963) and excluded by our batch carve (generate.cpp:5197-5198); standalone ≈ +0-2% at probe depths (KV-stream miss 0.6-2.9%, log :94/:103); value = parallel-8 funding. "8090676 blocks" not found anywhere — unverifiable | §1.2, §2, §3.3 |
| ADMIT_SHORT_FIRST | **NEW (arm)**: merged, dormant, env-only (server.py:310-317, 1651, 1670-1688, 1719-1729); 0% conc at 6 fixed clients; TTFT lever at 7+; cheapest first step of the fairness lane | §3.4 |
| maya fair-slice | **CONFIRMED** (fairness-only, 0% conc; port needs a PAUSE verb we lack — generate.cpp:12258-12263 vs maya generate.cpp:1131-1191 — plus slice re-entry under F-14 epochs; M-L). Charter-gated on the 7-12-client lane | §1.3, §3.5 |
| COHORTS>3 | **KILLED at 6 clients** (F-249, all-inside-noise); re-open condition unchanged (≥9 clients or cross-cohort tail reuse). **NEW in its place**: admission overlap INTO the cohort reactor (§3.1) is the named next lever, now with the exact insertion site | F-249; §3.1 |

## 6. THE 320-380 LADDER (where the next 25-65 tok/s come from; ranked by confidence)

Measured anchors: conc 315.7 five-boot (F-248), cohort-window ~11-14 ms (probe TREND lines), cohort A/B
+24.8% (F-233), COHORTS=3 flat (F-249), admissions ≈ 14% of probe wall pre-cohort (T17B #4), KV-stream
miss ~1%, expert miss 2.0-2.3%, F-183 slope ±513 slots = ±4.9% conc.
1. **decode-share raise** (env): hides a further ~20-35% of the 14%-of-wall admission term → +1-4% (ESTIMATE).
   Confidence M-H (mechanism in-tree, one env; bounded by the borrow-shrunken budget windows, §2).
2. **PIPED-for-cohorts port**: removes the residual admission stop entirely → +3-6% net (ESTIMATE) →
   ~335-350. Confidence M (two trees ship it; our port is new code; dma tax unmeasured on our arms).
3. **`--adapt-every 1`**: +0-2% on fixed prompts. Confidence M-L. ~335-352 combined with 1-2.
4. **parallel-8 with 4+4 cohorts** (funded by elastic-KV-style backing and/or vram-reserve, NOT by the
   expert cache — F-183 slope eats it): the only +15-25%-class mechanism → ~380-420. Confidence M
   (band per T17B #1; the 4+4 window fits kVerifyMaxT=8 exactly; two-graph contention at width 8 rows
   unmeasured — the R2-S3 pilot's question repeats at double width).
Ranking rationale: 1 and 3 are env-cheap and de-risk nothing; 2 is the campaign's named lever and the
largest non-structural win; 4 is the only route past ~360 and its funding decision (elastic vs reserve)
should be taken with §3.3's falsifier data in hand.

## 7. UNKNOWNS (graded)

- **Admission share of wall on the CURRENT cohort recipe** — T17B's 14% predates cohorts; likely higher
  now. One boot's log answers. Confidence in the 14% figure applying today: L; in the mechanism: H.
- **dma→kernel tax on our mixed-arm conc config** — upstream 1-3% decode at 8 requests
  (generate.cpp:9157); ours at 2.0-2.3% miss likely sub-1% (ESTIMATE). Unmeasured. M.
- **Two-graph SM contention at 4+4 (rows 8 per cohort)** — the R2-S3 contention number was measured at
  3+3/6 rows; width-doubling unmeasured. M.
- **Probe conversation depth distribution** — inferred ≤ ~32k-dominated from the 97-99% VRAM hit lines;
  no per-request depth log exists. H (for the conclusion), via aggregate evidence only.
- **basalt conc evidence** — none published (their 324.6 is a solo decode window; T25-3 §7.3): the +6-8%
  rests on OUR T17B pricing plus their shipped-default status, not on their table. M.
- **"8090676 blocks"** — not found in tree, configs, or logs. Unverifiable; the live quantities are 9469
  expert-cache slots / 17.96 GiB and 131072-cell K/V with 32768 resident. H (that it is not a tree fact).

## 8. METHODS NOTE

Read-only session; no builds, boots, GPU, or server contact. What was actually read at source this
session: our generate.cpp (12024-12296 pump/main loop, 13690-14203 read_part, 11496-11740 cohort round,
9100-9170 cohort init + pcie-mode comment, 5185-5218 kv gates, 790-860 verify carve), batch_read.hpp
(whole), verify.cpp (790-820, 920-1010, 2080-2199, 3140-3200, 3689-4097), expert_source.cpp (4700-4790),
verify.hpp (240-316, kVerifyMaxT in verify_kernels.hpp:21), serve/server.py (300-400, 1141-1143,
1640-1748), launch-custom-iq3s-8081.sh (whole), campaign/cohorts-iq3_s-8081.json (pcie/reserve lines),
custom-iq3s-cohorts.log (boot + steady-state lines cited), campaign/results/abab-cohorts2.log;
basalt generate.cpp (2960-2967, 3462-3560, 5150-5260, 6233-6357, 7990-8131), basalt verify.cpp (973,
1044-1090, 2072, 3020-3030, 3284-3285), basalt verify.hpp:118, basalt docs/running.md:5-31; maya
serve/server.py (141-235, 986-996, 2103-2260, 3721), maya generate.cpp (1131-1191, 1836-1851). Unproductive
searches (do not repeat): "8090676" anywhere in our tree (no hit); PAUSE in our engine (absent);
set_external_serve in basalt (absent — their design needs no doorbell thread). Estimates are marked
(ESTIMATE); no numbers were invented beyond the marked bands and the arithmetic stated with them.
