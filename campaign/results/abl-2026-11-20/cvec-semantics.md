# Control-vector runtime semantics and provenance — armed refusal-r.gguf on :8081

Task: `task-1` (cvec-semantics). Workspace `/home/svg/LLM/strata-custom-2`, CUDA engine live on 8081.
Scope: static source read + GGUF header read (owner tool `tools/gguf_reader.py`) + live telemetry read.
No engine/config/serve code changed, no restart. Every citation is `file:line` in this tree.
The Lead owns the numerical/behavioural analysis; this paper owns the mechanism and the provenance.

**Headline findings (details below):**

1. The armed file is **one 2560-d vector replicated 47 times**, not 47 per-layer directions. All 47 tensor
   payloads are bitwise identical (verified from the GGUF data section). `--cvec-dir per-layer` and
   `--cvec-dir single:47` therefore produce **identical tables and identical math** for this file.
2. Project mode is `h ← h − s_l (h·v_l) v_l`, applied **per hyper-connection stream**, **after** each
   block's FFN residual write. Scale is the **norm multiplier**, not a blend: 0.5 → half removal,
   1.0 → full removal (armed), 2.0 → component inverted (200% removal). The sign of the scale is lost.
3. Layer mapping is `direction.i → block i`; block 0 is unsteered by construction, block 47 is the last
   steered block, and its projection lands **before** the final norm/output head.
4. The per-request on/off switch is a **single per-device flag**, so concurrent batch slots with mixed
   `experimental_speed_projection` values are **not isolated** (a real, structural gap; the vector is
   default-on, so it only bites mixed traffic).
5. Provenance is **broken/drifting**: the record calls it "the huihui vector carried from the stock
   install", documents nothing about how it was produced, and every "canonical home" named in the record
   (repo root, `data/vectors/`, `/home/svg/LLM/data/vectors/`) is absent; the live file is untracked under
   `/mnt/DATA/...`. The shipped `data/experimental-speed-projection/README.md:19-20` claim that its repack
   is byte-identical to the published "refusal" vector is **refuted** by sha256 and by content.
6. Strength/coverage are **boot-time only**. The one graded lever is the `:SCALE` in
   `--control-vector-scaled`; the per-request key is on/off only.

---

## 0. Live armed state (Q7)

| Evidence | Where | Value |
| --- | --- | --- |
| Engine boot line (live boot, 16:28:07) | `logs/custom-iq3_s.log:105874` (start at `:105868`) | `strata generate: control vector mode = project, dir = per-layer, layers 1..47 (47 steered)` |
| Same line, every earlier boot | `logs/custom-iq3_s.log:7`, `:17`, `:175`, `:10379`, … | identical |
| Serve-side Layer-2 gate verdict | `logs/boot-restore-1629.out:17` (the live boot) | `[strata] control vector: armed project:1-47 (validated against the config)` |
| Engine INFO facts (live) | `GET /metrics` (engine block; source `src/program/generate.cpp:11633-11660`) | `"cvec": "project:1-47"`, `"cvec_steered": "47/48"` |
| Per-request history flag (live) | `GET /metrics` → history; source `serve/server.py:6008` | every request `"projection": true` |
| Engine version / geometry | `GET /metrics` | `engine 0.1.46`, `max_context 262144`, `batch_slots 6`, `prefill_stages 1` |

The engine **does** print the `INFO … cvec=<summary> cvec_steered=<S>/<N>` line (`src/program/generate.cpp:11633-11648`),
but the server consumes INFO lines into `engine.info` during the ready wait (`serve/server.py:1213-1216`) and never
echoes them, so `INFO cvec=` is **not** present in `logs/custom-iq3_s.log` (grep: 0 matches); the live values above came
from `/metrics`, which serves that dict. `/health` does **not** expose cvec (only status/context/model/images/prewarm).
The handshake gate is what prevents an inert or mismatched arm from serving — `serve/handshake.py:109-167`, called at
`serve/server.py:8902-8913`; live verdict printed by `serve/server.py:8912-8913`.

Geometry corroboration: `src/kernels/cvec_parity.cpp:57` fixes the self-test shape `L = 48, N = 2560, HC = 4`, and the
live `cvec_steered=47/48` matches "47 of 48 blocks steered".

---

## 1. Loader — index mapping, range, formula, scale, `single:L` (Q1)

`load_control_vectors` = `src/program/generate.cpp:1951-2036`.

### 1.1 `direction.<l>` → block `l` (n_layers = 48)

- The tensor name is parsed after `"direction."` (`src/program/generate.cpp:1970-1971`).
- **The layer index IS the model block index.** Indices `< 1` or `>= L` are skipped:
  `if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp`
  (`src/program/generate.cpp:1972`). Each row lands at `data[l*N + j]` (`:1978`).
- The tables are indexed by model layer in the kernel: `dir + layer * n` (`src/kernels/cuda/cvec.cu:56`),
  `s_l[layer]` (`:53`), and the application decision is `cvec().covers(l)` =
  `l >= first && l <= last && steered[l]` (`include/strata/kernels/cvec.hpp:35`).
- So with `L = 48`: `direction.1 … direction.47` → blocks 1 … 47; **block 0 never loads a direction** and is never
  steered; there is no block 48 in the model, so nothing is dropped on the high side for this file/range.

`--control-vector-layer-range 1 47` (`config/custom-iq3_s-8081.json:42-44`) sets `cvec_first = 1`, `cvec_last = 47`.
Loader clamping (`src/program/generate.cpp:1988-1989`): `first = (<=0 ? 1 : first)`, `last = (<=0 || >=L ? L-1 : last)`.
Selection is **inclusive**: `for (l = first; l <= last; ++l)` (`:1997`). Here that is blocks **1..47 = 47 steered**;
the default (flag absent) is `1 .. L-1` = the same 1..47.

### 1.2 The exact project-mode formula

Loader, project mode (`src/program/generate.cpp:2001-2007`):

```
d_l = Σ_files scale_f · raw_f.direction.l           // raw row, N = 2560 f32; scale multiply at :1978
s_l = ||d_l||₂                                       // :2002-2004
v_l = d_l / s_l                                      // unit direction, :2007
```

Kernel, per token `t` and per hyper-connection stream `c` (grid is `(hc, T)`; `hc = 4`, `N = 2560`;
`src/kernels/cuda/cvec.cu:46-56`, `:167-170`):

```
R ← R + bo_t ⊙ 2σ(inj_c / hc)                        // only when write=true: the pending FFN write, cvec.cu:57,65-67
dot = Σ_d R[t,c,d] · v_l[d]                          // cvec.cu:61-69
dot = dot · s_l                                      // cvec.cu:84
R[t,c,d] ← R[t,c,d] − dot · v_l[d]                   // cvec.cu:86-94
```

i.e. **h'_{t,c} = h_{t,c} − s_l (h_{t,c}·v_l) v_l**, per stream. The dot is taken **within each 2560-d stream**,
not across the concatenated 10240-d residual (the header states this: "every one of the hc residual streams h of
every token becomes", `include/strata/kernels/cvec.hpp:5-9`). The component along `v_l` after the update is
`(1 − s_l)(h·v_l)`.

Add mode (`--cvec-mode add`, stock llama.cpp semantics) is `h' = h + d_l` (`s_l = 1`, `dir = d`;
`src/program/generate.cpp:2008-2011`, `src/kernels/cuda/cvec.cu:91`).

Kernel/loader agreement is asserted by the host-reference self-test `src/kernels/cvec_parity.cpp:1-6`
(project against double precision; the steered component is gone; a loaded-but-off vector is bitwise the stock
engine; a layer without a direction is untouched).

### 1.3 Scale semantics in project mode (the loader re-normalizes)

The file `:SCALE` is **not** a blend factor; it multiplies the raw row, and the loader then takes the norm of the
scaled vector as the removal strength:

| `:SCALE` | effective `s_l` (unit rows) | component along `v_l` after | meaning |
| --- | --- | --- | --- |
| 0.0 | 0 → `nrm <= 0` → layer **skipped** (`generate.cpp:2005`); all-zero ⇒ load error (`:2014`) | unchanged | inert |
| 0.5 | 0.5 | ×0.50 | half removal |
| **1.0 (armed)** | **1.0** | **×0.00** | **full removal** |
| 2.0 | 2.0 | ×−1.00 | inversion / "over-ablation" (200% removal) |
| negative, e.g. −1.0 | 1.0 (norm loses the sign) | ×0.00 | **identical to +1.0** in project mode (projection onto a line) |
| several files | `s_l = ||Σ scale_f d_f,l||` | | norm of the **sum**, not the sum of norms; `v_l` = normalized sum (`:1978`, `:2002-2007`) |

Armed value is `:1.0` on a file whose rows are unit norm, so `s_l = 1.0` on all 47 blocks and the direction is fully
removed at every steered block — exactly the state F-376 calls "a full removal per armed layer"
(`SVG-CUSTOM.md:11909-11911`), its evidence file's "complete abliteration"
(`campaign/results/f376-abliteration-fix.txt:11-14`).

### 1.4 `--cvec-dir per-layer` vs `single:<l>`

- Parse: `src/program/generate.cpp:2867-2871`. In the loader: `single = (mode == 0 ? o.cvec_single : -1)`
  (`:1990`) — **`single:L` is project-mode only; in add mode the flag is silently ignored.**
- Validation (`:1991-1994`): `L` must be `< n_layers` and `direction.L` must exist, else the load fails.
- Applies `direction.<single>`'s row to **every** layer in `[first,last]`: `src = single >= 0 ? single : l`
  (`:1998`), so the same `v` and the same `s` are written to every steered layer. Layers' own directions are ignored.
- Summary/digest encode it (`:2021`, `:2028`), so a `single:L` arm has its own session identity even when the tables
  coincide.
- **For the armed file, per-layer and `single:47` are mathematically identical** (all rows are bitwise equal — §5),
  and therefore also byte-identical `dir`/`s` tables. Only the digest's `single` field differs.

Reference semantics (llama.cpp): the fork's loader implements llama.cpp's `common_control_vector_load` placement —
raw `direction.<l>` at row `l`, layer 0 absent — and the project-mode patch documented in `cvec.hpp:1-12`. Stock
llama.cpp's loader places `direction.<l>` at `data + n_embd*(l-1)` with a reject for index 0
(`third_party/llama.cpp/common/common.cpp:2039-2044`, `:2069-2070`); stock apply is additive
(`third_party/llama.cpp/src/llama-adapter.cpp:24-31`).

---

## 2. Where and when it applies — prefill, decode, fused path, last layer (Q2)

### 2.1 Application point: after each block's FFN residual write

The residual write being followed is the block's **FFN (second half) hyper-connection write**, and the steering lands
on the value the next layer's attention norm reads. Sources:

- Contract: "After layer l's FFN write — llama.cpp's `build_cvec(res_hc, il)` right after the layer's second
  `build_hc_combine`" (`include/strata/kernels/cvec.hpp:5-9`).
- Prompt path: the half loop is `for (half = 0; half < 2; ++half)` (`src/prefill/prefill.cpp:3127`); the vector is
  applied only on `half == 1` (`const bool steered = half == 1 && ...covers(l)`, `:4770`), i.e. after the FFN write,
  and the write is fused with the **next half's norm** when possible (`:4765-4782`).
- Decode path: `block_layer_post` — `const bool steer = cvec().covers(layer); // after this write`
  (`src/core/layer.cpp:1639`), then `gr_write` → `cvec_apply(..., write=false)` (`:1641-1646`) or
  `cvec_apply(..., write=true)` which performs the deferred write **inside** the cvec kernel (`:1647-1650`,
  formula at `src/kernels/cuda/cvec.cu:57-66`).
- Verification/speculative windows: identical rule — last layer `gr_write…` then cvec (`src/core/verify.cpp:2371-2376`);
  other steered layers `cvec_apply(write=true)` (`:2377-2379`).
- The write-fold is disabled upstream of the vector so the vector never sees an unwritten R:
  `pending_ffn = fused && layer > 0 && !covers(layer-1)` (`src/core/layer.cpp:1503-1504`),
  `pending = l > 0 && !covers(l-1)` (`src/core/verify.cpp:1339-1341`), and the prompt path's fusion gate
  (`src/prefill/prefill.cpp:4773-4776`).

### 2.2 Token positions steered

- **Prefill: every prompt position, every chunk.** The chunk loop is `for (c0 = 0; c0 < n; c0 += C)`
  (`src/prefill/prefill.cpp:2673`) and each chunk's `T = min(C, n - c0)` tokens (`:2682`) go through every layer;
  `cvec_apply(m.R, l, T, D, …)` covers all `T` rows (`:4828`; fused equivalent `:4816-4818` with grid `T*HC`,
  `src/prefill/kernels.cu:2564-2569`). Image-embedding rows are rows of the same prompt and are steered like text rows.
- **Decode: every emitted token** (one row per token), and in a speculative window **every candidate row** —
  `cvec_apply(Rt(tb), l, n, …)` with `n = te-tb` (`src/core/verify.cpp:2375/2378`); `T=1` for plain decode.
- No position is skipped by any code path that runs the block stack.

### 2.3 direction.47 vs the final norm / output head

Block 47 is the **last steered block**, and its projection is applied to the residual **before** the head:

- `src/core/verify.cpp:2371-2376`: at `l == g.n_layers - 1`, `gr_write_multi`/`gr_write` materialises R, then
  `cvec_apply` (no write). The head then runs after the layer loop: final mixer/norm/`lm_head` at
  `src/core/verify.cpp:2440-2468`; the fused-head shortcut is explicitly disabled when the vector covers the last
  layer: `fuse_head_gr = … && !cvec().covers(g.n_layers - 1)` (`:2189-2194`).
- So the last block's output — the residual the final norm and output head consume — **is steered**. The vector is
  not applied after the head or to logits.
- llama.cpp's equivalent ordering: `cur = ggml_add(cur, ffn_inp); cb(cur,"ffn_out",il); cur = build_cvec(cur, il);`
  then after the layer loop `build_norm(cur, model.output_norm …)` and `lm_head`
  (`third_party/llama.cpp/src/models/apertus.cpp:147-164`).

### 2.4 `STRATA_CVEC_FUSE` fused path — bit-equivalence, skips, double-apply

- Gate: `src/prefill/prefill.cpp:461-462` (`getenv("STRATA_CVEC_FUSE")` = `'1'`); fused choice at `:4770-4776`;
  kernel `gr_write_cvec_norm_rs_kernel` at `src/prefill/kernels.cu:209-270`.
- **Not armed on this box.** The default-on switch for `STRATA_CVEC_FUSE` lives only in the AMD arch-default table
  (`src/core/arch_defaults.cpp:28` gfx12, `:42` gfx1151), and that table "is never consulted by a CUDA build"
  (`src/core/arch_defaults.cpp:19-20`); `launch-custom-iq3s-8081.sh` does not export it (grep: 0). The live CUDA
  process therefore runs the three-step sequence (`gr_write` → `cvec_apply` → next norm), not the fused kernel.
- Bit-equivalence is by construction and instrumented: same thread-to-element map (`d = tid + 256k`), same two-level
  xor reduction, same `sigm = 1/(1+expf(-x))`, same norm order (`src/prefill/kernels.cu:209-212`, `:47`, `:224-268`;
  parity claim in `src/kernels/cvec_parity.cpp:1-6`). `STRATA_CVEC_FUSE_CHECK=N` runs the fused and unfused paths on
  a copy of R and compares R, row scales and the BF16 image (`src/prefill/prefill.cpp:4783-4814`) — the in-tree
  instrument for the "bitwise" claim.
- **No skip, no double-apply.** `cvec_apply` on a steered layer runs only in the `else` branch when not fused:
  `if (steered && !cv_fused) cvec_apply(...)` (`:4827-4828`), and the fusion itself is only allowed when
  `(!steered || cv_fused)` (`:4776`). Fused and unfused branches both cover the same layer range, the same `T` rows
  and the same `hc` streams; the only consequence of fusion is the next half's norm being computed from the already
  steered value (which is also what the unfused order does).

Add-mode `--cvec-dir` is ignored (see §1.4); add-mode fused kernel handles `mode == 1` with the same code path
(`src/prefill/kernels.cu:255`).

---

## 3. Paths that emit tokens or update state WITHOUT the vector (Q3)

| # | Path | Steered? | Behavioural consequence |
| --- | --- | --- | --- |
| 1 | **Block 0** (embedding → block 0 attn/FFN) | No, by design (no `direction.0`) | The residual entering block 1 is unsteered; steering begins after block 0's FFN write. Where the 47-of-48 count comes from. |
| 2 | **MTP draft proposals** (`mtp.prefill`, `mtp.draft`, `draft_first`) | Drafter forward contains **no cvec at all** (`src/core/mtp.cpp` has zero cvec references) | Proposals only; **no emitted token bypasses the vector**. Every committed token comes from the steered main verifier. Consequence is efficiency only: the drafter is unsteered but is conditioned on the steered final residual (`mtp.prefill(ver.final_R_all(), …)`, `src/program/generate.cpp:16962`). |
| 3 | **Suffix/lookup drafts** (`--suffix-draft`, `--lookup-chain`, n-gram sources) | Proposals only | Same as #2: the window is verified by the steered model (per-window "drafts accepted a of b" log lines). |
| 4 | **Batch slots / concurrent requests** | Rows are steered, **but** the switch is one per-device `int` (`src/kernels/cuda/cvec.cu:14-19,28,137-150`) | **Real gap:** all windows use whatever the single flag says. A new request's admission sets the flag before its prompt read (`src/program/generate.cpp:14097`, `:14591`), while other slots keep decoding during the read (`batch_step()` inside the read, `:15581-15587`; piped variant `:15206-15210`). Mixed `experimental_speed_projection` values across concurrent requests are therefore **not isolated** — one request's opt-out can unsteer another slot's windows for the duration (and an opt-in can re-steer an opted-out slot). Cache state is not cross-contaminated (keys below), only the compute flag is global. Default traffic is all-on, so this is latent until mixed opt-out traffic. (Static analysis of the flag plumbing and the read/decode interleave; not exercised live in this session.) |
| 5 | **Vision encoder** | The encoder (`engine/strata-vision`, `config:79-84`) produces embeddings only; it never emits text | No unsteered generation path. Image embedding rows re-enter the prompt and **are** steered by the main model like text rows. |
| 6 | **Conversation-cache restore / replay / parked sessions** | Restored K/V is not recomputed — and that is safe: entries carry the cvec state and are only matched against a request with the same state (`include/strata/core/conversation_cache.hpp:171`, `:315-321`, `:335-337`, `:465-478`); on a switch the engine clears the live session and checkpoints (`src/program/generate.cpp:14586-14590`); disk sessions carry the **full vector digest** in the session config fingerprint (`src/program/generate.cpp:9780-9781`, `:9833`; written/read at `src/core/conversation_file.cpp:896`, `:691`, `:770-776`) | None for output identity: a session built with the vector on cannot be reused with it off (and a differently-packed vector is a different digest). Theoretical exception: `--control-vector-phase` (F-17, not armed) creates a per-token mixed on/off state the cache key does not represent. |
| 7 | **Early exit / non-finite window** | The window's tokens are discarded (no token emitted) | The `#879` guard prints `nonfinite` and ends the request (`src/program/generate.cpp:16952-16960`); no unsteered token escapes. |
| 8 | **PLE block at layer 1** (residual add before layer 1's read) and the embedding residual | No vector on that add | By design: the PLE add is inside the unsteered span between block 0 and block 1's read; later blocks steer on top of it. |
| 9 | **`--prefill-head-alias`** (not armed here) | Head aliased onto the prompt path **after** the full layer stack | Even if armed, the last block's cvec has already been applied (`src/prefill/prefill.cpp:4825-4828`) before the alias head reads R. |
| 10 | **`STRATA_CKPT_REREAD=1`** (opt-in) | Re-reads the prompt instead of restoring | Steered (it is a normal prompt read). |
| 11 | **Router/profiling/adaptive-tier state** | Statistics only | No token content. |

**Explicit answer to "any serving path that emits tokens without the vector": none.** Every emitted token is produced
by the steered main-model verifier (decode or the first verify window after the prompt read); the exceptions are
(a) *proposals* that are never emitted without verification, (b) *restored states* that are provenance-bound to the
vector's state/digest, and (c) **block 0's own compute**, which is unsteered by design. The one genuine
configuration hazard is #4: the per-request switch is global under concurrency.

Also worth naming for the Lead: `STRATA_ROUTE_TAIL_SKIP=7` (launcher line 35) is an output-changing knob unrelated
to cvec; it is set to skip rank-≥7 CPU-missed experts. It is not an unsteered path but it is the other
standing output-shaping switch — do not confuse its effect with the vector's.

---

## 4. Layer-index convention, layer 0, handshake, boot evidence (Q4)

- **Convention: `direction.i → block i`, block 0 unsteered.** Loader skips `l < 1` and `l >= L`
  (`src/program/generate.cpp:1972`); rows are indexed by the same `l` in the range loop (`:1997-1998`) and the kernel
  indexes `dir + layer*n`, `s[layer]` by model block (`src/kernels/cuda/cvec.cu:53,56`); `covers(l)` gates the write
  at block `l` (`include/strata/kernels/cvec.hpp:35`). Range 1..47 with `L=48` ⇒ blocks 1..47; block 0 never steered;
  block 48 does not exist.
- **Why llama.cpp leaves layer 0 empty.** By construction, not by omission:
  `tensors.push_back(nullptr); // there's never a tensor for layer 0` (`third_party/llama.cpp/src/llama-adapter.cpp:66-67`);
  the loader rejects a `direction.0` outright ("invalid (zero) direction tensor layer index",
  `third_party/llama.cpp/common/common.cpp:2039-2044`); and the apply buffer has no slot for it
  (`const size_t off = n_embd*(il-1); // buffer doesn't have data for layer 0, since it's never present`,
  `:126-127`). The apply site is after each block's output (`build_cvec(cur, il)`,
  `third_party/llama.cpp/src/models/apertus.cpp:147-153`), so there is no "after block -1" write for index 0 to
  attach to; the vector is a between-blocks residual edit. The in-tree producing tool likewise exports `n_layers-1`
  tensors: it discards the final layer (`third_party/llama.cpp/tools/cvector-generator/cvector-generator.cpp:101`,
  `:192`) and names the tensors `direction.<il+1>` (`mean.hpp:18`, `pca.hpp:305`) — i.e. direction.1 is the first
  collected hidden layer, direction.0 is never written. (Caveat for the Lead: the generator's first collected
  hidden layer is block 0's output, named `direction.1`, while the runtime applies `direction.i` after block `i`;
  a generator-built file therefore carries a one-block alignment question (static reading of the vendored tool; not
  verified against the producing fork). It does not change the runtime mechanism and nothing in the record says
  refusal-r came from this tool.)
- **Handshake check.** `serve/handshake.py` is the single schema owner: it parses the engine's `IDENT <json>`
  (`:39-47`, schema `:11-18`), falls back to the legacy `INFO cvec=`/`cvec_steered=` (`:72-89`), derives what the
  config **asked** (`:92-106`), then refuses to serve on: no arm when asked, mode mismatch, range mismatch, a range
  past the model's layers, or `steered == 0` (`:143-167`). The gate is wired at `serve/server.py:8902-8913`
  (`armed_geometry_problem`). It is checked on the engine's **measured** `cvec_steered`, not the asked range
  (docstring `:159-161`).
- **Boot log evidence** (exact string, live boot): `logs/custom-iq3_s.log:105874` —
  `strata generate: control vector mode = project, dir = per-layer, layers 1..47 (47 steered)`; every earlier boot
  prints the same (`:7`, `:17`, `:175`, …). The serve side's verdict line is `logs/boot-restore-1629.out:17` —
  `[strata] control vector: armed project:1-47 (validated against the config)`. Live telemetry:
  `engine.cvec=project:1-47`, `engine.cvec_steered=47/48` (`/metrics`). All three agree.

---

## 5. Provenance (Q5)

### 5.1 Copies and hashes

| Path | sha256 | Notes |
| --- | --- | --- |
| `/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-refusal-r.gguf` (the armed file, `config:40-41`) | `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38` | 483,520 B, mtime Oct 16 01:41 |
| `/home/svg/LLM/strata/refusal-r.gguf` | `59f9ae8b…` (identical) | 483,520 B, mtime Oct 17 20:30; separate inode, not a hardlink |
| `/home/svg/LLM/data/vectors/qwen3.8-flash-next-refusal-r.gguf` | **absent** (`/home/svg/LLM/data` itself does not exist) | yet named as the canonical home by `campaign/README.md:38-39` and `.docs/research/2026-11-15-estate-consolidation-study.md:44,108` |
| `data/vectors/…` in-repo | **absent from every commit** (`git log --all -- data/vectors` = 0; the tree at F-380 has no `data/vectors/`) | F-380's message claims it moved the vector there |
| `data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf` (shipped package) | `ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b` | 483,520 B — **not** the armed file (§5.3) |

Git lineage of the exact bytes: fork snapshot `848f73a97` carried `refusal-r.gguf`; F-211 `d11342319` purged it
(and the first post-reboot boot died: F-212); F-212 `2dcf946e5` restored it — `git show 2dcf946e5:refusal-r.gguf`
is sha256 `59f9ae8b…`, i.e. **the live file's exact bytes** — then F-380 `a60e6f29e` deleted the repo-root copy
("data/vectors/ (steering vectors: qwen refusal-r from the repo root …)", and its diff deletes `refusal-r.gguf`);
no `data/vectors/` path exists in that tree. The artifact now lives untracked outside git. The record's own
recovery note (`.docs/research/2026-11-15-estate-consolidation-study.md:108`) says the `strata/` copy "deleted as
sha-identical to `data/vectors/` copy", but the `strata/` copy still exists and `data/vectors/` does not — the
record and the box have drifted apart (closure hazard, AGENTS.md law 13).

### 5.2 What the record claims about production

- `SVG-CUSTOM.md:258-259` (F-04 recipe): "cvec project 4-44 (`refusal-r.gguf`, **huihui vector**): carried from the
  stock install (Pass 173b-175 audits; ≤3% cost, per-request toggleable)." — i.e. imported from a pre-existing
  stock install, attributed by a name ("huihui"), cost-audited, made per-request toggleable. **No producer, URL,
  model, method, or prompt set is named.**
- `SVG-CUSTOM.md:11904-11916` (F-376): the file "carries 47 unit-normalized directions (`direction.1..direction.47`,
  `|v| = 1.0`; `layer_count=47`)"; the runtime formula and scale semantics; the range fix 4..24 → 1..47; the measured
  compliance table (temp 0, 1400-token budget: phishing/AV-bypass/ransomware refused at 4..24, complied at 1..47).
  This is the **validation** record, not a production record.
- `campaign/results/f376-abliteration-fix.txt:6-14` repeats the shape and formula and calls scale 1.0 + full range
  "complete abliteration".
- Nothing else in `campaign/README.md`, `.docs/research/`, `docs/`, or the journal describes how the vector was made.

### 5.3 What the packed provenance *would* claim — and why it does not apply

`data/experimental-speed-projection/README.md:19-20` says the shipped package is "a local package of the vector
(the published original uses filenames containing 'refusal'; **the bytes are the same**)". The JSON beside it is a
full recipe card for that package: `mode=project`, `dir=per-layer`, `scale=1.0`, `layer_range=[4,44]`, refusal
counts, `kl_mean 0.186`, `base_model_gguf unsloth/Qwen3.8-Flash-Next-GGUF UD-Q2_K_XL`, shard sha,
`worktree_commit 37e026e3b`, `branch cvec-hooks (github.com/Cudila/llama.cpp)`, note "projection ablation
h -= s*(h.v)*v, v unit-normalized per layer" (`…experimental-speed-projection.json:2-25`); `docs/HOW_IT_WORKS.md:76`
adds "Qwen Community License 1.0, from the model's activations" (activation-diff).

**That provenance cannot be transferred to the armed file.** Measured from the GGUFs with `tools/gguf_reader.py`:

| Property | armed `/mnt/…refusal-r.gguf` | shipped `data/experimental-speed-projection/…gguf` |
| --- | --- | --- |
| header | `controlvector`, hint `qwen4exp`, `layer_count=47` | identical |
| tensors | 47 × `direction.1..47`, F32, 2560 each, each unit norm | identical shape |
| per-row content | **all 47 rows bitwise identical** (one direction replicated; md5 per row `8b811e68…`) | 47 **distinct** rows (pairwise cos min 0.128 / mean 0.654 / max 0.988; 47 unique payloads) |
| cross-file | — | and not a repack of the armed file: the armed direction's best alignment with any of the 47 ESP rows is |cos| = **0.773** (the README's "bytes are the same" claim is false for this pair) |

Consequence for the mechanism: with the armed file, "47 per-layer directions" is nominal. The runtime removes
**one** direction at **47 depths**, at full strength each time. `--cvec-dir single:47` would give the same tables.
This is the single most important thing the record does not say, and it is the difference between "a layered
per-layer extraction" and "a single-direction abliteration applied at every block".

### 5.4 What is NOT documented (flags)

For the **armed** vector:

1. **Extraction method** — activation-diff (mean / PCA) vs weight-diff (task arithmetic / abliteration of weights) is
   nowhere stated. The only in-tree producing tool is llama.cpp's `cvector-generator` (mean-of-differences or PCA
   over positive/negative prompt completions, from hidden states:
   `third_party/llama.cpp/tools/cvector-generator/cvector-generator.cpp:356-373`, `:479-510`, `mean.hpp:18`,
   `pca.hpp:305`), but no record ties it to refusal-r.
2. **Prompts / completions used** — none recorded.
3. **Base model + quantization** — no model GGUF name, repo, or shard sha for the armed file (the JSON's
   `unsloth/…UD-Q2_K_XL` card belongs to the other package).
4. **Producer** — "huihui" is a name in one journal line; no URL, repo, commit, or hash.
5. **Why one direction replicated 47 times**, and whether that is intended or a packaging artifact.
6. **Scale/range provenance** — 1.0 / 1..47 is a campaign decision (F-376), not the producer's recommendation; the
   shipped package's own selected config was `scale 1.0, layers 4..44` (json:2-8).
7. The `controlvector.layer_count=47` metadata is the only self-description the file carries.

Caveat on the numeric comparison above: it is a file-content measurement (GGUF header + f32 payload comparison,
numpy), reproducible with `tools/gguf_reader.py`; the Lead's numerical analysis owns the interpretation of
collinearity, orthogonality and KL. It is reported here because it changes what the *mechanism* is doing.

---

## 6. Knob inventory — strength/coverage, boot-time vs per-request (Q6)

| Knob | Where parsed | When | Effect |
| --- | --- | --- | --- |
| `--control-vector-scaled FILE:SCALE[,FILE:SCALE…]` | `src/program/generate.cpp:2834-2847` | **boot-time** | strength; per-file scale multiplies the raw row, loader re-normalizes → `s_l = ||Σ scale·d||`; comma list sums several vectors (`:1957`, `:1978`) |
| `--control-vector FILE` | `:2833` | boot-time | shorthand, scale 1.0 |
| `--control-vector-layer-range A B` | `:2849-2851`; defaults `:1988-1989` | **boot-time** | coverage; inclusive; default 1..L-1; clamps `A<=0→1`, `B<=0 or B>=L→L-1` |
| `--cvec-mode add\|project` | `:2861-2866` | **boot-time** | add (`h+=d`) vs project (`h−s(h·v)v`) |
| `--cvec-dir per-layer\|single:L` | `:2867-2871`; loader `:1990-1998` | **boot-time** | which row feeds which layer; project only; silently ignored in add mode |
| `--control-vector-phase END[:START]` | `:2853-2860`; runtime `:14595-14597`, `:16969-16985` | **boot-time** (token ids) | F-17 phase-aware: request starts **unsteered**, arms at the think-END token, re-arms off at think-START. Not armed in this config. |
| `STRATA_CVEC_FUSE=1\|0` | `src/prefill/prefill.cpp:461-462` | boot (env) | prompt-path kernel fusion (cost only, bitwise); default only on AMD gfx12/gfx1151 (`src/core/arch_defaults.cpp:28,42`); **not armed on CUDA** |
| `STRATA_CVEC_FUSE_CHECK=N` | `src/prefill/prefill.cpp:4783-4814` | boot (env) | verification instrument: fused vs unfused comparison |
| `"experimental_speed_projection": true\|false` (request body) → engine `cvec=0\|1` | server `serve/server.py:1628-1633` (key emitted in `sampling_keys` at `:1626`); engine parse `src/program/generate.cpp:13881`, `:13915`; applied `:14097`, `:14591` | **per-request** | on/off only (no strength). Absent = on when a vector is loaded (`:14097`; engine parser default `req_cvec = 1`, `serve/server.py:1630-1633`). Honoured by **both** `/v1/chat/completions` and `/v1/messages` (and `/v1/responses`): each dialect's body **is** the sampling dict — `_openai` `serve/server.py:7630-7664` → `openai_chunks` `svc.run(…, req, …)` `:6339`; `_anthropic` `:7882-7898` → `anthropic_events` `svc.run(…, req, …)` `:6478`; responses `:7842`, `:7862-7871`. A non-bool in a request body is silently ignored (the key is emitted only for a real bool, `:1632-1633`); the *shared-settings* route validates it at `:8260-8262`. Config default via `"sampling": {"experimental_speed_projection": …}` (`:8442-8446`), merged under request values at `:5574-5579`. |
| Web UI switch | `serve/web/app.js:810`, `:1029` | per-request | sends the same body key |
| — | — | — | **There is no runtime strength/coverage knob.** Strength and coverage are boot-time; the switch is binary. Switching drops the conversation cache once (`src/program/generate.cpp:14586-14590`; `docs/DETAILS.md:2148-2152`). |

### Which knob is the plausible lever for "soft / toy-like / unserious"?

**The boot-time `:SCALE` on `--control-vector-scaled` is the only graded lever**, and the leading candidate.
Because this file has one replicated row and `s_l = scale` at every block, the served model is running the
strongest form this file admits: the same direction fully removed at 47 depths. Scale it down (0.5–0.7) and the
component is removed at 50–70% per layer while keeping the late-layer coverage F-376 fought for; narrow the range
instead (the pre-F-376 4..24) and you lose late-layer coverage, which is the exact regression F-376
(`SVG-CUSTOM.md:11904-11916`) fixed. `--control-vector-phase` is the secondary structural lever (leave reasoning
fully unsteered, arm at think-end) but it needs the template's think-end token id and is not armed.
A zero-restart, one-request falsifier is available immediately: the same prompt with
`"experimental_speed_projection": false` vs omitted — that isolates the vector's contribution to the "soft" feel
before any boot-time change (doctrine: cheapest falsifier first). All of this is a hypothesis for the Lead's
numerical/behavioural work, not a measured conclusion.

---

## 7. Answers index (Q1–Q7)

1. **Q1** → §1: `direction.i → block i`; range 1..47 selects 47 steered blocks (0..47 model); project formula
   `h ← h − s_l(h·v_l)v_l`, `s_l=||d_l||`, `v_l=d_l/||d_l||`, per hyper-connection stream; `:SCALE` is a norm
   multiplier (0.5 half, 1.0 full, 2.0 inverted; sign lost); `single:L` replicates row L to every steered layer
   (project only) and is math-identical to per-layer for this file.
2. **Q2** → §2: prefill steers all prompt positions in every chunk; decode steers every window row; application is
   **after** each block's FFN residual write; `direction.47` lands before the final norm/head; `STRATA_CVEC_FUSE`
   (not armed on CUDA here) is bit-equivalent, with no skip and no double-apply.
3. **Q3** → §3: **no token-emitting path bypasses the steered verifier.** Proposals (MTP/suffix/lookup drafts) are
   unsteered but never emitted unverified; restored state is cvec-bound (bool in-RAM, digest on disk); block 0 is
   unsteered by design; the real hazard is the global per-request flag with concurrent batch slots (mixed opt-outs).
4. **Q4** → §4: `direction.i → block i`, block 0 unsteered; llama.cpp has no layer-0 tensor by construction and its
   loader rejects `direction.0`; handshake gate validates mode/range/steered against the asked config; boot log
   lines and live telemetry agree on `project:1-47 (47 steered)` / `47/48`.
5. **Q5** → §5: identical sha256 on the two existing copies; no `data/vectors` anywhere; the vector is one
   direction replicated; record claims only "huihui vector carried from the stock install" + validation studies;
   extraction method, prompts, base model, and diff type are undocumented; the shipped package's provenance card
   belongs to a **different** file, and its "bytes are the same" claim is refuted.
6. **Q6** → §6: strength/coverage are boot-time (`--control-vector-scaled :SCALE`, `--control-vector-layer-range`,
   `--cvec-mode`, `--cvec-dir`, `--control-vector-phase`); per-request is on/off only
   (`experimental_speed_projection` → engine `cvec=0|1`), honoured on OpenAI chat, Anthropic messages and
   Responses; plausible lever = the boot-time scale.
7. **Q7** → §0: live state `project:1-47`, `47/48` steered, gate passed, `projection: true` on history entries.
