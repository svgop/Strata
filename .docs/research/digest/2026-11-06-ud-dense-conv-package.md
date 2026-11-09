---
type: digest
id: ud-dense-conv-package-2026-11-06
created: 2026-11-06
status: patch-delivered
topic: The UD dense-side conversion package — HC reads and the final mixer take the copies=0 bf16-rows form on UD packs (STRATA_UD_BF16_DENSE, default on), with the Q8_0 copies form as the =0 fallback; load-time conversion, compile-proven, boot-check named, VRAM accounted
scope: patch built in the scratch worktree /tmp/ud-dense-conv (fork 03615ebf, never the live tree); diff-driven against /home/svg/LLM/strata tip 9f7d977d (engine 0.1.57 line); evidence from campaign/results/dispatch-{ud-iq2xs,iq3s-gsq}.json, logs/custom-ud-iq2_xs.log, logs/custom-iq3_s.log, and the pack/GGUF tensors themselves (parsed, not assumed). Siblings: 2026-11-06-ud-slowdown-kernels.md (the M1/M2/M3 decomposition this package prices), 2026-11-04-past-ceiling.md.
---

# UD dense-conversion package (2026-11-06)

Verdict: **the package is built, linked, and delivered in `/tmp/ud-dense-conv` (patch: `ud-dense-conv.patch`, 5 files, +101/−9).** One mechanism serves both task items: for a pack whose GGUF carries the HC projections as Q8_0 (the UD sets — measured: `hc_attn/ffn_down/up` and `output_hc_down/up` are ggml type 8 in `Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf`; the GSQ GGUF carries the same names as bf16 type 30), the load path now dequantizes them to bf16 rows **once**, so every HC read — the verify windows' `fused_gr_read_multi`, the prefill path's `hc_bf16_view`, and the final mixer's `mix_q8` gate — takes the **copies=0 bf16-rows form**. `STRATA_UD_BF16_DENSE=0` falls back to the Q8_0 copies (the S23 form). The dispatch captures prove the target state is real: UD's capture reads `hc read: weights Q8_0 copies=1` ×16 and `final mixer q8=1` ×8; GSQ's reads `copies=0` and `q8=0`. The package makes UD read GSQ's line.

## 1. Where the forms were decided (the diff-drive, first as instructed)

- **Upstream tip (9f7d977d, the 0.1.57 line) has no Q8→bf16-rows converter.** `diff` of `src/core/native_dense.cpp` fork-vs-upstream shows exactly two upstream-only deltas: the gfx1151 auto-on arm of `hc_q8_env` (HIP-only) and #4741 `bf16_as_q8_0` — the **opposite** direction (a pack's BF16 projection served as Q8_0 like its Q8_0 siblings). Nothing in upstream converts the UD GGUF's Q8_0 HC tensors toward bf16 rows; the bf16-rows condition upstream is simply `hc_q8 == nullptr` (their dispatch line `hc read: weights Q8_0 copies=%ld`, `verify.cpp:1588`), reached only by not arming `STRATA_HC_Q8` and loading the pack's compat-bf16 rows. Ported: nothing pre-existed; the converter is new.
- **The fork's decision sites, now carrying the package:** `src/core/native_dense.cpp` (`load`, the GGUF Q8_0-projections block — was `hc_q8` attach only), `src/program/generate.cpp` (the `add_hc_skip` gate, `:3881`), `src/core/weights.cpp` (`load_hc_q8` is the unrelated `STRATA_HC_REQ8` int8+scales form — untouched), `src/core/verify.cpp` (`:2377` `mix_q8` gate; the fork has no dispatch::notef infra — upstream's per-window lines are not portable, so the package adds boot-time form lines instead), `src/kernels/cuda/fused_gr.cu` (read-only truth: `fused_gr_read_multi` prefers `q8_down` over `w_down` when both are set — the reason the package attaches exactly one form).
- **The "head" of task item 2 is the final HC mixer, not the LM head.** Upstream's line is `head: final mixer q8=%ld` (`verify.cpp:2541`) — `mix_q8` is `output_hc_down/up` read from `hc_q8`. The LM head (`output.weight`, Q6_K = 14 in **both** GGUFs, 521,472,000 B = 2,483,200 blocks × 210 in both boots) has no UD/GSQ divergence on the fork: both run `native_quantize_q8_1` + `native_mmvq(14)` (`native_head.cpp`), and upstream 0.1.57 adds no Q6_K direct form either (their head diff is the host-spill fallback only). The mixer preconvert is covered by the same conversion: `output_hc_down/up` are in `hc_q8_name` (S25), `hc_q8` stays null, `mix_q8=0`, and the mixer takes `head_mix_multi` (bf16 rows, default-on on CUDA).

## 2. What the package changes

- `src/core/native_dense.cpp` — the GGUF-Q8_0 block now branches: under `STRATA_UD_BF16_DENSE` (default on) a type-8 `hc_q8_name` tensor whose arena row was skipped uploads its Q8_0 payload to a stage buffer, runs `fused_gr_dequant_q8_0` (the same kernel the =0 read re-runs every window), syncs, frees the stage, and attaches the rows as `ref.data` (`resident=true`, `bytes=rows` — the `load_hc_q8` convention for non-arena bytes; freed via `weights_` in `~NativeDense`). The `hc_q8` attach (S23) stays intact under the fallback, and the bf16-pack quantize block (#2) is gated to `hc_q8_env() && !ud_bf16_dense_env()` so the forms can never mix (the attach would flip `fused_gr_read_multi` back to `launch_q8`).
- `src/program/generate.cpp` — `add_hc_skip` fires under `hc_q8_requested() || ud_bf16_dense_requested()` (single-GPU, as before): the pack's compat-bf16 rows stay out of the arena, so the runtime holds exactly one copy of the HC weights. Multi-GPU is unchanged (no skip → arena rows serve → package inert), and `STRATA_HC_REQ8` preempts the package by load order (REQ8 attaches `data` first; the package's branch requires `data == nullptr`; `read_multi`'s `q8w = s_down` form is pre-existing and untouched).
- `src/core/verify.cpp` — one boot-time form line in `Verifier::init` (the fork's boot-check; per stage, so a layer split prints each stage's truth).
- `docs/QUANTS.md` — the switch documented beside the `STRATA_HC_Q8` paragraph it modifies.
- `include/strata/core/native_dense.hpp` — `ud_bf16_dense_requested()`.

**Env switches (named):** `STRATA_UD_BF16_DENSE` — `=1`/unset (default) the rows form; `=0` the Q8_0 copies (S23) form. Precedence law: `STRATA_HC_REQ8` (explicit int8+scales opt-in) > `STRATA_UD_BF16_DENSE` > `STRATA_HC_Q8`. The inject rows are untouched (bf16 in the UD GGUF and the arena either way). No `exit()` added; the one GGUF reader owner (`strata::GgufFile`) is unchanged; the fused_gr static-assert wall untouched (no geometry touched).

## 3. VRAM accounting (this model: n_embd 2560, HC 10240, hc_lr 320, 48 layers, vocab 248320)

Per layer 4 tensors × 10240×320 = 13,107,200 elems; plus the final mixer 2 × 3,276,800 = 6,553,600 elems. Total converted set: 635,699,200 elems.

| form | bytes | note |
|---|---|---|
| bf16 rows (the package, =1) | 1,271,398,400 B = **1212.5 MiB = 1.184 GiB** | matches the digest's 1.25-GiB HC bf16 traffic line |
| Q8_0 copies (the fallback, =0) | 675,430,400 B = **644.0 MiB = 0.629 GiB** | matches the upstream boot line "STRATA_HC_Q8=1: 0.63 GiB" exactly |
| **delta** | **+595,968,000 B = 568.5 MiB = +0.555 GiB** | what the rows form costs over the copies form |

Against the armed-UD boot at 262144/q4_0 (vram-reserve 2000 MiB, ~1.5–1.8 GiB free after the caches): **it fits**, with ~0.9–1.2 GiB still free. Two honest notes: (a) versus the fork's **current unarmed boot** the delta is **zero** — the arena today holds the same 1.21 GiB as compat-bf16 rows from `dense.bin`; the package swaps their source (GGUF Q8_0) and frees the arena bytes into the same-shaped rows, so the expert-cache slot count barely moves (today 7691 slots at 2.18 MB/blob; the =1 form reclaims the arena 1.21 GiB into the rows 1.21 GiB). (b) The +0.555 GiB is priced against the **=0 fallback** (the S23 copies form), and it is also the price of running the stock UD recipe (`STRATA_HC_Q8=1`) with the package armed: the rows win, 0.555 GiB over what stock's recipe held. The head/mixer part of the delta is the final mixer's own 12.5 MiB (rows) vs 6.6 MiB (copies) — included in the totals above. Load-time transient: one Q8_0 stage buffer per tensor (max 6.64 MiB), freed after its dequant sync.

## 4. Compile proof

Configured fresh in the scratch (`/tmp/ud-dense-conv/build`, Ninja, Release, `CMAKE_CUDA_ARCHITECTURES=120`, `STRATA_ENABLE_CUDA=ON`, nvcc 13.4, the fork's `.venv` cmake/ninja; `tests/cuda/prefill_mmq_v31_test.cpp` copied from the live checkout because CMakeLists at 03615ebf names it but the file is untracked there — noted for the next commit). Logs: `/tmp/ud-build.log` (the `strata` target), `/tmp/ud-build-all.log`, `/tmp/ud-build-final.log`. All targets link, zero errors; the only nvcc diagnostics are the pre-existing `native_mmvq.cu` `#128-D` template warnings (present before the package):

```
[161/165] Linking CXX static library libstrata_kernels.a
[162/165] Linking CXX static library libstrata_engine.a
[163/165] Linking CXX static library libstrata_prefill.a
[164/165] Linking CXX static library libstrata_qwen35.a
[165/165] Linking CXX executable strata          (exit 0)
[342/342] Linking CXX executable strata-dsv41    (exit 0, all targets)
```

## 5. Boot-check sequence for the operator (GPU was serialized; not booted here)

Armed boot (default, no env) of the UD config must print, in order:

1. `strata: STRATA_UD_BF16_DENSE=1: 1.18 GiB of the GGUF's Q8_0 hyper-connection projections dequantized to bf16 rows at load: the HC read and the final mixer take the copies=0 form (STRATA_UD_BF16_DENSE=0 keeps the Q8_0 copies, 0.555 GiB less VRAM on this model)` — `native_dense.cpp`, the HC read form line. The expected exact bytes: 1.18 GiB (1212.5 MiB).
2. `strata verify: hc read weights bf16 rows copies=0, final mixer q8=0, native head=1` — `verify.cpp`, the head/mixer form line.
3. Absence of `strata: STRATA_HC_Q8=1: ...` (that line now prints only when the fallback actually attached copies).

The stock-engine cross-check (if the capture rig runs the 0.1.57 engine over the same pack): `hc read: weights bf16 rows copies=0` and `head: final mixer q8=0` — GSQ's lines, on UD. Rollback: `STRATA_UD_BF16_DENSE=0` returns the S23 copies boot (line 1 disappears, `STRATA_HC_Q8` semantics resume).

## 6. Boundary and the one falsifier

Not booted (GPU serialized — compile proof only), so three claims ride on the boot: (a) the converted rows equal the pack's compat-bf16 rows bit for bit (expected: both are the same Q8_0 grid rounded to bf16 — QUANTS.md states the compat-bf16 copy "rounds them"; the kernel promotes the fp16 d to f32, multiplies, rounds — the falsifier is the golden-hash mimic compare on the first armed boot: identical to the unarmed boot's hash means the rows are value-identical); (b) the window wait reverts to GSQ's ~0.5 ms class now that UD no longer re-converts Q8_0 per window (the digest's M2: 0–15%, priced after M1's draft-vocab fix); (c) `--expert-cache auto` re-sizes around the new rows without a second-class allocation (the rows load before the cache, same as the arena they replace). Next executable probes: boot the armed config with `STRATA_DISPATCH_REPORT` set and diff the two lines above; then the 6-lane battery with and without `STRATA_UD_BF16_DENSE=0` to price the form at the post-M1 window sizes. Pack-side follow-up the package unlocks: a future UD build can drop the compat-bf16 HC rows from `dense.bin` (−1.21 GiB on disk per pack) with no runtime change on this engine.

## Files

- Patch: `/tmp/ud-dense-conv/ud-dense-conv.patch` (also `git diff` in the worktree); changed: `src/core/native_dense.cpp`, `src/program/generate.cpp`, `src/core/verify.cpp`, `include/strata/core/native_dense.hpp`, `docs/QUANTS.md`.
- Worktree: `/tmp/ud-dense-conv` at 03615ebf; build dir `/tmp/ud-dense-conv/build`; binary `/tmp/ud-dense-conv/build/strata`.
- Build logs: `/tmp/ud-build.log`, `/tmp/ud-build-all.log`, `/tmp/ud-build-final.log`.
- Evidence read: `campaign/results/dispatch-{ud-iq2xs,iq3s-gsq}.json`, `logs/custom-ud-iq2_xs.log`, `logs/custom-iq3_s.log`, `/home/svg/LLM/strata/strata-unsloth-ud-iq2_xs.log`, pack `index.txt` ×2, UD GGUF shards 1–3 and GSQ GGUF shard 1 (tensor-list parse), `/home/svg/LLM/strata/src/core/{native_dense,native_head}.cpp` (tip diff).
