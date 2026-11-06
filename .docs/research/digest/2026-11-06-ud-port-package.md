---
type: digest
id: ud-port-package
created: 2026-11-06
status: evidence
topic: UD3.1 (Unsloth Dynamic v3.1) Qwen-set support ported into strata-custom-2's engine 0.1.46 — patch package, build proof, boot-check sequence
scope: scratch worktree /tmp/ud-port (fork 9728436a); live tree touched only by this digest
---

## 0. Verdict

The fork's engine 0.1.46 already serves almost the entire UD-IQ2_XS contract. Diff-drive against upstream tip (v0.2.11-3, engine 0.1.57) found exactly ONE engine-code gap for THIS set: the Q5_0 expert-dot offset (upstream `aa678f10d`, round 0214) — one of the set's 48 layers (layer 4, IQ2_S gate/up + Q5_0 down) hit the old numerics (whole-expert error 2.1e-2..3.1e-2 vs 1.2e-2 for every other down type). Everything else — the type matrix, the 3-shard loader, the PLE-in-last-shard discovery, the pack v3 reader, the mtp path — is byte-identical or already-present in the fork. The patch package lands: the Q5_0 dot fix (CUDA + SYCL), the tightened parity limit, and upstream's round-0214 parity-test pins (5d56ff8c + c1d5e4d5) so the operator's next GPU window can close the boot proof.

- Patch: `/tmp/ud-port/ud-port.patch` (911 lines; sha256 `4efdd553a329e055fe7283599271267a57bf7f5470f16580048b0f9a48944243`), plus the changed files in the worktree itself.
- Build: `strata` engine target linked to completion — `[165/165] Linking CXX executable strata`, exit 0, binary `/tmp/ud-port/build-ud/strata` (87,042,376 B, reports `strata 0.1.46`), plus all five parity targets linked (`native_expert_parity`, `native_grouped_parity`, `native_dense_parity`, `native_dense_types_parity`, `prefill_mmq_v31_test`), exit 0. Config: Release, `CUDA_ARCHITECTURES=120`, `STRATA_ENABLE_CUDA=ON`, `STRATA_BUILD_TESTS=ON`, `STRATA_Q6K_EXPERTS=ON` — the live tree's `build-fresh` flags (ORCA also OFF there). **Boot proof is PENDING: the GPU is serialized for the operator's benches; nothing below was run on the device.**

## 1. The support map (task items 1-4, each verified in source)

### 1.1 The expert type mixes
On-disk census (`native_experts.txt` v3, 48 layers; tools/gguf_reader.py = the Python owner):

| (gu, down) ggml types | layers |
|---|---|
| IQ2_XXS(16) / IQ4_NL(20) | 3 |
| IQ2_XXS(16) / Q2_0(42) | 5 |
| IQ2_S(22) / IQ4_NL(20) | 21 |
| IQ2_S(22) / Q2_0(42) | 18 |
| IQ2_S(22) / Q5_0(6) | 1 (layer 4) |

Fork coverage at 9728436a, verified line by line:
- `is_iq()` (`src/kernels/cuda/iq_kernels.cu:3025`) carries 16/17/18/20/21/22/23/29/42 + (Q6K build) 14/6; `STRATA_GU_FMTS` has 16..22, `STRATA_D_FMTS` has 20/42/6 — identical to tip minus Gyro trellis 147-149 (not a UD type). Upstream tip adds ONLY 147/148/149 vs the fork here.
- `native_expert_supported(16|22, 20|42|6, 2560, 640)` passes every gate (`iq_kernels.cu:4046`): row bytes nonzero, qk divisibility (2560%256=0 for the IQ2 gu; 640%32 for IQ4_NL/Q5_0, %64 for Q2_0), (n_ff*n_embd)%256=0. This is the same gate `generate.cpp:3555` refuses a layer through — no layer of this set trips it.
- CPU pool (`src/kernels/cpu/native_expert.cpp`): `iq512_supported`/`iq256_supported` carry 16/17/18/21/22; down types delegate to ggml-cpu traits (Q2_0 64/18, Q5_0, IQ4_NL all in the shared ggml lineage; the fork's production Q2_0 shadow store already exercises Q2_0 CPU rows daily).
- Dense side (shard 2 census: Q6_K x85, Q8_0 x410, F32 x388, BF16 x195, IQ4_XS x1 = token_embd): fork `native_mmvq_supported` / the `native_mmvq` switch (`src/kernels/cuda/native_mmvq.cu:2892,2927`) cover every one (14=Q6_K, 20=IQ4_NL, 23=IQ4_XS, 42=Q2_0, 6=Q5_0, i-quants 16/17/18/21/22/29 via `iq_mmvq`) — the same case list as tip.
- Parity pins: upstream `5d56ff8c` + `c1d5e4d5` (the two commits the catalog line UNSLOTH_V31_ENGINE (0,1,57) is tested by) port clean — `native_grouped_parity.cpp` at the fork is byte-identical to c1d5e4d5's parent, so its Q5_0 hunk applied verbatim; the pair-enumeration ctests (24 expert pairs at 2560x640) and the two dense-type tests are added with it.

### 1.2 The loader (3/4-shard sets, PLE in the last shard)
Already present in the fork, no change needed:
- `strata::gguf_split_paths` (`include/strata/artifact/gguf_split.hpp`) rewrites the shard number for any `-NNNNN-of-MMMMM.gguf`; 3 shards fine; a missing shard is a named error.
- `generate.cpp:3316-3345` (fork): opens shard 1's metadata, family-gates, then **finds the PLE table by tensor name** across all shards (`model.find("per_layer_token_embd.weight", &at)` -> `o.ple_gguf = o.native_shards[at]`) and sets `native_dense_gguf = all shards`. This is the same auto-derive stock runs (`--ple-gguf` may be omitted entirely).
- Verified against the real files with the fork's own reader: shard 1 = 0 tensors (pure metadata, 10.9 MB), shard 2 = 1223 tensors (all dense + 48 `ffn_*_exps`), shard 3 = exactly `per_layer_token_embd.weight`. `native_experts.txt`'s per-line shard column (`...00002-of-00003.gguf`) is parsed by the shared `expert_layout` v3/v4 reader (fork's `expert_layout.hpp` is byte-identical to tip's).
- Four-shard sets: same mechanism (the loop is 1..N); upstream's 9ed3c778 "four-shard layout accepted" needs no C++ delta.

### 1.3 The pack format
No drift. The UD pack (`/home/svg/LLM/data/packs/unsloth-ud-iq2_xs`, written by stock's current `tools/iq_pack.py`) and the fork's production iq3_s pack are the same format: `index.txt` rows are 19 fields in both (1079 rows, identical tensor-name sets — all 387 `hc_*` hyper-connection tensors included, same model graph the fork already serves); `native_experts.txt` v3 with the shard column; `dense.bin` + `tokenizer/`. The fork's `weights.cpp` (index parser, kinds 0-5 incl. 4=BF16) is byte-identical to tip's; `tools/iq_pack.py` differs only by tip's interrupt-resume feature for experts.bin writes (no format change). `conversions.json` (compat_bf16: 194x Q8_0->BF16, 96x F32->BF16, 1x BF16->F16) is documentation, not an engine input.

### 1.4 The drafter
No engine change. Upstream's own Install flow never serves the set's `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` directly: it fetches its own MTP tensors and builds `rt/` via `tools/mtp_rt.py --gguf ... --out rt` (setup.py:10915). The fork's `--mtp /home/svg/LLM/data/mtp-en/rt` layout (`dense.bin`, `dense.txt`, `draft_vocab.bin`, `experts.bin`) is the same contract stock's config used (`/home/svg/LLM/data/mtp/rt`). The shipped Q8_0 mtp GGUF is an ALTERNATIVE source: if the operator wants it instead of the r17 drafter, build an rt from it with the fork's `tools/mtp_rt.py` (it dequantizes GGUF tensors and re-packs; Q8_0 input round-trips through its quantizers) — but that is a new drafter, a quality decision, not a port item.

## 2. What landed (file inventory)

| File | Change |
|---|---|
| `src/kernels/cuda/iq_kernels.cu` | `vec_dot_q5_0_q8_1`: the -16/weight offset now multiplies the sum of the QUANTIZED activations (`dp4a 0x01010101 * d8`), result `d5 * d8 * (sumi - 16*sumu)` — verbatim hunk of upstream aa678f10d (comment extended with the port note). |
| `sycl/src/kernels/cuda/iq_kernels.dp.cpp` | Same fix, SYCL twin (keeps the SYCL build numerically identical to CUDA). |
| `src/kernels/native_expert_parity.cpp` | Q5_0-down GPU parity limit tightened 3e-2 -> 1.8e-2 (pins the fix; fails the old dot). |
| `src/kernels/native_grouped_parity.cpp` | Q5_0 in the grouped down sweep + the v31 pair matrix at 2560x640 (c1d5e4d5 hunk, applied verbatim). |
| `src/kernels/native_dense_parity.cpp` | NEW from 5d56ff8c: dense Q2_0/Q4_0/Q5_0/Q8_0/IQ4_NL/IQ4_XS/Q3_K/Q4_K/Q5_K/Q6_K — MMVQ 1..8 cols vs a double product of ggml's own dequant, prompt f32/f16/bf16 dequantizers; `--only=NAME`. |
| `src/kernels/native_dense_types_parity.cpp` | NEW from c1d5e4d5: the same dense types at the model shapes with the exact multi-column layout; `--bench`. |
| `tests/cuda/prefill_mmq_v31_test.cpp` | NEW from 5d56ff8c: the v31 expert formats through the MMQ prompt path (Q5_0 skipped honestly unless `STRATA_ORCA_Q4KS_MMQ=ON`, `--require-q5_0` enforces), `--bench` Q5_0 MMQ vs dequant+cuBLAS. |
| `CMakeLists.txt` | Registers the three new targets/tests + the 24 `native_expert_parity_v31_<gu>_<dn>` ctests, in the fork's existing blocks. |
| `config/custom-ud-iq2_xs-8082.json` | NEW: ready-to-boot serving config for the on-disk pack (pack/native/vision/mtp paths filled; **`api_key` is `REPLACE_ME_BEFORE_BOOT`**). Modeled on `config/custom-iq3_s-8081.json` minus iq3_s-specific knobs (`--ple-gguf` omitted on purpose: exercises the auto-find; expert-profile args omitted: per-pack learned state is the operator's call). |
| `ud-port.patch` | The whole package as one `git diff` (worktree root). |

Not ported, by the minimal-set rule: upstream's Gyro trellis types (147-149), GP100 nibble slots (`nib_gu_proto.cuh`/STRATA_EXPERT_NIBBLE), RDNA4/window-16 grouped-kernel work, Metal Q6_K/Q5_0 (no Metal target here), DGX mode-8, the round-0214 setup.py catalog/Install flow (Python-side; see §5), `dflash2`.

## 3. Build proof (log tails)

Engine (fresh build dir `/tmp/ud-port/build-ud`, `-j 24`):
```
[165/165] Linking CXX executable strata
BUILD_EXIT=0
```
`strata --version` -> `strata 0.1.46`. My changed TU rebuilt inside it: `build-ud/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.cu.o`.

Parity targets:
```
[4/10] Linking CXX executable native_dense_types_parity
[6/10] Linking CXX executable native_grouped_parity
[8/10] Linking CXX executable prefill_mmq_v31_test
[10/10] Linking CXX executable native_expert_parity
TESTS_BUILD_EXIT=0
```
(also `native_dense_parity` linked, step [3/10]). The 301-line MMQ test compiling against the fork's `strata_mmq`/`moe_mmq.hpp` is itself an API-parity proof: `mmq::Context/Product/q8_bytes/quantize/supported/built/matrix_bytes` are signature-identical between the fork and tip.

## 4. Boot-check sequence (operator, next free GPU window)

1. **Install the patch**: `git -C /home/svg/LLM/strata-custom-2 apply /tmp/ud-port/ud-port.patch` (or build from the scratch: `/tmp/ud-port/build-ud` is already built with the live-tree flags; `engine/strata` swap = copy `/tmp/ud-port/build-ud/strata` after applying).
2. **Kernel pins, no model** (~10 min GPU): `ctest` in a build dir of the applied tree —
   - `native_dense_parity` (expect every line `ok`, `decode rel` a few 1e-3, `col0 same as 1 col: yes` for all ten types) — pins the dense side incl. Q6_K.
   - `native_dense_types_parity` — pins the exact multi-column layout at 2560/640.
   - `ctest -R native_expert_parity_v31_` (24 pairs) — pins every (gu, down) the set uses; **`native_expert_parity_v31_iq2_s_q5_0` is the one that fails if the Q5_0 dot fix regressed** (old dot sits at 2.1e-2..3.1e-2 > the new 1.8e-2 limit).
   - `native_grouped_parity` (now sweeps Q5_0 down + the v31 matrix at model shapes).
   - `prefill_mmq_v31_test` — with the live flags it passes with `(Q5_0 not built)`; rebuild later with `-DSTRATA_ORCA_Q4KS_MMQ=ON` + `--require-q5_0` to pin the prompt-path Q5_0 MMQ.
3. **Boot the set**: edit `api_key` in `config/custom-ud-iq2_xs-8082.json`, then from the fork root: `.venv/bin/python serve/server.py --engine strata --config config/custom-ud-iq2_xs-8082.json --port 8082`. Log lines that confirm each stage:
   - shard derivation: no `missing model shard` error; the PLE auto-find means **no `--native requires --ple-gguf` refusal** (if you see it, shard 3 is misnamed — the finder works by tensor name, not position).
   - pack/expert handshake: `check_experts_gguf` passes silently (a type mismatch prints `the pack's native_experts.txt does not match the model: ... is <type>, the pack says type <N>`).
   - the per-layer gate (`generate.cpp:3555`) staying silent = all 48 layers have GPU kernels; any `layer %lld's experts are ... no GPU kernels for` is a stop, not a boot.
   - experts mapped from shard 2 (~42.3 GB blob total): the `experts` mmap line (~44 GB, file `...-00002-of-00003.gguf`); PLE table from shard 3 with `--ple-io ram`.
   - first tokens: compare a fixed prompt against the stock engine's UD output on the same seed; layer 4's Q5_0 down is the ONLY layer whose numerics changed vs an unpatched fork build (expected small differences, by design of aa678f10d).
4. **Quality gate**: the banked MMLU floor for this model is 93.3 (fork IQ3_S bank 94.29; stock UD measured 97.14 at half the bytes).

## 5. Known risks, each with the check that catches it

1. **IQ2_XXS/IQ2_S gate/up never ran live on this rig** (production mileage is IQ3_S-class only). The kernels are template-instantiated (compile proof) and parity-pinned (the v31 ctests), but no device run has happened. Catch: step 2's `native_expert_parity_v31_iq2_xxs_*` / `_iq2_s_*` legs before any model boot.
2. **Q5_0 decode numerics change on layer 4** vs any unpatched fork binary. Expected (the fix), but it breaks byte-identity A/Bs against old binaries for prompts routed through layer 4's experts. Catch: golden-hash A/Bs must use a patched control.
3. **Dense BF16 kinds from compat-bf16 conversion**: the UD pack widens 291 tensors (Q8_0->BF16 with RNE, max_abs_err up to ~7.8e-3 recorded in conversions.json). The index kind-4 path is code-identical to iq3_s's (weights.cpp diff = 0), but the CONVERSION ERRORS are new data. Catch: the boot's hc selftest + a `--load-check` (native_expert_parity mode) against the real pack/GGUF pair before serving.
4. **`ggml-cpu` traits for type 42/16/22 on the prompt path**: the CPU pool delegates to ggml-cpu vec_dot; the fork's ggml is the same vintage for these types (production Q2_0 shadow store proves 42; 16/22 have upstream-tested traits). Catch: `native_expert_parity --synthetic` legs run the CPU path alongside the GPU path (`cpu rel` column) — any trait gap prints `ggml-cpu has no dot product for type N` from `native_fmt` at boot.
5. **Setup.py catalog is still the old one** (fork setup.py:93 pins the 2026-09-30 unsloth revision; no UD3.1 rows, `UNSLOTH_V31_ENGINE (0,1,57)` gate absent). The ENGINE serves the set; `setup.py` cannot install/configure it from the catalog. Mitigated by the delivered config; a catalog port (UNSLOTH_V31_SHARDS pins, Install flow, per-model revision d9620119) is a separate, self-contained Python task if the operator wants setup-driven installs.
6. **The `--mtp-en/rt` drafter with the UD set**: same model family, but the r17 drafter was built from the GSQ-RCO set. Acceptance should carry over (same MTP weights); if acceptance drops, build a fresh rt from the set's own `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` via `tools/mtp_rt.py` and re-run the draft-acceptance leg. Catch: the boot's draft-acceptance read (`P<` lines) vs the iq3_s bank.

## 6. Boundary

Compile proof complete (engine + all five parity targets, exit 0, operator's exact build flags). Device proof pending by constraint (GPU serialized for benches): no parity test, no boot, no synthetic run was executed — nothing here reports a device pass. The scratch (`/tmp/ud-port`, worktree at 9728436a) holds the applied changes, the patch file, and the build dir; the live tree carries only this digest.
