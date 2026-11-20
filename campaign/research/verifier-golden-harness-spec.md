# The Verifier/MTP Golden Master harness — the unlock task for the specdec front (F-448 spec)

## Why this exists

The three biggest remaining god-modules are all engine-pinned — zero unit-test callers:

| Module | Lines | Worst function | Current net |
|---|---|---|---|
| `src/core/verify.cpp` | 4,727 | `record_window` **1,308** (23 captured lambdas; the `pre`/`post` pipeline) | engine only |
| `src/core/mtp.cpp` | 2,169 | `draft` 247, `record_rest` 252, `bind` 160 | engine only |
| `src/program/generate.cpp` | 18,744 | `main` ~16,280 (214 captured lambdas) | engine only |

`verify_parity`/`spec_verify_parity` test KERNELS (`strata_kernels`), not `Verifier`.
`spec_prob_test`/`coupled_draft_test` test extracted headers, no GPU.  Engine-level A/B
(`campaign/bit-check.py`) needs a second engine instance: impossible while the operator's engine
holds 30.7/32.6 GB VRAM, and deploy-verified-battery is never one command (law 9).  Every
extraction from these modules is therefore compile-only until this harness exists.

## The hard constraint

The kernels are compiled for `kCompiledGeometry` (F-421 pinned three prefill kernels to it): a
synthetic model must run at REAL Qwen4Exp dimensions (n_embd 2560, 48 layers, 512 experts,
n_vocab 248320).  A tiny geometry trips the compiled-shape asserts.

## Design sketch

1. `tests/gpu/verifier_golden_test.cpp` (CUDA-gated block, links `strata_engine`).
2. WeightTable at real dims, values = deterministic LCG.  VRAM budget is the long pole:
   `token_embd` alone is 248320x2560 (2.5 GiB fp32 / 1.27 GiB bf16).  Two options:
   a. **Engine-down window** (the operator's): full-size buffers, run in the battery slot.
   b. **Capped-vocab trick**: allocate R rows only, drive token ids < R, IF no kernel bounds-checks
      against `n_vocab` (must be verified per kernel; the lm_head reads by token id, but the sampler
      and any vocab-wide memset would need the full width — audit before relying on it).
   Per-layer weights at real dims total ~1-2 GiB more; fp16/bf16 refs where the kernel takes them.
3. Expert blobs: reuse the dispatch golden test's synthetic-pack machinery (`expert_layout_load` +
   LCG blobs) — `FileExpertSource` is already proven under it.
4. Drive: `Verifier::init` (small T) → `record_window(T=1, cs)` → fingerprint `R_` + `head_logits_`
   (FNV-1a 64, machine-pinned), plus `commit()`'s state advance.  The MTP drafter rides the same
   scaffolding (`record_front`/`record_rest`/`draft` on the same synthetic model).
5. Determinism pins: the env gates the dispatch golden pins (`STRATA_COLD_OVERLAP=0`,
   `STRATA_ROUTE_TAIL_SKIP=0`) + every knob `record_window` reads (grep `std::getenv` in
   verify.cpp: `STRATA_DF_PDL`, `STRATA_DF_BRANCH`, `STRATA_BATCH_G2`, `STRATA_LFUSE`,
   `STRATA_BATCH_SELF_COMMIT`, ...) — pin to their DEFAULT arm so the fingerprint freezes the
   default path; other arms become second ctest registrations (the F-440 onepass pattern).
6. A/B protocol: build the harness at the pre-extraction commit and HEAD; identical fingerprints
   certify each cut.  This is F-438's method at model scale.

## Cost and acceptance

Cost: one focused session (the weight-name surface is enumerable: ~6 literal `wt.find` in
verify.cpp + the per-layer patterns; `Verifier::init`'s allocation demands are the unknown).
Acceptance: ctest green with frozen fingerprints on this box; `campaign/return_audit.py` extended
to run on every extraction PR; then the `record_window` decomposition (extract the 23-lambda
`pre`/`post` clusters one at a time) proceeds with an execution oracle.

## Do NOT

Do not extract from `record_window` before this harness exists - twice this campaign has paid the
UB price of refactoring blind (F-441, F-447); a hot-lane UB in the verify path would be paid in
acceptance regression, not test minutes.
