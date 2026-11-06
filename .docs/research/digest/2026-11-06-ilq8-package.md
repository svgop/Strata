---
type: digest
id: ilq8-package-2026-11-06
created: 2026-11-06
status: patch-delivered
topic: IL-Q8 — the dense decode MMVQ's interleaved-rows (IL) fast path extended to Q8_0 weights (ggml type 8): IlQ8_0 traits, STRATA_MMVQ_IL_Q8 kill-switch (default on), conservative Q6_K-mirror sm_120 table rows with the operator's marked emit slot, parity cases, compile-proven; emission + bench legs priced, not run (GPU serialized)
scope: patch built in the scratch worktree /home/svg/LLM/ilq8-wt (fork HEAD 1465fa6b, never the live tree); patch /home/svg/LLM/ilq8-wt/ilq8-mmvq-q8_0.patch (4 files, +89/−7), dry-applies clean on the live checkout. Machinery read before building: src/kernels/cuda/native_mmvq.cu (IL traits, launch_il_rows, il_rows tables, native_mmvq_il_supported/dispatch), docs/MMVQ_IL_TABLE.md, src/kernels/mmvq_il_parity.cpp, src/core/verify.cpp call sites. Sibling: 2026-11-06-ud-dense-conv-package.md (the load-side Q8_0 package this kernel-side arm pairs with).
---

# IL-Q8 package: the IL fast path takes Q8_0 (2026-11-06)

Verdict: **built, linked, delivered.** Patch `/home/svg/LLM/ilq8-wt/ilq8-mmvq-q8_0.patch` on fork HEAD `1465fa6b` — the live tree untouched. `strata` (91.8 MB) and `mmvq_il_parity` (15.4 MB) link clean in the worktree's fresh build (`build-ilq8`, Release, CUDA 120, nvcc 13.4, the fork's `.venv` cmake/ninja, exit 0), and all 24 `native_mmvq_il_kernel<IlQ8_0, NC 1-8, R 1/2/4>` device instantiations are present in the compiled sm_120 cubin (`cuobjdump`, checked). The arm is GSQ-inert by construction: GSQ's dense projections are the K-quants that already take IL; only the UD packs' ~2 GB of Q8_0 dense projections (attn_qkv, ssm_out, the QSA q/k/v/o) change behavior, and only where a non-zero table cell says so.

## 1. Traits design (follows the established contract exactly)

`IlQ8_0` sits beside `IlQ4K`/`IlQ5K`/`IlQ6K`/`IlQ3K` in `native_mmvq.cu` and rides the generic `native_mmvq_il_kernel<I, NC, R>` unmodified:

- `using F = SmallTraits<Q80Block, 8>` — the plain path's own traits: `DIV=32, T=4, KBY=1, BPI=32`, `kqs(tid)=2*(tid%4)`, `W={int v[2]; float d0}`. The dequant is trivial against the k-quants: `load` (unchanged, inherited) reads two 32-bit weight words and the fp16 block scale once per (row, block); no scales-unpacking, no high-bit surgery, no table lookup.
- `PM=false` — block-major interleave, like the K-quant traits. Rationale: Q8_0's four lanes per block cover words 0..7 of one 32-value block (`T=4`, `KBY=1`, so `kby` is the activation block itself); a block's whole activation fragment is contiguous only in the block-major copy. IQ4_XS takes position-major because its lanes read the same int of consecutive blocks — the opposite pattern.
- `apply` reads the interleaved copy's words `iqs, iqs+1` of block `kby` per column (`Q81Il::u`) and the block scales (`Q81Il::scales`), then runs the plain `SmallTraits<Q80Block,8>::apply` expression per column in the plain order: `sumi = DP4A(v0,u0, DP4A(v1,u1,0)); out = d0 * d8 * float(sumi)`. Same operand order, same two-multiply chain — bitwise parity by construction, same law the other five types hold.
- **Bitwise argument** (the property the parity binary pins): the IL kernel's chunk decomposition maps chunk `m` of lane `l` to what plain thread `32*(m%4)+l` does at iteration `m/4` (`kbx = 8m + l/4` vs plain `8*(m%4) + l/4 + 32*(m/4)` — equal, since `BPI=32`, `SUB=8`), `acc[v]` accumulates exactly the plain warp-`v` partial in the same add order, and the reduction is the same warp-order sum then XOR tree. Every output bit equals `native_mmvq`'s multi-column exact kernel, so the table is only speed.
- **Dispatch + whitelist**: `case 8: launch_il_rows<IlQ8_0>` in `native_mmvq_il`; `ggml_type == 8` added to the `native_mmvq_il_supported` whitelist. Engine-side routing needs zero edits — `verify.cpp:1487/1528/1764/2450` already consult `native_mmvq_il_supported` per call.

## 2. Kill-switch: STRATA_MMVQ_IL_Q8 (default ON; =0 restores the plain path)

Copied from the `STRATA_MMVQ_IL_Q3K` pattern (`il_env_on`, cached at first use). Placed in three positions so `=0` is the **master** switch, authoritative over everything:

1. `native_mmvq_il_rows_for`, wide branch (5-8 columns): returns 0 for type 8.
2. `native_mmvq_il_rows_for`, 2-4 columns: placed **ahead of** the `STRATA_MMVQ_IL_ROWS` override loop (one documented asymmetry: the Q3_K gate sits behind the override loop; Q8_0's does not — the operator's =0 always wins, even with a rows override armed).
3. `native_mmvq_il_supported`: `ggml_type == 8 && !il_q8_on()` → false, even under `native_mmvq_il_tune` (the parity sweep still exercises Q8_0 because the bench runs with the switch unset).

Bench legs are armed (unset or `=1`) vs `STRATA_MMVQ_IL_Q8=0`; if the plain path wins on the untuned table, `=0` restores with no rebuild.

## 3. The table: conservative defaults + the operator's emit slot

Rows added **only** to the sm_120 tables — the card this arm is for:

- `kIlRows_120` (serves cc `1201`, a 12.0 card with ≥80 SMs; ncols 2-4): `{8, {{0,0,2,1,1},{0,0,4,2,4},{0,2,2,2,2}}}` — a cell-for-cell mirror of that table's **Q6_K** row.
- `kIlRows58_120` (ncols 5-8, rows 4 run as 2): `{8, {{0,2,4,4,4},{0,4,2,4,2},{0,2,4,2,2},{0,4,4,4,2}}}` — the Q6_K mirror of the wide table.

Why Q6_K and not Q5_K: Q6_K is the closest simple k-quant (a float scale applied after the integer dot, like Q8_0's one fp16 per block) and its row is the more conservative of the two in this table (e.g. ncols 2 class 1: Q6_K 0 vs Q5_K 1). No other table got a type-8 row — `kIlRowsShared` and the measured per-arch tables (`60/61/70/75/86`) answer **0** for type 8, so every other card keeps the plain kernels exactly as before (the pre-patch behavior). `check_tables()` in the parity binary now asserts this shape: legal rows (0/1/2/4) on cc 120/1201, zero on every other cc, and three pinned mirror cells (`1201:8:2:6144 → 2`, `1201:8:3:6144 → 4`, `1201:8:5:2560 → 2`).

The n_out classes these cells speak (our Q8_0 shapes): 512 = k/v (cls 0), 2560 = ssm_out/o (cls 1), 6144 = gate (cls 2), 10240 = attn_qkv (cls 3), 12288 = q (cls 4). Under the mirror, the armed defaults take IL for qkv/gate/q at 2-4 columns and most 5-8 cells; k/v-sized rows keep plain (conservative 0s).

**The marked slot**: the mirror lines in `kIlRows_120` and `kIlRows58_120` carry `// Q8_0 (conservative Q6_K mirror; replace with the measured row)` plus a comment block naming the emit command. The operator's measured table replaces them in place.

### Emission instructions (run on the card, idle GPU)

The worktree is already configured and built — bench straight from it, live tree untouched:

```
cd /home/svg/LLM/ilq8-wt
build-ilq8/mmvq_il_parity --bench --emit-table 2>&1 | tee ilq8-emit.txt
```

A few minutes. It must end `mmvq_il_parity: OK (every case, T 1-8, rows 1/2/4 and the table, bitwise)` — the harness now carries 7 Q8_0 cases (2560x12288/10240/6144/640/512, 6144x2560, 2560x2051 odd) run bitwise against `native_mmvq` at T 1-8, rows 0/1/2/4, and `--emit-table` prints Q8_0 lines in all three blocks (2-4 columns, the 5-8 block, the one-column block). Then:

- Paste the emitted `    {8, {...}},   // Q8_0` line from the first block over the marked mirror line in `kIlRows_120`; the emitted line from the `// 5-8 columns` block over the marked line in `kIlRows58_120` (per `docs/MMVQ_IL_TABLE.md`).
- No-rebuild cell sweeps stay available: `STRATA_MMVQ_IL_ROWS="8:2:r0,r1,r2,r3,r4;8:3:...;8:4:..."` (rows 0/1/2/4 per class), then A/B the decode, 10 interleaved pairs, medians.

## 4. Bench legs (operator runs; GPU was serialized, nothing executed here)

- **Leg 0 — micro, mandatory first**: the `--bench --emit-table` above. Every Q8_0 `BENCH` line prints `multi` vs `il rows1/2/4` per shape and token count; the emit rule keeps a cell only when the best rows choice takes ≥3% off plain. This alone decides every table cell; the patch is safe to arm before it because the mirrored defaults can only mis-guess speed, never bits.
- **Legs 1-2 — engine pair + conc, per pack**: the standing recipes `launch-custom-ud-iq2xs-8081.sh` and `launch-custom-ud-iq4xs-8081.sh`, A/B on `STRATA_MMVQ_IL_Q8` (unset/1 vs 0), interleaved pairs, medians, the house 3v3 ABAB fence. The lanes the arm touches are the multi-row decode windows' dense projections through `verify.cpp`'s IL gates (`:1487` dense, `:1528` gate/wg, `:1764` q_proj, `:2450` head — the head is Q6_K/Q5_K, unaffected). Watch pair + conc M-tokens and per-window p50 (the SERVE_TELEMETRY lines); acceptance law unchanged: no lane negative beyond noise, M3b in-band, and the bitwise guarantee means MMLU/mimic gates should read identical outputs (the path is output-identical, so a quality regression is a bug, not a tradeoff).
- What to expect: Q8_0 is the most bandwidth-dense type in the window (34 B per 32 values), so the plain kernel is already weight-stream-bound; IL's win here comes from per-block decode amortization across R rows and warp-shape effects, not activation reuse. Leg 0 decides whether the mirror or the measured table (or `=0`) stands.

## 5. Risks and named boundaries

1. **34-byte block stride vs interleave granularity**: the interleaved copy is of the *activations* (q8_1 blocks), never the weights — Q8_0's 34-byte GGUF stride keeps every weight int 2-byte-aligned (`load_int_b2` = two 16-bit loads), exactly as the plain kernel reads them. IL adds no new misalignment; but it also means IL cannot reuse the aligned-plane trick of `STRATA_Q8_PACKED` (a load-time repack). The two forms are disjoint arms: packed is opt-in (`=1`) and its registry preempts IL inside `native_q8_0_mmvq`; default boots are unaffected.
2. **Untuned defaults**: the Q6_K mirror is a guess until Leg 0 lands. Conservative cells (several 0s, e.g. all ncols-2 class-0/1) mean small Q8_0 calls keep the plain path out of the box; a wrong guess costs time only (bitwise law), and `STRATA_MMVQ_IL_Q8=0` restores the exact pre-patch behavior per boot with no rebuild.
3. **Pair-kernel boundary**: the shared expert's gate+up Q8_0 pair takes `native_mmvq_pair` (`shared_expert.cu:230`) before IL is consulted; that path is untouched by this arm. If Leg 0 shows Q8_0 IL winning broadly, the pair path is the next candidate, priced separately.
4. **Shape boundary**: `native_mmvq_il` validates `n_in` as a multiple of 256 (`QK`), not Q8_0's 32 — an `n_in` that is a 32-multiple but not a 256-multiple silently keeps the plain path. All real dense projections (2560/4096/6144/10240/12288) are 256-multiples; noted so nobody reads the fallback as a defect.
5. **ncols 1**: no one-column Q8_0 table row was added (the dedicated single-column kernels won by margins the fork deemed noise for the other types), so `ncols==1` Q8_0 always takes the plain single-column kernels. The emit block still prints a one-column Q8_0 line if the operator ever wants to price it.
6. **Zero new VRAM**: type 8 consumes the same interleaved activation buffer (`xil_`, sized once per verifier) the K-quant IL already uses; no new device allocations anywhere in the patch.

## 6. Compile proof

Configured fresh in `/home/svg/LLM/ilq8-wt/build-ilq8` (Ninja via the fork's `.venv`, `CMAKE_MAKE_PROGRAM=.venv/bin/ninja`, Release, `CMAKE_CUDA_ARCHITECTURES=120`, `CMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc`, `CMAKE_CXX_COMPILER=/usr/sbin/c++`, `STRATA_ENABLE_CUDA=ON`, `STRATA_BUILD_TESTS=ON`):

```
[163/167] Linking CXX static library libstrata_engine.a
[164/167] Linking CXX static library libstrata_prefill.a
[165/167] Linking CXX executable mmvq_il_parity
[167/167] Linking CXX executable strata            (exit 0)
```

Zero errors; no new warnings from `native_mmvq.cu` or `mmvq_il_parity.cpp` (the log's diagnostics are the pre-existing ones in `gdn.cu`, `layer.cpp`, `qsa_select.cu` and friends). Device-level proof: `cuobjdump -elf .../native_mmvq.cu.o | grep IlQ8_0` → 360 hits, including all `native_mmvq_il_kernel<IlQ8_0, Li1..8ELi1/2/4>` instances in the sm_120 cubin. The patch dry-applies clean on the live checkout (`git apply --check`, exit 0).

**Apply sequence for the operator**: bench Leg 0 from the worktree → paste the emitted Q8_0 lines into the marked slots → `git apply /home/svg/LLM/ilq8-wt/ilq8-mmvq-q8_0.patch` in the live repo (or keep the worktree's `build-ilq8/strata` as the boot binary until then) → run Legs 1-2 armed vs `STRATA_MMVQ_IL_Q8=0`.
