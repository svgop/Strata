# Shadow expert store (STRATA_SHADOW_EXPERTS) — build notes

F-359. Falsifier (l1-falsifer premise, PASSED 2026-10): the CPU expert pool is byte-bound at ~150 us/entry
on the IQ3_S native blobs (2,046,400 B mean); the same model's Q2_0 blob store streams at ~75 us/entry
(1,382,400 B/blob) — 32-40% less per-entry byte traffic on the ~1.2 ms/group CPU-rows wait (22.6% of wall).
Mechanism: a SHADOW store that serves MISS routing entries only.

## The knob

- `STRATA_SHADOW_EXPERTS=<dir>` — e.g. `/home/svg/LLM/data/packs/q2_0`.
- Unset / empty: off. No mapping, no banner, no armed() branch taken. Byte-identical behavior.
- Set: at boot (`generate.cpp`, right after `expert_layout_load` + the GPU-kernel support checks) the engine
  maps `<dir>/experts.bin` and validates `<dir>/experts.json` against the canonical layout, then prints one
  banner. Refusals (missing file, wrong blob_bytes, wrong layer count/order/offsets, wrong file size,
  Hadamard-folded pack) print one line and leave the engine unshadowed — the native store serves every miss.

## Layout: reused, not rebuilt

The q2_0 store's layout IS the engine's compile-time canonical `ExpertLayout` (`native = false`):
`layer_offset(l) = l * n_expert * BLOB`, `BLOB = 1,382,400` (`include/strata/kernels/cpu/expert.hpp`;
blob-internal offsets `O_GU_CODES=0`, `O_D_CODES=819200`, `O_GU_SCALES=1228800`, `O_D_SCALES=1331200` —
exactly `/home/svg/LLM/data/packs/q2_0/experts.json`). The engine never read experts.json before: it is the
packer's description (`tools/pack_layer.py`) of the canonical form the kernels decode. The shadow builder
(`src/core/shadow_experts.cpp`) parses that file and REFUSES anything that is not byte-for-byte the canonical
layout, so the compute side needs no new constructor and no kernel-signature change.

## Compute path (misses only)

`expert_pool_dispatch_multi` (`src/core/expert_source.cpp`), gated on `const bool shadow =
strata::core::shadow::armed()`:

1. Activation quantize: armed → `act_quant_q8_1` per token (the canonical ActQ contract the Q2_0 rows
   kernels take), instead of `native_quant_act` / `act_quant_any`.
2. Prefetch: armed → skipped. The misses read the shadow's own mmap; prefetching the native 2 MiB blobs
   would only pull pages the window never touches. cold_mode 1-4 (overlap / zero-copy native roles) are
   bypassed with it (`zc` is `!shadow && ...`).
3. Job construction (`kind[i] == -1` only): `nj.blob = shadow::blob(layer, expert)`
   = `base + layer_off[layer] + expert*1382400`; `fmt = nullptr`, `nact = nullptr`.
4. Pool run: armed → `run_split_multi` (canonical rows phases 3/4; reads only `blob`/`act`/`out`) instead of
   `run_split_multi_native` (modes 5/6). Both call sites (cold-overlap batches and the final run).

Untouched, by construction: the plan pass (`kind` 0/1/2 assignment, `nmiss`, tail-skip), VRAM hits, the PCIe
share (still fed `d.src->blob` for its picks, still staged with the native `bb` stride), peer/remote helper
GPUs (native formats in their VRAM), `expert_hit_run` cache admissions, the prompt path
(`expert_pool_dispatch`, single-token, native), and every residual/combine path.

## Gate-off byte-identity argument

`arm()` returns false before touching the filesystem when the env is unset. `armed()` is then false, and
every gated expression reduces to the original one: `if (shadow)` false → the old else-if chain runs;
`!shadow && A` ≡ A; `native && !shadow` ≡ `native`; the blob ternary takes `d.src->blob`. The pool receives
the same jobs (same `blob` pointers from the same source) through the same `run_split_multi_native` call.
The only residue is one bool load per window and a dead counter. No output byte can differ.

Armed, the misses differ ONLY by the shadow's quantization (Q2_0 at 2.06 bpw-ish vs the native IQ3_S) —
that is the experiment. Hits remain bit-identical: same blobs, same GPU kernels, same order.

## Counter

`ExpertDispatch::shadow_misses` (distinct miss experts served from the shadow; equals `multi_misses` while
armed, 0 when off). Printed at end of run under the `pool multi` line:

```
  shadow experts           N of M CPU misses computed from the shadow Q2_0 store (<dir>)
```

## Risks

- **Model mismatch**: arm() validates geometry (layers x 512 x 1,382,400 B, offsets), not model identity.
  Pointing STRATA_SHADOW_EXPERTS at another model's q2_0 pack produces valid-looking wrong misses. The
  campaign must pair the IQ3_S pack with ITS OWN q2_0 repack (this one: 48 x 512, total_bytes
  33,973,862,400).
- **Quantization drift on misses** is the design; expect MMLU to move within Q2_0's error, not IQ3_S's.
  A/B the same seed; watch first-token divergence only on windows that had misses (most of them).
- **Page-cache pressure**: the shadow adds a 31.6 GiB mapping beside the native pack's 34 GiB experts.bin
  (the native blobs are still read for the PCIe share and cache admissions). 64 GB DDR5 will thrash if the
  OS keeps both fully warm; the shadow's skip-prefetch choice keeps it demand-faulted. If RSS pressure
  shows, `STRATA_ARENA_MMAP`-style aliasing is the later move, not this knob.
- **First-touch faults in the pool drain**: cold misses fault inside worker threads (the Phase-2 model).
  Expect the first pass over each layer's shadow range to pay disk; steady-state is DRAM.

## Smoke plan (no live server touched; build/ only, never deployed)

1. Gate-off boot: `build/strata generate ...` with the usual IQ3_S pack and NO env. Expect no banner, no
   `shadow experts` line; behavior identical to the previous binary (same seed → same tokens).
2. Forced-miss boot: raise `--vram-reserve-mib` so the VRAM expert tier holds ~0 experts (every routed
   expert becomes a CPU miss), plus `STRATA_SHADOW_EXPERTS=/home/svg/LLM/data/packs/q2_0`. Expect:
   - the boot banner (`48 layers x 512 experts x 1,382,400 B = 31.64 GiB`),
   - `pool multi` GB/s roughly doubling over the rows phases (multi_bytes now counts 1,382,400/blob),
   - `shadow experts` = `multi_misses`,
   - `d.ms_run` / `ms_jobs` down ~2x on miss-heavy layers; wall down toward the 22.6% CPU-rows share.
3. Refusal paths: point the env at an empty dir, then at the IQ3_S pack dir itself (no experts.json /
   wrong blob size) — one stderr line each, decode continues on the native store.
4. MMLU: mmlu-pro-gate.py A/B, gate-off arm vs armed arm, same seed; accept the Q2_0-sized delta the
   falsifier priced, investigate anything larger.

## Files

- NEW `include/strata/core/shadow_experts.hpp`, `src/core/shadow_experts.cpp` (arm/parse/mmap/blob).
- `src/core/expert_source.cpp` — 6 gated sites in `expert_pool_dispatch_multi` (flag, actq, prefetch, zc,
  blob+counter, two run calls); `include/strata/core/expert_source.hpp` — `shadow_misses`.
- `src/program/generate.cpp` — boot arm + summary line.
- `CMakeLists.txt` — `src/core/shadow_experts.cpp` added to `strata_engine`.
