---
type: research
date: 2026-11-14
topic: MiniMax M3 engine subsystem import inventory (maya src/mm3 -> strata-custom-2 as third engine family)
maya_tree: /home/svg/LLM/data/references/project-maya (read-only reference, v1.1.17 lineage)
our_tree: /home/svg/LLM/strata-custom-2
mm3_size: 19,281 lines across 32 files (src/mm3), plus 4 shared-substrate files it pulls from src/ds4
precedent: src/dsv41 was imported the same way from maya src/ds4 (vendored, MIT NOTICE headers, "Ported into Strata" provenance line, no symbol shared with strata_kernels)
---

# mm3 import map: MiniMax-M3.1-Flash as a third engine family

Every claim below is anchored `file:line` against the two trees as read on 2026-11-14. MAYA = `/home/svg/LLM/data/references/project-maya`, OURS = `/home/svg/LLM/strata-custom-2`.

## 0. Verdict summary

| Verdict | Files | Lines |
|---|---|---|
| IMPORT-AS-IS (works against our substrate after include/env renames) | 17 | ~2,300 |
| ADAPT (small, named edits) | 7 | ~8,900 |
| VENDOR-AS-NEW-FAMILY (bring maya ds4 files mm3 needs; ours diverged) | 4 (from src/ds4) + 2 headers | ~1,150 |
| SKIP (no equivalent wanted / maya-only infrastructure) | 4 (+ all of src/metal) | ~480 (+ metal's 6,700) |

The dominant fact: **mm3 is deliberately self-contained** — its own EXL3 fork (`strata::mm3::exl3`), its own GGUF layer (`mm3::gg`), its own expert streaming (resident/staging/pool/promote inside `mm3_model.cu`), its own attention kernels (`mm3::k`). Its only external surface is 6 `../ds4/*` files, 3 `strata/kernels/cpu/*` headers, 2 `strata/spec/*` headers, `strata::GgufFile` (artifact), and `../core/ablate_capture.hpp`. Of those, four have real drift against our tree; the rest import clean.

---

## 1. mm3 file-by-file table

Lines from `wc -l`. "Shared APIs consumed" lists maya-shared headers by the symbols actually referenced (grep-verified, not guessed).

| File | Lines | Purpose, key kernels/classes | Shared APIs consumed | Guards | Verdict |
|---|---|---|---|---|---|
| `mm3_kernels.cuh` | 253 | `namespace mm3::k`: MSA + router + KV-cache-format + TP-sync + DSpark-draft + Markov entry points. `KvQ`/`KvHost` (KV formats bf16/i8/fp8/fp4 + host ring), `kCandCap/kCandExtra`, `CandRow`, `QRow`, `SampleArgs` | none (self-contained decl) | none | **IMPORT-AS-IS** |
| `mm3_kernels.cu` | 2618 | all of the above's device kernels: `rms_kernel` :136, `qk_kernel` :347, `index_kernel` :381, `block_scores_kernel` :401, `block_scores_tok_kernel` :452, `block_scores_wide_kernel` :510, `select_kernel` :594, `sparse_attn_kernel` :769, `sparse_attn_w_kernel` :909, `attn_combine_kernel` :1059, `route_kernel` :1106, `plan_dev_kernel` :1158, `swigluoai*` :1215-1224, `mx_rows_kernel` :1349, `window_attn_kernel` :1540 + `_split` :1628, `markov_*` :1727-2075, `topk_rows_kernel` :2052, `wait_flag/pull/publish` :2336-2454. Includes `../ds4/ds4_numerics.cuh` :9 for `bf2f/f2bf/e4m3_dec/e2m1_val/had128w` device helpers | `ds4::bf16_rn`, `ds4::e4m3_value` family via `ds4_numerics.cuh` (see §2.1) | `STRATA_USE_HIP` x10, `__gfx906__` x8 (wave64 reductions :486-497, :597-660), `__CUDA_ARCH__` x1, `_WIN32` x1 | **ADAPT** — only because ds4_numerics.cuh is vendored per §2.1; kernel code itself unchanged |
| `mm3_exl3.hpp` | 241 | `namespace strata::mm3::exl3`: EXL3 Mat/MvJob/MoeArgs/WinArgs/SetArgs/MixedLayouts, `kSwigluAlpha=1.702` :41, `kTypeEXL3=200` :38, `kFmtGG` handed to `mm3::gg`. May's own fork of the shared `include/strata/kernels/exl3.hpp` (header says so at :4) | none | none | **IMPORT-AS-IS** |
| `mm3_exl3.cu` | 3016 | the fork's kernels: `mv_kernel` :594, `moe_gate_up/down_kernel` :703/:727, `win_gu/down/combine` :791/:821/:842, mixed-class kernels :908-999, `win_wait_kernel` :1023, recon/rot kernels :1065-1240, `gemm_kernel` :1353, wmma tiles :1779-1855, cuBLAS FP16 GEMM path before sm_80 (`blas_enable`) | `strata/kernels/dp4a.hpp` :30 (present in OURS, identical file — verified `diff -q` clean), `mm3_gguf.cuh` :22 (fmt-4 dispatch) | `STRATA_USE_HIP` x26, `__CUDA_ARCH__>=800` x3 (:303, :391, :398), Volta wmma guard :312, `mma.h` :28 | **IMPORT-AS-IS** (vendor as new static lib; §5) |
| `mm3_gguf.cuh` | 76 | `namespace mm3::gg`: GGUF stand-in behind the x3 interface, `kFmtGG=4` :34, `type_of/row_bytes/block_values`, `mv/gemm_rows/moe_*/win_experts/set_*` decls, `set_gemm` backend :62 | includes `mm3_exl3.hpp` :27 | none | **IMPORT-AS-IS** |
| `mm3_gguf.cu` | 594 | GGUF device path: `namespace iq = strata::ds4::gguf::iq` :23 (activations rounded to Q8_K by default, `MAYA_MM3_IQ_Q8K`), prompt GEMMs through cuBLAS or hipBLASLt `strata::ds4::lt::{available,set_store,gemm}` :330-335, :458 | `ds4_iq.cuh` (maya version — `Args.act/tu/pexp` fields, see §2.2), `ds4_lt.hpp` :8 (MISSING in ours), `ds4_numerics.cuh` :9, `mm3_env` :5, `mm3_gv.cuh` :6 | `STRATA_USE_HIP` x6, gfx x2 | **ADAPT** — vendor maya's `ds4_iq.cu` next to it (namespace-identical include path swap); AMD-only `ds4_lt` per §2.3 |
| `mm3_gv.cuh` / `mm3_gv.cu` | 49 / 479 | dense GEMVs over Q4_K/Q5_K/Q6_K/Q8_0 (`kQ4_K=12..kQ8_0=8` :23), row-blocked warp kernels bit-identical to ds4_iq's `mv`; `covered()` :431 refuses jobs then falls back to ds4_iq `mv`; **CUDA only** (HIP keeps ds4_iq's mv, :46-47 of the .cuh) | falls through to `ds4_iq` kernels on HIP and uncovered jobs | `STRATA_USE_HIP` x3, gfx x1 | **IMPORT-AS-IS** |
| `mm3_gguf_host.hpp` | 150 | host dequant Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16 rows (ggml-cpu's loops) for token embedding + indexer projections | none | none | **IMPORT-AS-IS** |
| `mm3_pack.hpp` | 119 | `mm3::Pack`: mm3.txt/tensors.txt manifest, mmap of shards via `os::map_file` :81 | `mm3_os.hpp` :18 | none | **IMPORT-AS-IS** |
| `mm3_os.hpp` | 136 | `mm3::os` wrappers (`map_file` :72, `drop_mapped` :89, `ram_avail` :38 with cgroup-v2 walk, `ram_total`, `read_stdin`, `pin_self`) over `strata::ds4::os` (`namespace sys =` :17) | `../ds4/ds4_os.hpp` :12 — all `sys::` symbols it calls exist in OURS (`open_ro` :66, `map_ro` :275, `file_size` :244, `close_fd` :236, `kBadFd`, `ram_avail`, `pin_self`, `replace_file` :229 — verified) | `_WIN32` x6 | **IMPORT-AS-IS** |
| `mm3_env.hpp` | 16 | `MAYA_<name>` then bare `<name>` env lookup | none | none | **ADAPT** — one line: prefix `MAYA_` -> `STRATA_` (keep `MM3_*` aliases; our tree's convention is `STRATA_*`, cf. `STRATA_DS4_NO_LIGHT` in ours' ds4_iq.cuh:96 vs maya's `MAYA_DS4_NO_LIGHT`) |
| `mm3_geo.hpp` | 73 | `mm3::Geo::read` — geometry from mm3.txt :30-51 + coverage checks (hd==128, n_head==16*n_kv, ix_heads==n_kv==4, ix_dim==128, blk==128, local==1 :55-59) | `mm3_exl3/env/pack` | none | **IMPORT-AS-IS** |
| `mm3_model.hpp` | 220 | `mm3::Model` public API (load/forward/forward_cands/forward_rows/pipe_*/draft*/vision), `Options` :22-79, `Stats` :81-97, `Cands` :102, `DraftSampling` :111 | none | none | **IMPORT-AS-IS** |
| `mm3_model.cu` | 3549 | the engine: `Layer` :83-130 (x3::Mat per dense matrix, `wiq/wik` indexer projections or `wiq8/wik8` Q8_0 :101-102), `assemble()` :690 (expert blob from mapped shards), pinned pool build :1601-1665, staging + `cudaMemcpyAsync` :798, `stage()` :1800, `layer_window`/`window_pre` :~1810+, prompt path :1879-1935 (index_norm_rope -> block_scores -> select_blocks -> x3::mv qkv -> sparse_attn :1885-1931), decode `layer_token` :2615 (`T==1 && fast_decode` :2613), GGUF/BLAS prompt chunk path :2635-2662 (cublasSgemm :2641), KV streaming host cache :1005, `tune_key`/profile save (`strata::ds4::os::replace_file` :3340) | `../core/ablate_capture.hpp` :32 (`strata::ablcap::on/put`) — **MISSING in ours**; `ds4_numerics` transitively | `STRATA_USE_HIP` x3, `_WIN32` x1 | **ADAPT** — ablate capture stub or port (§2.5); else as-is |
| `mm3_tp.cu` | 2884 | tensor-parallel engine: GPU groups, host-resident residual stream, `wait_flag/wait_pull/publish` sync, per-GPU threads, CPU lane as tp sub-experts :472 | mm3-internal only (no strata headers) | none | **IMPORT-AS-IS** (optional target; phase 6) |
| `mm3_tp.hpp` | 59 | `mm3::TpEngine` class | `mm3_model.hpp` :19 | none | **IMPORT-AS-IS** |
| `mm3_cpu.hpp` | 77 | CPU lane: `CpuLane` class :34, `CpuExpert` :26 (EXL3 `Exl3Part[3]` or GGUF `GgFmt`), `gg_fmt/gg_run/gg_run_multi` :71-75 | `strata/kernels/cpu/exl3_cpu.hpp` :13 — **MISSING in ours as that path** (§2.4) | none | **ADAPT** — include-path rename to the vendored copy |
| `mm3_cpu.cpp` | 466 | CpuLane impl over `strata::kernels::cpu::ExpertPool` (`pool.hpp` :32 — present in OURS; `physical_cores` gained a `PoolAffinity` default arg, source-compatible :126 of ours) | `pool.hpp`, `exl3_cpu.hpp` :13, `mm3_lane_plan` :26 | none | **ADAPT** — same include swap |
| `mm3_cpu_gg.cpp` | 169 | GGUF experts on the lane: `namespace dg = strata::ds4::gguf` :20, `dg::iq_cpu_fmt` :46, swigluoai `oai()` :30-34 | `../ds4/ds4_gguf_cpu.hpp` :12 (both trees have it; ours differs by 50 diff lines — `IqCpu`/`iq_cpu_fmt` symbols present in ours, verified) | none | **IMPORT-AS-IS** against ours' header after symbol check at build time |
| `mm3_host.hpp/.cpp` | 19 / 58 | AVX2 host helpers (`dot`, `rms_rows`, `sum_into`, `pause`) kept out of nvcc's reach | none | GCC/clang target attr :13 | **IMPORT-AS-IS** |
| `mm3_dspark.hpp` | 83 | `mm3::Draft`: MiniMax M3.1 DSpark drafter (5 taps :3, block 7, window 4096, MXFP8 weights :11-13, Markov bias head) | `mm3_exl3.hpp` (head Mat), `mm3_kernels.cuh` | none | **IMPORT-AS-IS** |
| `mm3_dspark.cu` | 519 | Draft impl: mx_rows GEMVs, ring-window attention, propose/propose_sampled | `mm3_kernels.cuh` :13, `MAYA_MM3_DSPARK_FC16` env | `STRATA_USE_HIP` x1 | **ADAPT** — env rename only |
| `mm3_vision.hpp` | 67 | `mm3::Vision`: CLIP-style tower (Conv3d patch 2x14x14, 32 layers, projector 24576->6144), `set_lender` :49 (borrow staging slots) | `mm3_pack.hpp` | none | **IMPORT-AS-IS** |
| `mm3_vision.cu` | 666 | Vision impl, weights staged per encode ("nothing stays between pictures" :43-44) | `../ds4/ds4_numerics.cuh` :18 (BF16-on-HIP), `mm3_env` :20 | `STRATA_USE_HIP` x2 | **ADAPT** — numerics include swap only |
| `mm3_sample.hpp` | 164 | `mm3::sample`: `Req` :15-36, distributions from logits or `Cands`, penalties, speculative-sampling residual, think budget | `mm3_model.hpp` :14 | none | **IMPORT-AS-IS** |
| `mm3_cands.hpp` | 33 | host fallback for `k::topk_rows` (rows with more ties than room) | none | none | **IMPORT-AS-IS** |
| `mm3_slots.hpp` | 47 | `pick_slot` — --parallel conversation placement (shared-prefix reuse weighting) | none | none | **IMPORT-AS-IS** |
| `mm3_lane_plan.hpp` | 75 | worker-count arithmetic + decode lane timing model (pure host math) | none | none | **IMPORT-AS-IS** |
| `mm3_ablate.hpp` | 88 | `mm3::ablate::load` — control-vector direction (uses `strata::GgufFile`, then `strata::ds4::ablate::load_direction` :76) | `../ds4/ds4_ablate.hpp` :20 (**MISSING in ours**), `strata::GgufFile/MetaValue/TensorInfo` (present in ours, API-equal — class diff is 46 shifted lines, no signature change) | none | **ADAPT** — optional: vendor `ds4_ablate.hpp` (191-line header-only) or compile the feature out |
| `mm3_main.cpp` | 2218 | CLI + the engine protocol: `INFO engine=mm3 ...` :317, `READY <ctx> stop` :379, GEN/GENI parse :397-460, `T <id>` :584, `DONE ...` :752, `--parallel` multi mode (rid-tagged lines) :772-1186, `--calibrate-prefill`, sampler via `mm3::sample`; GENI pictures file MMI1 :96-127 | `strata/spec/draft_policy.hpp` :62 (`policy.trust` :619 — **maya-only method**), `strata/spec/suffix_drafter.hpp` :63 (ours equal: 6 diff lines), `mm3_cands/cpu/env/kernels.cuh/lane_plan/model/os/pack/sample/slots/vision` | `_WIN32` x3, `timeapi.h` :65 | **ADAPT** — DraftPolicy (§2.6) is the one real edit; `spec` targets exist in ours |

Counts: IMPORT-AS-IS 17 files ≈ 8,300 lines of mm3 + 4 vendored ds4 files ≈ 1,150; ADAPT 7 files (small, named edits) ≈ 9,830 lines touched but mostly unchanged content; SKIP 4 files (`mm3_ablate.hpp` optional port, and on the maya side `src/metal/mm3_*` — Metal-only, 1,387 lines, no CUDA equivalent wanted).

---

## 2. Shared-substrate diff (what mm3 pulls in vs what we have)

### 2.1 `../ds4/ds4_numerics.cuh` — present in ours, 9 diff lines
- Consumed symbols: `strata::ds4::bf16_rn` (ours :19), `e4m3_value` (:30), `e4m3_round` (:45) — all present in `/home/svg/LLM/strata-custom-2/src/dsv41/ds4_numerics.cuh` with identical signatures. The 9-line diff is a comment plus provenance header. The mm3 device helpers (`bf2f`, `f2bf`, `e4m3_dec`, `e2m1_val`, `had128w`) are defined inside `mm3_kernels.cu` on top of these.
- **Verdict: use ours as-is.** Include path in mm3 is relative (`../ds4/...`) — becomes `../dsv41/...` or a vendored copy next to `src/mm3/` (§5 recommends vendoring, matching the dsv41 precedent that vendored files live in the family dir).

### 2.2 `../ds4/ds4_iq.cuh` + `ds4_iq.cu` + `ds4_iq_cpu.cpp` + `ds4_gguf_cpu.{hpp,cpp}` — OURS DIVERGED
mm3's GGUF expert path needs maya's variant, and ours is NOT it:
- Maya `iq::Args` has `act` (`kActSwiglu`/`kActOai` — swigluoai), `tu` (gate/up of different types), `pexp` (maya ds4_iq.cuh:69-79). **Ours has none of these three fields** (ours ds4_iq.cuh:53-74 — struct ends at `tg, td`).
- Maya `SetArgs` has `skip_ge`, `xh_vec`, `act`, `tu`, `graw` (maya :96-113); ours instead grew `xf16` (ours :88-90) and a different function surface: ours exposes `hipk_force/hipk_launches/light_gate_up/light_down` (ours :112-125) which maya's lacks, and maya's HIP-only `MmqRowsScope` (maya :84) which ours lacks.
- Both share: `dequant_rows`, `tq` namespace, `MvJob`, `mv` (identical field-for-field).
- `ds4_gguf_cpu.hpp`: 50 diff lines; `IqCpu`/`iq_cpu_fmt`/CPU rows symbols mm3_cpu_gg.cpp needs are present in ours, but the maya version is what `MAYA_DS4_IQ_ONLY` builds were validated against.
- **Verdict: VENDOR.** Bring maya's `ds4_iq.cuh/.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.hpp/.cpp` into the mm3 family dir (e.g. `src/mm3/vendor/`), compiled with maya's `MAYA_DS4_IQ_ONLY` define (maya CMakeLists.txt:869) so the dsv41 kernels stay untouched. Do NOT extend our `dsv41/ds4_iq.cu` with `act/tu` — that would put MiniMax activation semantics inside dsv41's bit-verified kernels.

### 2.3 `../ds4/ds4_lt.{hpp,cu}` — MISSING in ours
- `namespace strata::ds4::lt { bool gemm(void* stream, const uint16_t* W, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy); void set_store(const char*); bool available(); }` (maya ds4_lt.hpp:9-26; impl ds4_lt.cu 231 lines, hipBLASLt with per-shape measured picks kept in a store file).
- mm3_gguf.cu calls it only for the AMD prompt path (:330, :335, :458; `set_gemm(1)`). On NVIDIA `set_gemm(0)` uses plain cuBLAS — our dsv41 does exactly that everywhere (ours ds4_model.cu:3028-3031, :3327 — no hipBLASLt in the model layer).
- **Verdict: VENDOR for AMD builds** (`roc::hipblaslt` link + `STRATA_HIPBLASLT_AVAILABLE`, maya CMakeLists.txt:864-866); CUDA-only builds never reference it at runtime but the symbol must link — vendor it unconditionally, it is 2 files.

### 2.4 `strata/kernels/cpu/exl3_cpu.hpp/.cpp` — MISSING in ours at that path, functionality vendored under dsv41
- MayA: `include/strata/kernels/cpu/exl3_cpu.hpp` (90 lines, `strata::kernels::cpu::Exl3Part` :19-24, `exl3_cpu_prepare/band/rows_acc/band_gemm` :44-88) + `src/kernels/cpu/exl3_cpu.cpp` (1497) + `exl3_cpu_avx512.inc` (283).
- Ours: `include/strata/dsv41/exl3_cpu.hpp` + `src/dsv41/vendor/exl3_cpu.cpp` — same lineage, **1134 diff lines** vs maya's (ours diverged with its own optimizations and `strata::dsv41k::exl3` namespace, ours include/strata/dsv41/exl3.hpp:27).
- mm3_cpu.hpp includes the `strata/kernels/cpu/exl3_cpu.hpp` path and `mm3_cpu.cpp` pools jobs through `strata::kernels::cpu::ExpertPool` from `pool.hpp` (which OURS has, with `PoolAffinity` added — source-compatible default arg, ours pool.hpp:126).
- **Verdict: VENDOR maya's exl3_cpu trio into the mm3 family dir** (namespace `strata::kernels::cpu` collides with ours' lib — either rename namespace to `strata::mm3k::cpu` (mechanical, ~40 references) or compile it into the mm3 static lib with the namespace preserved; the dsv41 precedent renamed to `dsv41k`).

### 2.5 `../core/ablate_capture.hpp` — MISSING in ours
- MayA only (`src/core/ablate_capture.hpp`); gives `strata::ablcap::on/put` (MAYA_ABLATE_CAPTURE direction-fitting readout). Included only by mm3_model.cu:32.
- **Verdict: SKIP + stub.** Two-line shim (`ablcap::on(){return false;} put(...){}`) unless we want ablation capture; `--ablate` itself (mm3_ablate.hpp + k::project_out) is independent of this file and needs only §2.6-adjacent `ds4_ablate.hpp` if enabled.

### 2.6 `strata/spec/draft_policy.hpp` — OURS IS THE SMALLER FORK
- Ours: `kMaxT=8`, methods `choose/observe/chain/observe_chain/chain_rate/lookup_rate/cost_ms/bucket/mtp_tokens/stale` (ours draft_policy.hpp:24-54).
- Maya adds: `kMaxT=16` (:38), `kBands=3` context bands (:40), `trust()` (:56, `kTrustMatch=12` :57), `accept_rate` (:69), `chain(const float* conf,...)` overload (:71), `tokens/window/plan/observe_cost/observe_drafts/row_ms/sample_cap` (:73-94).
- mm3_main.cpp uses `policy.trust(...)` :619 and `policy.choose(...)` :620,:647,:657, `policy.observe` :679,:724; mm3's sampled-draft path wants windows up to 16 (kWin=16 in mm3_model.cu:73).
- `suffix_drafter.hpp`: 6 diff lines — effectively equal.
- **Verdict: ADAPT.** Either vendor maya's DraftPolicy under the mm3 target (recommended; it is self-contained), or demote mm3 to kMaxT 8 + lose `lookup_trust`. Vendor: one header + draft_policy.cpp (ours 154-line impl vs maya's larger one — check impl diff at port time).

### 2.7 `strata::GgufFile` (artifact) — equal for mm3's use
- mm3 uses `GgufFile(path)`, `get()`, `tensors()`, `find()` (mm3_ablate.hpp:34-59). Both headers carry the same class; diff is 46-line offset + provenance. **Verdict: use ours.**

### 2.8 Not shared, and not needed by mm3
- maya `src/core/glm_*` (the GLM engine), `include/strata/core/glm_*`, `src/kernels/cuda/glmd_*`, `glm_dsa_*` parity — none referenced by mm3. Our tree's equivalents-absence is irrelevant.
- maya `src/metal/*` (`mm3_metal*.cpp`, `mm3_main.cpp`, `mm3_serve.cpp`, 1,387 lines for mm3) — Apple Metal path. **SKIP entirely.**
- maya `src/ds4/ds4_iq512_kern.inc`, `ds4_iq1_codes.hpp`, `ds4_lane_parity*`, `ds4_multi_attn.hpp`, `ds4_markov_q8.hpp`, `ds4_marks.hpp`, `ds4_draft_placement.hpp`, `ds4_borrow.hpp`, `ds4_host_ar.cpp`, `ds4_logprobs.hpp`, `ds4_cpu_prepare_rows.hpp`, `ds4_cpu_exl3.hpp`, `ds4_gguf.cu`, `ds4_kernels.cu`, `ds4_model.cu`, `ds4_vision`, `ds4_engram_io`, `ds4_evt/ablate beyond above` — NOT in mm3's include closure except the six files already covered (§2.1-2.3, §2.5-adjacent ds4_ablate.hpp).

---

## 3. MSA attention anatomy (mm3_kernels.cu + mm3_model.cu) — and the QSA comparison

**Geometry** (`mm3_geo.hpp:30-51`, checks :55-59): hd=128, `n_head == 16 * n_kv` (16:1 GQA), indexer: `ix_heads == n_kv == 4`, `ix_dim == 128`, block `blk == 128`, `local == 1`, `topk` 1..64 (model default 16 = "top-16-of-128-token blocks"). `sparse_attention_config.sparse_attention_freq` (mm3_geo.hpp:44) picks which layers are sparse; others are plain dense attention over the same caches.

**Per-layer flow, prompt path** (`mm3_model.cu:1879-1935`; decode `layer_token` :2334-2352):
1. Indexer projections: `index_q_proj`/`index_k_proj` from the normed hidden via `k::rows_dot` — or, in the GGUF pack, stored as Q8_0 rows and read by `k::rows_dot_q8` directly (mm3_model.cu:878-884; Layer fields `wiq/wik` vs `wiq8/wik8` :101-102). One kernel produces all index queries `iq [T][4][128]` and the single index key per token `ik [T][128]` — the indexer K is *not* per-head.
2. `k::index_norm_rope` (mm3_kernels.cu:96): per-head RMSNorm, partial NeoX RoPE, queries rounded through BF16 (model dtype), index key written into the index-key cache `ic [ctx][128]` BF16 at row `pos[t]` (mm3_model.cu:939).
3. `k::block_scores` (mm3_kernels.cu:401, plus `_tok` :452 for a single token and `_wide` :510 for prompts): score of block b for query t, head h = **max over the block's keys j<=pos[t]** of `iq[t][h] . ic[j]` (F32 fmaf chains, deterministic bit order — comment :398, :437). Output `bs [Hi][T][nblk]`. This is the "learned block-scores" stage; there is **no bias term and no ReLU** — max-pooling, unlike QSA.
4. `k::select_blocks` (`select_kernel` :594): one warp per (head, query); keeps **the query's own block first (`local=1`) then topk-1 rounds of the best remaining block**, ties to the lower block — exactly `torch.topk` set semantics with sparse_local_block 1 (:587-590). Output `kept [Hi][T][topk]`, `-1` padding. Blocks *after* the query are never candidates (score rows only written up to `last` :445).
5. `k::sparse_attn` (`sparse_attn_kernel` :769, `sparse_attn_w_kernel` :909 wide-decode variant, `attn_combine_kernel` :1059 for split-KV): per (query, KV group) a 128-thread block walks the group's kept blocks (`kept[g]` — **each KV group's own indexer head picks its own blocks**), 64-key tiles, **online softmax in F32**, hd=128. `kPart=130` partials (128 sums + max + total) when `S>1` splits (MAYA_MM3_ATTN_SPLIT / decode windows), combined in run order — same-bits guarantee (:902-906).
6. **No sink/streaming logits part.** grep for `sink` in mm3: zero hits. The "streaming" that exists is KV *streaming* (host spill), not attention sinks.

**KV layout and formats** (`mm3_kernels.cuh:31-83`): caches are **contiguous by position, not paged**: `kc/vc [ctx][Hkv][128]` BF16 (Layer :96-97), or quantized `KvQ` — INT8 (row max/127), FP8 E4M3 with UE8M0 power-of-two scale per 32, FP4 E2M1 with E4M3 scale per 16 — optionally with a 128-point Hadamard rotation applied to rows before rounding (`rot` bit0 K / bit1 V; queries rotated alike at :788-798, output rotated back :888-899). Default in Options is **INT8 + rot** (mm3_model.hpp:65-66). Index keys stay BF16 always (:64). Attention decodes tiles to BF16 in shared memory (`load_tile<F>` :683-758).

**KV streaming / host spill** (`KvHost` :78-83, `kv_row` :668-673): VRAM cache is a ring of `kv_resident` rows (position p at row `p % ring`, `tag[row]`); every position also lives in pinned, mapped host memory; attention reads a key from VRAM when the ring holds it, else over PCIe. Host cache sizing at mm3_model.cu:1125, :1005.

**Prefill vs decode**: prompt runs in chunks (`Options.chunk`, calibration shrinks it :991-1005) through the same sparse path; GEMMs are cuBLAS FP16 over decoded FP16 weight rows (GGUF: `gemm_rows` decodes bands; BLAS prompt chunk path mm3_model.cu:2635-2662; EXL3: `mma` tiles sm_80+, cuBLAS FP16 over fully decoded weights before sm_80, `blas_enable`). Decode is `layer_token` (:2613 `T==1 && fast_decode`) — GEMV `x3::mv` qkv :2340-2341, same indexer/sparse_attn kernels at T=1 (:2334-2351) with the `_tok`/wide kernels auto-selected (MAYA_MM3_MSA_TOK/MAYA_MM3_MSA_WIDE). Verify windows (T<=16, kWin mm3_model.cu:73) reuse the prompt path with `win_experts`.

**vs our qwen35 QSA** (`include/strata/kernels/qsa.hpp` — the spec of record, and `src/qwen35/q36_kernels.cuh`):

| Property | mm3 MSA | ours QSA |
|---|---|---|
| Index unit | 128-token block, one pooled score per block (max over key dots) | per-cell scores; indexer pools one key per 4 cells (`idx_block=4`, qsa.hpp:103) |
| Score form | `max_j dot(iq_h, ic[j])` over block keys, F32, no bias, no activation | `bias[cb] + sum_h Relu(dot(pooled[cb], q_idx[h]))`, tail bias +1e9 (qsa.hpp:258-260) |
| Selection | own block + topk-1 best earlier blocks, ties low (topk blocks) | largest `width` cells ascending (`idx_top_k=2048 cells`), identity under bound (qsa.hpp:133-136) |
| KV store | contiguous `[ctx][Hkv][128]`, optional INT8/FP8/FP4 + Hadamard rot, host ring spill | paged FP16 `[page][kv_head][page_size][head_dim]` with page table, gather to scratch (qsa.hpp:63-77, :186-195) |
| Attention input | reads kept blocks straight from cache | gathers selected cells into scratch, then attends (kv_gather) |
| Gate | none | `attn * sigmoid(gate)` (qsa.hpp:303-312) |
| Graph capture | not graph-safe (host scalars T/pos/ring in args) | graph-safe `*_step` variants with device step buffer (qsa.hpp:138-168) |
| Geometry | hd 128, 16:1 GQA, 4 index heads = 4 KV groups | hd 256, 24/2 heads, 4 index heads (qsa.hpp:94-105) |

**Conclusion:** the two are cousins in shape (few index heads, top-k over an indexer) but NOT in contract — different score form, different cache layout, different selection unit. **mm3's MSA is fully self-contained** (namespace `mm3::k`, zero `strata::kernels` references), so it imports wholesale and *neither adapts to QSA nor should*: keep both indexers as separate families. No reuse of `qsa_select/qsa_scores/qsa_decode_attn` is possible without rewriting the numerics contract. The only genuinely shared concern is the CUDA-graph pattern — if our mm3 engine later needs graph capture, `KvHost` ring indices and `pos` must move into a step buffer the way qsa.hpp:138-168 does; that is new work, not an adaptation.

---

## 4. The GGUF expert serving path (mm3_gv.cu + mm3_gguf.cu + mm3_cpu_gg.cpp) vs our dsv41 pool/store

**What the blobs are.** Routed experts live as per-layer GGUF blobs: each expert = `gate | up | down` concatenated, each matrix `[rows][blocks]` of one ggml type, 256-byte aligned (`ExpertLayout::trellis[r]` = role offset, `gg_type[r]` per role — mm3_exl3.hpp:104-111, mm3_gguf.cuh:17-18). Dense matrices are Q4_K/Q5_K/Q6_K/Q8_0; routed experts are the low-bit i-quants (IQ1_S..IQ3_XXS per tools/mm3/pack_gguf.py:2) — this is the "vector-quantized experts per-layer GGUF blobs" (the codebooks live inside the quant grids; there is no separate GVQ1/GVR8 container in code — those names map to the IQ* grid families served by ds4_iq).

**Device path** (`mm3_gguf.cu`): activations rounded to **Q8_K** (scale per 256, so CPU lane and GPU produce the same sums; `MAYA_MM3_IQ_Q8K=0` drops to per-32) by `act_quant` (decl mm3_gguf.cuh:70-74); `gate_up`/`down` kernels come from maya's `ds4_iq.cu` (`iq::gate_up/down`, grid (n_ff/8, groups) — maya ds4_iq.cuh:82-86) with `act=kActOai` fusing swigluoai. Entry points stand behind the x3 interface (`fmt=kFmtGG`), so token/window/prompt paths are format-agnostic (mm3_gguf.cuh:4-16).

**Dense GEMV path** (`mm3_gv.cu`): Q4_K/Q5_K/Q6_K/Q8_0 GEMVs that keep ds4_iq's `mv` bit-exact sums but read faster (warp covers R rows, wide loads); CUDA-only — on HIP and on uncovered jobs it falls back to ds4_iq's `mv` (mm3_gv.cuh:39-47). Dequant happens **on the GPU at GEMV time** (no gather stage, no decoded copy); prompt path decodes to FP16 bands and uses cuBLAS/hipBLASLt (`gemm_rows`, `set_gemm` mm3_gguf.cuh:59-65).

**Host→device lane design** (mm3_model.cu — NOT dsv41's store):
- Load: experts resident by VRAM fit per layer (`res_slot/res_ids` :108-109); the rest assembled once into a **pinned host pool** (`o.pin_pool`, `assemble()` :690 copies the three matrices from the mapped shards into pool bytes, :1601-1665) — one expert = one contiguous blob so a staging copy is a single DMA.
- Serve: 48 staging slots/GPU (`Options.staging` :35, slot arrays :129-140). A routed miss either goes to the **CPU lane** (`CpuLane::run/run_multi`, mm3_cpu.hpp:43-51, reading pool blobs from host RAM) or is `cudaMemcpyAsync`'d pool→slot on the copy engine (:798) and computed by the GPU; `calibrate_lane` (:712-837) times both at load and splits per-token.
- **Promote**: hot misses get copied into a slot permanently with decay scoring (:405-411, `promote_lag` :52-53) — a small software cache over the slots.
- TP path keeps the same scheme with per-member slices (mm3_tp.cu:472).

**vs our dsv41** (`src/dsv41/ds4_store.hpp`, `ds4_pool.hpp`, `src/core/expert_source.cpp`):
- Ours' `ExpertStore` (ds4_store.hpp:38) is a host slot cache with **GPU-mapped chunks** (`Cfg::map_chunk` :67-68, `Vis{dev}` visibility records :53) — the GPU reads host RAM directly over BAR, no cudaMemcpyAsync staging; plus `IoQueue` direct-IO reads (ds4_pool.hpp:137) and heat eviction (`heat_` :93). mm3 instead copies (DMA) and never maps expert blobs to the GPU. Two different transport contracts; **do not merge them for the import** — mm3's model code (`mm3_model.cu`) owns its scheme end-to-end and never calls core/expert_source (verified: no `expert_source|ExpertSource|expert_cache` symbol in mm3_model.cu/mm3_tp.cu).
- Ours' `strata::kernels::cpu::ExpertPool` (pool.hpp:156) IS shared by mm3's CPU lane (mm3_cpu.cpp:32) — the one real substrate reuse in the expert path, and it is source-compatible (`physical_cores` gained a defaulted `PoolAffinity` arg, ours :126).
- `core/expert_source.cpp` (ours, 6301 lines) is our cross-family orchestration layer; mm3 bypasses it by design (its slot/pool/profile machinery predates and duplicates parts of it). Import decision: leave mm3 self-contained in phase 1; a later phase may back its pinned pool with `ExpertStore` if BAR mapping wins on the target hardware — that is an optimization, estimated separately (§7 phase 7), not a prerequisite.

**Codebook residency:** i-quant grids are inline in the blob bytes (ggml grid tables compiled into ds4_iq kernels as `__constant__`/device tables — see maya ds4_iq.cu); nothing to manage at runtime. MXFP8 (DSpark draft) is the exception: E4M3 + UE8M0 scales decoded by `mx_rows` on the fly (mm3_kernels.cuh:191-198).

---

## 5. Boot / wiring surface

**maya side** (`src/mm3/mm3_main.cpp` + CMakeLists.txt:833-940):
- `maya-mm3` executable = mm3_main.cpp + mm3_model.cu + mm3_kernels.cu + mm3_cpu.cpp + mm3_host.cpp + `mm3_exl3/dspark/vision/tp.cu` + `mm3_gguf.cu mm3_gv.cu ds4_iq.cu ds4_lt.cu` + `mm3_cpu_gg.cpp ds4_iq_cpu.cpp ds4_gguf_cpu.cpp` (maya CMakeLists.txt:856-861), links `strata_kernels_cpu strata_spec`, define `MAYA_DS4_IQ_ONLY` (:869), arch floor sm_70 (:155-156, :837-850).
- Protocol = the same stdin/stdout engine protocol our server.py already drives for dsv41: `INFO engine=mm3 ...` (mm3_main.cpp:317), `READY <ctx> stop` :379 / `READY <ctx> stop multi=N` :786, `GEN`/`GENI <file> <ids>` (pictures MMI1 :96-127), `T <id>` :584, `STAT` per 8 tokens, `DONE <generated> <prompt> <prompt_ms> <decode_ms> <reason> ...` :752, `STOP [rid]`, `QUIT`. `INFO vision=mm3 image_token=...` :333.
- Pack tooling: `tools/mm3/pack.py` (safetensors -> EXL3 pack, `ARCH` in pack_gguf.py:115 = `minimax-m3` GGUF), `pack_gguf.py` (GGUF variant: dense Q4_K..Q8_0 + experts IQ1_S..IQ3_XXS), `calibrate.py`, `kv_gate.py`, `ref.py/ref_gguf.py/st.py` references, `chat_template.jinja`, `tokenizer/` writer.

**ours side, the pattern to follow** (both precedents exist):
1. dsv41 precedent — own executable: `option(STRATA_BUILD_DSV41)` + `strata-dsv41` target, vendored kernels in a private static lib so **no symbol is shared with strata_kernels** (ours CMakeLists.txt:2309-2319). Recommend the same: `option(STRATA_BUILD_MM3)`, static `strata_mm3_kernels` = `src/mm3/vendor/exl3.cu`-equivalents (i.e. mm3_exl3.cu, mm3_gv.cu, mm3_gguf.cu, vendored ds4_iq.cu, ds4_lt.cu, exl3_cpu.cpp, exl3_legacy if kept) and `strata-mm3` executable (or a `strata_mm3` static lib + family switch, qwen35-style).
2. qwen35 precedent — family switch inside the main `strata` engine: `strata::model_family(meta) == ModelFamily::Qwen35Moe` -> `strata::qwen35::run(argc, argv)` (ours src/program/generate.cpp:2511-2523). For mm3 as a third branch: add `ModelFamily::MiniMaxM3` mapped from arch `"minimax-m3"` in `include/strata/artifact/model_family.hpp:21-24`, extend `model_family_runs` :54, and route to `strata::mm3::run`.
3. server.py needs (all small): `FAMILY_MODEL_NAMES` entry for `minimaxm3` (ours serve/server.py:8357-8365), family sampling defaults table :8377+, `set_family` hook :4223-4238 (MiniMax template/end tokens differ from the Qwen form), config JSON in `config/` naming the new exe (pattern: `config/custom-dsv41-8081.json`), and vision passthrough — GENI/pictures already supported for dsv41 (same MMI1 file format), no parser work. The `INFO engine=mm3` line is new: `engine_kind` handling in ready/handshake (ours serve/ready.py, handshake.py) — one branch.
4. env prefix: mm3 reads `MAYA_*`/`MM3_*` (mm3_env.hpp:10-15). Our convention is `STRATA_*` (e.g. `STRATA_DS4_SLOT_KEEP` in server.py:3448). Rename in mm3_env.hpp; the ~25 `MAYA_MM3_*` / `MM3_*` knobs (list in §1 mm3_env row) keep their suffixes.

---

## 6. Adaptation risk register (top 10, each with evidence)

1. **ds4_iq fork collision (highest).** mm3 needs maya's `iq::Args.act/tu/pexp` + `MmqRowsScope`; ours replaced them with `xf16/light_*/hipk_*` (§2.2). Extending ours breaks dsv41's bit-verified kernels; linking both under one namespace collides. Evidence: struct diffs at maya ds4_iq.cuh:53-113 vs ours :53-125. Mitigation: vendor under `src/mm3/vendor/` + `MAYA_DS4_IQ_ONLY`, rename namespace to `strata::mm3k::…` if linkers complain about static-lib ODR (both are static libs, so collision is actually unlikely — but the dsv41 comment "no symbol is shared" (ours CMakeLists:2311) exists precisely because of this).
2. **EXL3 CPU lane namespace + AVX2 build flags.** `strata::kernels::cpu::exl3_cpu_*` (§2.4) must not merge with ours' diverged copy (1134 diff lines). mm3_lane_test builds it with `-mavx2 -mfma -mf16c` (maya CMakeLists.txt:918-921). Risk: silently linking ours' (different bits) instead of maya's — outputs would still be finite. Mitigation: namespace rename to `mm3k` at vendor time.
3. **DraftPolicy API gap.** mm3_main.cpp calls `policy.trust()` (maya-only) and relies on `kMaxT=16` windows (kWin=16, mm3_model.cu:73); ours caps at 8 and lacks trust/bands (§2.6). Mitigation: vendor maya's spec pair for the mm3 target only.
4. **HIP divergence in the attention kernels.** mm3_kernels.cu carries gfx906 wave64 paths (:486-497, :597-660) and STRATA_USE_HIP branches x10, but nothing for gfx1100/gfx12 (our tree has `qsa_scores_gfx1100_parity`, `gfx_arch.hpp`, RDNA4 prefill in ours' prefill/). On RDNA targets mm3 compiles via the generic HIP path only; expect untested behavior and possible LDS-size failures (`block_scores_wide` kBwK=64 on AMD :505-508 assumes 64 KB LDS). Evidence: guards listed §1; ours has newer-arch parity tests for QSA but none for MSA. Mitigation: first bring-up on the CUDA target; gfx1100 bring-up is its own phase item.
5. **sm_70 floor + pre-sm_80 cuBLAS decode path.** mm3 refuses Pascal (maya CMakeLists:837-850) and keeps a Volta/Turing wmma + FP16-GEMM-BLAS decode path (`blas_enable`, MM3_EXL3_BLAS_ROWS, exl3_legacy not used by mm3 but its wmma kernels are in mm3_exl3.cu:1779-1855). If our fleet is sm_80+/RDNA only, this is dead code to carry; if any V100/T4 remains, it must be exercised (its bits differ from the mma path by design, not by accident).
6. **KV-cache format assumptions in serving code.** Options default `kv_k=kv_v=1` (INT8) + `kv_rot=true` (mm3_model.hpp:65-66); server-side surfaces that assume exact BF16 caches (ours' telemetry/vram maps) will report different capacities, and the `kv_gate.py` calibration decides FP4. Risk is misconfiguration, not wrongness — kernels are exactness-tested (maya tests/mm3_kv_test.cu, CMakeLists.txt:902-908).
7. **Env-prefix drift.** 25+ `MAYA_MM3_*`/`MM3_*` knobs (§1). A silent half-rename (some read STRATA_, some MAYA_) would make flags no-ops. Mitigation: single choke point mm3_env.hpp:10-15; grep-audit after rename (list captured above).
8. **`ablate_capture` + `ds4_ablate` optional deps.** mm3_model.cu:32 hard-includes `../core/ablate_capture.hpp`; mm3_ablate.hpp:20/76 needs `ds4_ablate.hpp` + its `strata::GgufFile` usage. Missing files = build failure at phase 1 even if the feature is off (the include is unconditional). Mitigation: stub header in phase 1 (§2.5), real port optional.
9. **Windows-only paths rotting the Linux build.** `_WIN32` blocks in mm3_os.hpp (x6), mm3_main.cpp (x3, `timeapi.h` :65, `#pragma comment(lib, "winmm.lib")` :66-68), mm3_kernels.cu VirtualUnlock. Harmless on Linux but keep or strip consistently; mm3_os.hpp's cgroup walk (:38-68) is Linux-only and load-bearing for memory planning.
10. **Server/protocol `engine_kind=mm3` + vision fan-out.** `INFO engine=mm3` (mm3_main.cpp:317) is a new engine identity for ready.py/handshake/monitor; GENI image flow requires the MMI1 pictures file path the dsv41 flow already writes (same format — verified mm3_main.cpp:96-127 vs ours' dsv41 ds4_main.cpp:10-12 GENI description). Risk: monitor VRAM accounting (`vram_lines` mm3_model.hpp:175) keyed by dsv41 shapes. Mitigation: one branch per consumer; test with serve fixtures.

Non-risks, recorded so nobody re-investigates: `dp4a.hpp`, `pool.hpp`, `gguf_reader.hpp`, `suffix_drafter.hpp`, `ds4_os.hpp`, `ds4_gguf_cpu.hpp` symbols, `ds4_numerics.cuh` — all present and API-equal in ours (verified by symbol greps and diff line counts in §2).

---

## 7. Sized work order

Assumption: one engineer, CUDA-first target, GGUF pack first (the MiniMax Maya quants), EXL3 pack second; dsv41 stays untouched; no Metal; AMD/RDNA bring-up tracked as its own phase. Days are focused days, honest, including the parity tests ported with each phase.

### Phase 0 — scaffolding (0.5 d)
- `option(STRATA_BUILD_MM3)`, `strata_mm3_kernels` static lib + `strata-mm3` exe target skeleton; third_party/project-maya NOTICE line; `ModelFamily::MiniMaxM3` + arch `"minimax-m3"` in model_family.hpp:15-60; mm3_env.hpp prefix swap.
- Files: CMakeLists.txt, include/strata/artifact/model_family.hpp, src/mm3/mm3_env.hpp.

### Phase 1 — vendored kernel substrate (2.5 d)
- Vendor: maya `ds4_iq.cuh/.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.{hpp,cpp}`, `ds4_lt.{hpp,cu}`, `ds4_ablate.hpp` (optional), `exl3_cpu.{hpp,cpp}` + `exl3_cpu_avx512.inc` (+ namespace rename to `mm3k`), `ablcap` stub.
- Wire with `MAYA_DS4_IQ_ONLY`; build mm3_exl3.cu + mm3_gv.cu + mm3_gguf.cu against them; port tests `mm3_exl3_test`, `mm3_gguf_test` (maya CMakeLists.txt:872-908) and get them green on one GPU.
- Files: src/mm3/vendor/* (7), src/mm3/mm3_exl3.cu, mm3_gv.cu, mm3_gguf.cu, tests/mm3_*.

### Phase 2 — geometry, pack, loader (2 d)
- mm3_pack.hpp + Geo + mm3_gguf_host.hpp; load path in mm3_model.cu up to first logits (`--ids` greedy run, no serve): device.cu interplay (ours' device.cpp vs maya's — both trees have src/core/device.cu, differ, but mm3 only needs a CUDA context + streams: verify our `strata_core` device init API matches maya's usage in mm3_model.cu load), pack.py run against a real MiniMax-M3.1 checkpoint (GGUF path first).
- Files: mm3_pack.hpp, mm3_geo.hpp, mm3_model.cu (load half), tools/mm3/pack_gguf.py + pack.py.

### Phase 3 — attention + decode correctness (2.5 d)
- mm3_kernels.cu/.cuh import; greedy + `--dump-logits` parity against `tools/mm3/ref_gguf.py`; prefill chunk path + BLAS GEMM selection; decode `layer_token`; KV INT8+rot default path; port `mm3_kv_test` + `mm3_kernels_test`.
- Files: mm3_kernels.cu/.cuh, mm3_model.cu (forward half), tests/mm3_kv_test.cu, tests/mm3_kernels_test.cu.

### Phase 4 — expert serving (resident/staging/pool/promote/CPU lane) (3 d)
- mm3_model.cu slot/pool machinery + mm3_cpu.cpp + mm3_cpu_gg.cpp + mm3_lane_plan.hpp + calibrate_lane; profile save/load (`replace_file`); Stats/vram_lines.
- Files: mm3_model.cu (expert half), mm3_cpu.cpp/.hpp, mm3_cpu_gg.cpp, mm3_host.cpp/.hpp, mm3_lane_plan.hpp, mm3_cands.hpp.

### Phase 5 — sampler, DSpark, vision (2.5 d)
- mm3_sample.hpp + main serve loop sampling; mm3_dspark.cu/.hpp (MXFP8 path, Markov head); mm3_vision.cu/.hpp + GENI in main; port `mm3_sampler_test`.
- Files: mm3_sample.hpp, mm3_dspark.cu/.hpp, mm3_vision.cu/.hpp, mm3_main.cpp (sampling+GENI).

### Phase 6 — server wiring + protocol (2 d)
- mm3_main.cpp --serve (single + `--parallel` multi mode), server.py family entries + ready/handshake `engine_kind=mm3`, config/custom-mm3-8081.json, chat template + tokenizer from pack_gguf, launcher script; end-to-end chat against server.py.
- Files: mm3_main.cpp, serve/server.py, serve/ready.py, serve/handshake.py, config/*.json, tools/mm3/* (tokenizer/template).

### Phase 7 — pack tooling hardening + optional extras (2 d)
- tools/mm3/calibrate.py + kv_gate.py + ref.py/st.py oracle checks; EXL3 safetensors pack path (`mm3_exl3` fmt-0) end-to-end; optional: mm3_tp.cu behind a flag; optional: ds4_ablate real port + --ablate; optional: ExpertStore-backed pool experiment.
- Files: tools/mm3/*, src/mm3/mm3_tp.cu (opt), src/mm3/mm3_ablate.hpp (opt).

### Explicitly deferred / skipped
- maya src/metal/mm3_* (Metal engine) — skip, no owner here.
- RDNA (gfx1100/gfx12) bring-up of MSA kernels — new parity work, unsized here, evidence §6.4.
- CUDA-graph capture of the MSA layer (step-buffer refactor like qsa.hpp:138-168) — new work, unsized.
- Merging mm3's expert transport with core/expert_source.cpp / ExpertStore — optimization study only.

**Total: ~17 focused days** to a serving third family on CUDA (phases 0-6, 15 d) plus 2 d of tooling/extras; the 80/20 core (greedy generation + GGUF experts + serve loop) lands at phase 4 (~11 d).
