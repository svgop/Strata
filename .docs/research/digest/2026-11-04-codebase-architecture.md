---
type: research-digest
id: 2026-11-04-codebase-architecture
topic: software architecture of elite inference engines — drift-proof structures
method: shallow clones of llama.cpp / vllm / sglang / safetensors / onnxruntime at HEAD 2026-11-04; every mechanism below read in source, not recalled; OSS-Fuzz claims checked against google/oss-fuzz (2 negative results)
head-shas:
  llama.cpp: 9e2a04a439edc49f08c5acdafefae19dfa25a658 (2026-11-04)
  vllm: 42c043c6b0e4905d590e9cb4bf00bb644bfebb53 (2026-11-04)
  sglang: b702f0d80e5c2161e7a4a0470e3435924f983ff8 (2026-11-04)
  safetensors: 2f216f9a7105e1144004072d4611cc44c3ee2365 (2026-11-03)
  onnxruntime: f340e685c80bdb163ce01a44a9e88797d10c426f (2026-11-04)
---

# Digest 2026-11-04 — Inference-engine codebase architecture: how the elite make drift structurally impossible

Scope: mechanisms that keep multi-backend, multi-variant perf codebases coherent with small teams. Verdicts target the strata-custom-2 shape: two engine trees (Qwen-class graph engine, DS-class serial-round engine), shared substrate, Python serve layer on a line protocol, CMake per-engine targets, bench fleet, team of one + agents.

## 0. Corrections to the brief (verified negatives)

- **OSS-Fuzz has no llama.cpp project.** `google/oss-fuzz` contains no `projects/llama.cpp` (API 404, checked 2026-11-04). GGUF parser fuzzing upstream = `tests/test-gguf.cpp` round-trip tests, not OSS-Fuzz. https://github.com/ggml-org/llama.cpp/blob/master/tests/test-gguf.cpp
- **OSS-Fuzz has no safetensors project either.** Fuzzing lives in-tree as a cargo-fuzz target: https://github.com/huggingface/safetensors/tree/main/safetensors/fuzz — one libfuzzer target that hurls arbitrary bytes at `SafeTensors::deserialize` (https://github.com/huggingface/safetensors/blob/main/safetensors/fuzz/fuzz_targets/fuzz_target_1.rs). Not CI-gated.
- **vLLM's v0/v1 fork is closed — by deletion.** No `VLLM_USE_V1` remains in `vllm/envs.py` (HEAD). V1 is the only engine. The fork-then-migrate lifecycle is documented in the V1 post: https://blog.vllm.ai/2025-01-27-v1-alpha-release.html

## 1. llama.cpp: 15+ backends without drift

### 1.1 The backend registry: compile-time ifdefs + runtime dlopen, one enumeration point

- **Mechanism.** `ggml/src/ggml-backend-reg.cpp` holds a process-global `ggml_backend_registry`. The constructor calls `register_backend(ggml_backend_X_reg())` per compiled backend, each guarded by `#ifdef GGML_USE_X` (CUDA, Metal, SYCL, Vulkan, WebGPU, zDNN, VirtGPU, OpenCL, ZenDNN, Hexagon, CANN, BLAS, RPC, OpenVINO, ET, CPU). `register_backend` dedupes by pointer; the registry owns `backends` + flattened `devices`. Dynamic loading: `ggml_backend_load_all_from_path()` dlopens backend libs built with `GGML_BACKEND_DL` (loader in `ggml/src/ggml-backend-dl.cpp`), so vendors ship .so without touching the registry switch. Runtime kill-switch precedent: Vulkan checks `GGML_DISABLE_VULKAN` env at registration. https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-reg.cpp
- **The device interface is the drift-killer.** `ggml/include/ggml-backend.h` defines per-device probes: `ggml_backend_dev_type()` (CPU/GPU/IGPU/ACCEL/META), `ggml_backend_dev_memory()`, `ggml_backend_dev_supports_op(dev, op)`, `ggml_backend_dev_offload_op(dev, op)`, and resolution by type (`ggml_backend_dev_by_type`, `ggml_backend_init_by_type`). Consumers enumerate devices from the registry; nobody hand-lists backends downstream. https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h
- **Vendor isolation.** Each backend lives in its own directory (`ggml/src/ggml-cuda/`, `ggml/src/ggml-metal/`, …) with its own CMakeLists and a private header; the only contract with the core is `ggml-backend-impl.h` + the exported `_reg()` factory. Per-vendor CI: `.github/workflows/build-cuda-ubuntu.yml`, `build-vulkan.yml`, `build-sycl.yml`, `build-opencl.yml`, `ci-self-hosted-{cuda,metal,vulkan,openvino,cpu,webgpu,kleidiai}.yml` — one workflow file per vendor, self-hosted runners for GPU truth. https://github.com/ggml-org/llama.cpp/tree/master/ggml/src , https://github.com/ggml-org/llama.cpp/tree/master/.github/workflows
- **Drift class killed:** "new backend added but consumers don't see it" (manual lists), "backend compiled but not linked", vendor toolchain leaking into core.
- **Verdict: ADAPT.** Do not clone the full device API; clone the shape — one registry in the shared substrate, each engine tree registering itself as a "device" with `supports_op`-class probes, consumers enumerate. Direct fit for the shared-substrate layer linking both engines.

### 1.2 test-backend-ops: differential parity with zero manual backend lists

- **Mechanism.** `tests/test-backend-ops.cpp` (~14k lines): a `test_case` class hierarchy enumerates every op × param grid; `main` walks `ggml_backend_dev_count()/ggml_backend_dev_get(i)` — the registry, not a list — so every new backend is auto-covered. Four modes: `MODE_TEST` (compare vs CPU with `ggml_backend_cpu_set_use_ref` reference kernels), `MODE_GRAD`, `MODE_SUPPORT` (record supports_op verdicts), `MODE_PERF`. Plus `make_test_cases_from_file(path)` — recorded op tests replay from golden files. Parallel workers with shrinking-chunk work-stealing. https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp
- **The support matrix is committed documentation, generated by CI.** `MODE_SUPPORT` output lands as `docs/ops/{CUDA,Metal,Vulkan,CPU,...}.csv` with columns `backend_name,op_name,op_params,test_mode,supported,error_message,backend_reg_name`; `.github/workflows/update-ops-docs.yml` regenerates on drift. A stale capability claim fails CI. https://github.com/ggml-org/llama.cpp/tree/master/docs/ops
- **Drift class killed:** op works on CPU but silently diverges on the other pipeline; docs claiming op support that regressed; new backend missing op-coverage truth.
- **Verdict: ADOPT-AS-IS (renamed).** One differential harness over the shared substrate's ops, engines enumerated from the registry, support matrix exported to a committed CSV refreshed by CI. This is the single highest-value structure for a two-engine repo.

### 1.3 test-llama-archs: whole-graph parity per declared variant, NMSE ≤ 1e-4

- **Mechanism.** `tests/test-llama-archs.cpp` builds tiny synthetic models for each declared arch (`arch_matches()` filters by `llama_arch_name`), runs logits on CPU and on each GPU device, compares with normalized MSE and FAILS above `1e-4`; also compares chunked/mixed compute paths (`nmse(logits_cpu, logits_dev)`, chunked variant). https://github.com/ggml-org/llama.cpp/blob/master/tests/test-llama-archs.cpp
- **Drift class killed:** pipeline-level (not op-level) numeric divergence after a refactor — the class op tests miss.
- **Verdict: ADOPT-AS-IS.** One tiny Qwen-class fixture + one DS-class fixture; every engine change must hold `nmse < 1e-4` against its own CPU/serial reference. Cheap on a 12 GB GPU box.

### 1.4 llama-arch: declared branches on one spine — and where it drifts

- **Mechanism.** ~150+ architectures declared as `LLM_ARCH_*` in `src/llama-arch.h` with name maps in `src/llama-arch.cpp`; graph construction dispatches per-arch. But dispatch is a **switch repeated at multiple sites** (`src/llama-model.cpp` lines ~47, ~2223, ~3631 in HEAD) — adding an arch means touching several switches; upstream compensates with the archs test, not with a single dispatch table. https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp
- **The cleaner spine: the memory family.** `src/llama-memory.h` defines one context interface implemented per variant (`llama_kv_cache_context`, `llama_kv_cache_iswa_context`, `llama_memory_recurrent`, `llama_memory_hybrid`, …) — deliberately different pipelines (attention-KV vs recurrent state vs hybrid) behind one interface, composed at build time. https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory.h
- **Drift class killed:** variant added to one pipeline but missed in another phase (prep/step/rewind). Upstream still tolerates multi-switch edits; the interface family is the part that scales.
- **Verdict: ADAPT.** For DS-class serial rounds vs Qwen-class graphs: one memory/context interface, one enum, one dispatch site, one fixture per branch in the archs test. Do not copy the repeated switches.

### 1.5 ggml-rpc: the typed engine↔server handshake with a semver gate

- **Mechanism.** `ggml/include/ggml-rpc.h` pins `RPC_PROTO_MAJOR_VERSION 8 / MINOR 0 / PATCH 1`. Client sends `rpc_msg_hello_req`, server replies `rpc_msg_hello_rsp{major,minor,patch}`; compatibility rule: **major must equal, server minor must not exceed client's** (`ggml/src/ggml-rpc/ggml-rpc.cpp:435`). Commands are a fixed enum of typed, length-prefixed messages. https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-rpc.h
- **Drift class killed:** serve layer and engine binary silently disagreeing on wire format after one side updates.
- **Verdict: ADOPT-AS-IS.** This is the four-layer program's control-plane handshake, exactly. Hello message on connect; reject on major mismatch or minor ceiling violation; print both versions in the error. ~50 lines.

### 1.6 Golden-hash and golden-snapshot gates

- **Pre-tokenizer hashes.** `.github/workflows/pre-tokenizer-hashes.yml`: regenerates pre-tokenizer hashes via `convert_hf_to_gguf_update.py --check-missing`; CI fails if the committed hashes differ. Any behavior change to tokenization is a declared, reviewed diff. https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/pre-tokenizer-hashes.yml
- **GGUF metadata snapshots.** `tests/snapshots/*.schema` store golden metadata for real HF models (qwen3-0.6b, gpt-oss-120b, glm-4.6v, …); `test-quant-type-selection.cpp` compares derived quant decisions against `SNAPSHOT_DIR` (wired in `tests/CMakeLists.txt`). https://github.com/ggml-org/llama.cpp/tree/master/tests/snapshots
- **Drift class killed:** silent semantic change in owned formats (vocab, quant selection) — the "it still builds and passes unit tests" drift.
- **Verdict: ADOPT-AS-IS.** Golden-hash any owned-format output that should change only with a human-reviewed diff.

### 1.7 Agent+human discipline, upstream form

- **AGENTS.md policy.** "AI-generated code is allowed. What is not allowed is submitting code you do not understand. You are 100% responsible for every line." Prohibits AI-written PR descriptions/commits; autonomous agents told to exclude the repo. https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md
- **Agent skills encode repo invariants:** `skills/add-new-model` (the arch multi-switch checklist), `skills/code-review`, `skills/slice-model`, `skills/update-fusion-table` — the drift-prone procedures are packaged as checklists instead of tribal memory. https://github.com/ggml-org/llama.cpp/tree/master/skills
- **Server contract tests:** `tools/server/tests/` — pytest suite hitting the real HTTP server (`tests.sh` orchestrates), separate sanitize/self-hosted workflows (`server.yml`, `server-sanitize.yml`). https://github.com/ggml-org/llama.cpp/tree/master/tools/server/tests
- **Verdict: ADAPT.** Keep the ownership invariant verbatim; convert the two-engine repo's drift-prone procedures (add an engine op, add a serve route, touch the line protocol) into skill checklists with the tripwire lint as first step.

## 2. vLLM v1 / SGLang: platform separation, invariants, perf gates

### 2.1 vLLM platform layer: one activated platform, resolved lazily, enforced at runtime

- **Mechanism.** `vllm/platforms/interface.py`: `Platform` base class with per-platform overrides — `get_attn_backend_cls()` returns a class qualname string, `get_supported_vit_attn_backends()` returns an `AttentionBackendEnum` list, `check_and_update_config()` mutates config per platform. `vllm/platforms/__init__.py`: `resolve_current_platform_cls_qualname()` probes builtin + plugin detection functions and **raises if more than one platform activates**; `current_platform` is lazily resolved after plugins load (explicit comment block). https://github.com/ggml-project/vllm equivalent: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/__init__.py , https://github.com/vllm-project/vllm/blob/main/vllm/platforms/interface.py
- **Plugin seams via entry-point groups.** `vllm/plugins/__init__.py` defines groups `vllm.general_plugins`, `vllm.platform_plugins`, plus `endpoint_plugins`, `io_processors`, `lora_resolvers`; out-of-tree packages register through Python entry points; vLLM's own `pyproject.toml` registers its builtins through the same channel — no second mechanism. https://github.com/vllm-project/vllm/blob/main/vllm/plugins/__init__.py
- **Drift class killed:** two platform codepaths both firing; out-of-tree forks to add a device.
- **Verdict: ADAPT (minimal form).** The valuable parts for us: (a) "exactly one engine variant activates, enforce with a raise"; (b) extension through one registration channel, never a second. The entry-point machinery itself is NOT-FIT — no out-of-tree consumers.

### 2.2 vLLM KV-cache spec registry: completeness check that raises in production

- **Mechanism.** `vllm/v1/kv_cache_spec_registry.py`: `@register_kv_cache_spec(manager_class=…, uniform_type_base_spec=…)` binds each frozen `KVCacheSpec` dataclass to its manager; `check_kv_cache_spec_registry()` walks every layer's spec at engine build and **raises ValueError when a spec has no registered manager** — code comment: "use raise instead of assert to make it effective in production environment". https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_spec_registry.py
- **Drift class killed:** add a cache/spec variant, forget the manager — caught at startup on the real box, not in a test you forgot to run.
- **Verdict: ADOPT-AS-IS.** Pair with finding 1.1: registries self-enumerate AND self-check. This kills the most common small-team drift: the variant that works in dev and crashes in a config nobody exercised.

### 2.3 vLLM v1 test architecture: invariant-named test files + sampler parity

- **Mechanism.** `tests/v1/core/` names tests after the invariant they pin: `test_deferred_block_free.py`, `test_priority_preemption_bug.py`, `test_swa_inflight_window_free.py`, `test_prefix_replay.py`, `test_repetition_detection.py` — a regression file per killed bug. `tests/v1/sample/test_rejection_sampler.py` compares GPU rejection sampling against constructed reference probability distributions with a statistical-distance bound (lines ~407-429). https://github.com/vllm-project/vllm/tree/main/tests/v1/core , https://github.com/vllm-project/vllm/blob/main/tests/v1/sample/test_rejection_sampler.py
- **Drift class killed:** scheduler/sampler regressions reappearing; stochastic-numeric drift in sampled paths.
- **Verdict: ADAPT.** Name test files after the invariant; keep the serial-round engine's sampler under a reference-distribution comparison like the rejection-sampler test.

### 2.4 ABI audit with a must-shrink allowlist

- **Mechanism.** `.buildkite/check-torch-abi.py`: audits every compiled extension for PyTorch **stable-ABI** compliance via `torch_abi_audit`; carries `ALLOWED_UNSTABLE_LIBRARIES` (documented "shrink and remove over time") and `_find_stale_allowlist_entries()` — CI fails if an allowlist entry no longer matches anything. https://github.com/vllm-project/vllm/blob/main/.buildkite/check-torch-abi.py
- **Drift class killed:** silent ABI coupling to a moving dependency; allowlist rot.
- **Verdict: ADAPT.** Same pattern for our engine ABI surface (exported symbols, struct layouts): audit + allowlist that CI forces to shrink. Cheap, pure-Python.

### 2.5 Perf gates: continuous dashboard + local 1-hour suite

- **Mechanism.** `.buildkite/performance-benchmarks/` — developer-runnable suite (latency/throughput/fixed-QPS serving) constrained to ~1 hr, JSON-configured cases (`tests/{latency,serving,throughput}-tests.json`); continuous tracking at perf.vllm.ai under PyTorch CI HUD, manually triggerable per PR via `run-performance-benchmarks.sh`. https://github.com/vllm-project/vllm/tree/main/.buildkite/performance-benchmarks
- **SGLang counterpart:** nightly NVIDIA workflow runs benchmarks and a "Consolidate performance metrics from all jobs" step; `ci-auto-bisect.yml` auto-bisects scheduled test failures to find the guilty commit (timeout 30 min, workflow_run-triggered). https://github.com/sgl-project/sglang/blob/main/.github/workflows/nightly-test-nvidia.yml , https://github.com/sgl-project/sglang/blob/main/.github/workflows/ci-auto-bisect.yml
- **Drift class killed:** perf drift discovered weeks later, no bisectable baseline.
- **Verdict: ADAPT.** One box: nightly bench run that consolidates metrics into a committed artifact (log/jsonl) + compare-against-last-N gate in the bench fleet. Auto-bisect is overkill until tests are flaky; the consolidation artifact is not.

## 3. Typed contracts in inference systems — concrete formats

| System | Format | Mechanism (verified) | Take for us |
|---|---|---|---|
| ggml RPC | length-prefixed binary + semver | hello handshake, major-equal + minor-ceiling (`ggml-rpc.cpp:435`, version constants in `ggml-rpc.h:9-11`) | ADOPT — it is our line protocol's missing preamble |
| ONNX Runtime opsets | per-domain version ceiling tables | `ValidateOpsetForDomain()` in `onnxruntime/core/graph/model_load_utils.h`: each op domain carries a map of first-released opset; models stamped above the ceiling throw or warn by flag | ADAPT — encode max-supported-format-version per owned format; refuse newer, warn on draft |
| TRT-LLM executor | typed request union | `cpp/include/tensorrt_llm/executor/executor.h`: `RequestType` enum (`REQUEST_TYPE_CONTEXT_AND_GENERATION`, …), typed getters/setters, background-error shutdown contract in Python `tensorrt_llm/executor/executor.py` (`_handle_background_error`, `get_kv_cache_capacity()`) | ADAPT — typed request/response union for the control plane instead of loose dicts |
| OpenVINO plugins | XML manifest registry | `src/inference/src/dev/core_impl.cpp`: `m_plugin_registry` device→descriptor; `plugins.xml` stores plugin location "resolved next to itself" (comment at lines 127-128); devices resolved through registry lookups, not dlopen-by-convention | NOT-FIT as format; ADAPT as concept — declare each engine binary's capabilities in a file next to the binary, parsed by the serve layer |
| safetensors | 8-byte u64 header length + JSON header | parser invariant: header size bounds all metadata; single fuzz target `SafeTensors::deserialize` | ADAPT — length-prefixed framed records with strict bounds reads |

URLs: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/model_load_utils.h , https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/executor/executor.h , https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/executor/executor.py , https://github.com/openvinotoolkit/openvino/blob/master/src/inference/src/dev/core_impl.cpp , https://github.com/huggingface/safetensors/blob/main/safetensors/fuzz/fuzz_targets/fuzz_target_1.rs

**Schema-evolution policy observed across all four:** major-equal/minor-ceiling (ggml), ceiling-table (ORT), additive-enum-only (TRT-LLM), self-describing-JSON-header (safetensors). None attempts bidirectional compat. One-direction refusal beats negotiation.

## 4. Executable invariants — state of the art beyond static_assert

Ranked by what transfers to a 12 GB GPU + 184 GB RAM box:

1. **Differential op parity, enumerated from the registry** (llama.cpp `test-backend-ops`, §1.2). The core invariant: identical op graph, N device implementations, one reference. ADOPT-AS-IS.
2. **Whole-model NMSE gate per declared arch** (`test-llama-archs`, 1e-4, §1.3). ADOPT-AS-IS.
3. **Statistical sampler parity vs reference distributions** (vLLM rejection-sampler test, §2.3). ADOPT for the DS engine's sampling path.
4. **Production-time completeness checks that raise** (vLLM `check_kv_cache_spec_registry`, §2.2). ADOPT-AS-IS — costs ~20 lines, kills a whole drift class.
5. **Golden-hash/snapshot gates in CI** (pre-tokenizer hashes, GGUF metadata snapshots, §1.6). ADOPT-AS-IS.
6. **ABI audit with shrink-only allowlist** (vLLM torch-abi, §2.4). ADAPT.
7. **Parser fuzzing** (safetensors cargo-fuzz; **not** OSS-Fuzz — see §0). ADAPT: one libfuzzer/AFL target over the line-protocol and owned-format parsers; run it nightly, not in pre-commit. The negative finding matters: even the elite run fuzzing outside CI gates.
8. **Auto-generated capability matrix as docs** (`docs/ops/*.csv`, §1.2). ADAPT — regenerate per-engine op/route support tables in CI.

## 5. Variant management without drift ("one spine, declared branches")

- **The spine must enumerate its own branches.** Verified three ways: llama.cpp registry → test harness iterates devices (§1.2); vLLM KV spec decorator → startup completeness check (§2.2); vLLM platform resolution → runtime raise on double-activation (§2.1). Manual switch tables are the pre-refactor state everywhere.
- **Contrast case (anti-pattern in the wild):** SGLang's `python/sglang/srt/layers/attention/attention_registry.py` is a large hand-maintained if/elif dispatch selecting backend classes per hardware — the exact drift surface vLLM and llama.cpp engineered away. https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- **Deliberately-different pipelines under one interface:** llama.cpp `llama-memory.h` family (KV-cache / recurrent / hybrid / iswa contexts, §1.4) — the declared-isolation pattern done right: shared lifecycle contract (prepare/compute/rewind/status-combine helpers), fully independent state layout per branch. https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory.h
- **C++ trait/typeclass mechanism upstream:** backend self-registration via exported `_reg()` factories + `ggml_backend_reg_get_proc_address` for optional capabilities (seen used for `ggml_backend_set_n_threads`, `ggml_backend_cpu_set_use_ref` in the test harness) — C-level traits without RTTI or C++ templates across the boundary. https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp
- **Verdict:** ADAPT. Two engines register through one registry interface; capability gaps surface as failed probes (`supports_op`-style), never as compile-time `#ifdef` forests in shared code. Feature flags: env-disable at registration (the `GGML_DISABLE_VULKAN` pattern) — a flag that only *removes* a registered variant cannot silently fork behavior; flags that *add* behavior are the ones that drift.

## 6. Build/test for agent+human small teams

- **Declared-cost test registries.** SGLang: every test file registers `est_time, suite, nightly, disabled` (`python/sglang/test/ci/ci_register.py`, `_PARAM_ORDER`); `test/run_suite.py` reads the registry, maps suites to `HWBackend` (cpu/cuda/amd/musa/npu/xpu/mlx/mps/ppu), and `auto_partition()` splits suites across runners. One operator, full matrix. https://github.com/sgl-project/sglang/blob/main/python/sglang/test/ci/ci_register.py , https://github.com/sgl-project/sglang/blob/main/test/run_suite.py
- **Per-vendor CI isolation** (llama.cpp one workflow per backend + self-hosted GPU runners, §1.1). For our box: one fast CPU-only parity lane in pre-commit range; one nightly GPU lane.
- **Golden outputs as CI gates** (§1.6) — cheapest possible perf/behavior baseline for a repo without a fleet.
- **Ownership policy for agent-produced code** (§1.7) — upstream words are the right contract for our team: agents may write; a human must be able to defend every line.
- **Verdict: ADAPT.** Order of implementation: (1) declared-cost registry over the existing bench scripts, (2) fast CPU parity lane gated pre-push, (3) nightly GPU lane writing one consolidated metrics artifact, (4) skills/ checklists for the three drift-prone procedures.

## Ranking — structural value per implementation cost

| # | Mechanism | Source | Value | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | Registry-enumerated differential op parity + CI-regenerated support CSV | llama.cpp test-backend-ops + docs/ops | kills op-level cross-engine drift permanently | days | ADOPT-AS-IS |
| 2 | Semver hello handshake on line protocol | ggml-rpc | kills serve↔engine wire drift | hours | ADOPT-AS-IS |
| 3 | Whole-model NMSE parity per declared engine variant | llama.cpp test-llama-archs | kills pipeline-level numeric drift | 1-2 days | ADOPT-AS-IS |
| 4 | Self-checking registries (decorator + raise-in-production) | vLLM kv_cache_spec_registry | kills forgot-to-register variant drift | ~20 lines | ADOPT-AS-IS |
| 5 | Golden-hash/snapshot CI gates on owned formats | llama.cpp pre-tokenizer-hashes, snapshots/*.schema | kills silent semantic format drift | hours | ADOPT-AS-IS |
| 6 | One interface, per-variant context classes (declared isolation) | llama.cpp llama-memory.h | keeps DS vs Qwen pipelines composable | week-level | ADAPT |
| 7 | Device abstraction with supports_op/offload_op probes | ggml/include/ggml-backend.h | capability negotiation without version strings | days | ADAPT |
| 8 | Declared-cost test registry + suite auto-partition | sglang ci_register.py + run_suite.py | bench fleet of one becomes schedulable | 1-2 days | ADAPT |
| 9 | ABI audit with shrink-only allowlist + stale-entry detection | vLLM check-torch-abi.py | kills silent ABI coupling + allowlist rot | day | ADAPT |
| 10 | Consolidated nightly perf metrics artifact + auto-bisect | sglang nightly-test-nvidia, ci-auto-bisect | perf drift bisectable | day | ADAPT |
| 11 | Statistical sampler parity vs reference distributions | vLLM test_rejection_sampler | kills stochastic-path drift | day | ADAPT |
| 12 | Agent skills encoding repo procedures | llama.cpp skills/ | agent-passable drift checklists | hours | ADAPT |
| 13 | One-activation platform resolution with runtime raise | vLLM platforms/__init__.py | kills double-active variant | small | ADAPT |
| 14 | Entry-point plugin groups | vllm/plugins/__init__.py | extension seam for out-of-tree consumers | medium | NOT-FIT (no out-of-tree consumers yet) |
| 15 | Hand-maintained if/elif backend dispatch | sglang attention_registry.py | — | — | NOT-FIT (the anti-pattern; do not copy) |
| 16 | OSS-Fuzz on model parsers | (negative results, §0) | — | — | NOT-FIT as-is; run one in-tree fuzz target nightly instead |

## Source index

- llama.cpp @ 9e2a04a4: https://github.com/ggml-org/llama.cpp — files: `ggml/src/ggml-backend-reg.cpp`, `ggml/src/ggml-backend-dl.cpp`, `ggml/include/ggml-backend.h`, `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tests/test-backend-ops.cpp`, `tests/test-llama-archs.cpp`, `src/llama-arch.cpp`, `src/llama-memory.h`, `docs/ops/`, `.github/workflows/pre-tokenizer-hashes.yml`, `.github/workflows/update-ops-docs.yml`, `tests/snapshots/`, `tools/server/tests/`, `AGENTS.md`, `skills/`
- vLLM @ 42c043c6: https://github.com/vllm-project/vllm — files: `vllm/platforms/__init__.py`, `vllm/platforms/interface.py`, `vllm/plugins/__init__.py`, `vllm/v1/kv_cache_spec_registry.py`, `vllm/v1/kv_cache_interface.py`, `tests/v1/core/`, `tests/v1/sample/test_rejection_sampler.py`, `.buildkite/check-torch-abi.py`, `.buildkite/performance-benchmarks/`; V1 architecture post: https://blog.vllm.ai/2025-01-27-v1-alpha-release.html
- SGLang @ b702f0d8: https://github.com/sgl-project/sglang — files: `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/test/ci/ci_register.py`, `test/run_suite.py`, `.github/workflows/ci-auto-bisect.yml`, `.github/workflows/nightly-test-nvidia.yml`
- ONNX Runtime @ f340e685: https://github.com/microsoft/onnxruntime — `onnxruntime/core/graph/model_load_utils.h` (`ValidateOpsetForDomain`)
- TensorRT-LLM (raw fetch): https://github.com/NVIDIA/TensorRT-LLM — `cpp/include/tensorrt_llm/executor/executor.h`, `tensorrt_llm/executor/executor.py`
- OpenVINO (raw fetch): https://github.com/openvinotoolkit/openvino — `src/inference/src/dev/core_impl.cpp` (plugin registry, plugins.xml)
- safetensors @ 2f216f9a: https://github.com/huggingface/safetensors — `safetensors/fuzz/fuzz_targets/fuzz_target_1.rs`
- OSS-Fuzz: https://github.com/google/oss-fuzz — `projects/llama.cpp` and `projects/safetensors` both absent (404, 2026-11-04)
