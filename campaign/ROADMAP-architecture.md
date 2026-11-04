# The Architecture Elevation Program (binding direction, 2026-11-04)

The diagnosis in one sentence: correctness lives in convention and duplicated code — two engine
trees that can drift, a text protocol with no schema, formats parsed wherever they are met, laws
enforced by whoever remembers them. The elevation moves every one of those into structure:
shared substrate, typed contracts, owned formats, executable invariants. After that,
hand-rolling is not forbidden — it is unreachable.

Migration law: **strangler-fig, never a rewrite.** Every step lands with the banks and floors
held (Qwen 232/222/315, DS 65/62/70) and a before/after proof specific to the hazard it
removes. At no step is the system worse than the moment before it. Owner paths and exit
criteria per item — an item without both is an aspiration and does not belong here.

Progress markers: [x] landed 2026-11-04 (`cc05319f`, `c216f423`, `d08c2d11`).

---

## Layer 1 — The substrate (both engines link it, neither owns it)

One library of engine-agnostic capability in `include/strata/` + `src/core/`: GGUF I/O
(`strata::GgufFile`), pack-manifest I/O, host-memory sampling (`sample_host_memory`), the error
contract (one CK family, err-collect, never `exit()` in hosted code), CUDA device/stream/graph
utilities, KV slot I/O, the control vector as a subsystem. The dsv41 target already carries the
include path (`CMakeLists.txt:2283`).

**Declared out of the substrate** (isolation by design, not drift): `src/dsv41/vendor/` (the
sampler and weights stack, `CMakeLists.txt:2262` "no symbol shared"), and the two engines'
model-specific pipelines — Qwen's doorbell/graph machinery versus DS's serial rounds are
different design points. Isolations get a parity test and a one-line declaration here, never a
merge.

| Step | Owner path | Exit criteria |
|---|---|---|
| [x] cvec parse onto `GgufFile` | `src/dsv41/ds4_model.cu` `cvec_load` | standing file parses identically (base 1568 both ways); an alignment-64 vector with header_end=1601 garbles before, parses after; DS lanes same-hour within noise (`d08c2d11`) |
| [ ] `ram_avail` onto `sample_host_memory` | `src/dsv41/ds4_model.cu:176` → `src/core/conversation_memory.cpp:258` | DS guard honors `--memory-limit-mib`; the cgroup-v2 walk is one implementation; boot + lanes before/after |
| [ ] CK macro family unified | `src/qwen35/q36_engine.cu:67` (`exit(1)` in hosted code) vs `src/dsv41/ds4_model.cu:74` (err-return) | hosted code paths can never `exit()`; a fault becomes a refused request, not a dead server; build + parity |
| [ ] pack-manifest I/O into the substrate | a `strata::artifact` pack reader | `tensors.txt`/`ds4.txt` parsed in exactly one place; the dsv41 loader and the prewarm derivation import it |

## Layer 2 — The control plane (a typed contract, not line-regex folklore)

The stdout line protocol becomes a versioned handshake: at boot the engine reports structured
identity — version, context, capabilities, and **armed geometry** (cvec mode/range/layers,
prewarm state, tier sizes). The serve layer validates it against one schema module both sides
share (`serve/` + a header both trees include), and refuses or warns on mismatch.

The byte-identical boot-line ritual performed by hand becomes the server's job on every boot: a
wrong layer range or a dead parser is a **refused boot, not a silent no-op**. (Precedent: the
4..24 inert-vector bug — structurally impossible instead of a war story. The inert
`engine_ready_s` key was the same class, fixed 2026-11-04.)

| Step | Owner path | Exit criteria |
|---|---|---|
| [ ] schema module + engine `IDENT` emission | `serve/handshake.py` (new) + both engines' serve banners | the server parses the handshake or refuses with the reason; old engines degrade to the line protocol with one warning |
| [ ] armed-geometry validation | same | negative test: an injected mismatched cvec range refuses the boot; the standing configs boot clean |
| [ ] prewarm state joins the handshake | `serve/server.py` `Prewarm` | /health's word derives from the engine's own report, not from the server's inference |

## Layer 3 — Owned formats (one owner module and one validation gate per format)

Every load-bearing format gets exactly one owner: the pack manifest through its library
(`vision_pack.py` and every future tool goes through it, never through text edits); GGUF read
only by `GgufFile` (C++) and `tools/gguf_reader.py` (Python). **A tool that edits a load-bearing
format textually is the defect this layer exists to make impossible.**

| Step | Owner path | Exit criteria |
|---|---|---|
| [ ] Python GGUF readers fold | `tools/quantscope.py:170`, `tools/dsv41/vision_pack.py:65` → `tools/gguf_reader.py` | one GGUF implementation in Python; quantscope and vision_pack import it; their outputs byte-identical on the standing files |
| [ ] pack-manifest owner library | the substrate reader (Layer 1) exported to tools | `vision_pack` composes manifest entries through it; a hand-written tensors.txt line anywhere is a lint hit (Layer 4) |

## Layer 4 — Executable invariants (the laws leave the markdown)

Static asserts tie every cross-file literal to its owner constant — the window-width wall of
2026-11-04 generalized. Parity tests pin every declared isolation. The handshake validates
armed geometry. A tripwire lint greps the tree for re-implementation signatures: GGUF magic
outside its owners, kill patterns outside `campaign/battery-lib.sh`, `exit(` in hosted engine
code, `prewarm2`-class references to retired artifacts. The bench laws stay as instruments;
the invariant classes become compile errors, test failures, or refused boots. **An agent who
forgets the doctrine is stopped by the tree, not by memory.**

| Step | Owner path | Exit criteria |
|---|---|---|
| [x] window-width wall | `verify.hpp`, `mtp.hpp`, `verify_kernels.hpp`, `native_mmvq.cu`, `cpu/expert.hpp`, `generate.cpp` | a scratch `kVerifyMaxT=9` build fails with the message (`c216f423`); negative-proven, lanes in band |
| [ ] tripwire lint | `tools/lint_tree.py` (new, runs in CI/pre-commit) | flags GGUF magic, kill patterns, hosted `exit(`, retired-artifact references; zero standing hits |
| [ ] parity tests for declared isolations | `src/kernels/*_parity.cpp` pattern extended | every isolation declared in Layer 1 has a test pinning its behavioral boundary |

---

## The law this program adds to AGENTS.md

- **A new consumer imports the owner; a second implementation is the defect.**
- **Declared isolation gets a parity test; undeclared duplication gets deleted.**
- **Every migration step carries a hazard-specific before/after proof and holds the floors.**
- **Isolation is architecture when declared and tested; it is drift when silent.**
