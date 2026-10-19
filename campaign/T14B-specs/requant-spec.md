# T17F #1 — PACK RE-QUANT: EXECUTION SPEC (routing/imatrix-weighted chooser over the IQ3_S pack)

Written 2026-10-19. Status: SPEC, nothing executed. This is the hard precondition of the T14B
harvest (harvest-spec.md §8 item 1: "PACK RE-QUANT FIRST — the sequencing law"; harvest-spec.md
§2 stamps `manifest.json` with the pack sha256s, so the harvest artifact names its pack).
Read-only inputs everywhere: the current pack and shards are NEVER written; the re-quant writes a
NEW pack dir + NEW shards and the config swaps by path. Rollback is a config edit (§7).

Box: RTX 5090 32 GB sm_120 (96 MB L2), 9950X3D + 184 GB DDR5, 705 GB free on /home (ext4,
`df` this session). Standing anchors (SVG-CUSTOM.md:7998-8013): M1 198.5 / pair 197.6 / conc 312.4
five-boot means, M3b 74.9%, MMLU-Pro 95.0, mimic 87.9. Quality floor law: MMLU 95.0 / mimic 87.9,
No-Regression absolute (SVG-CUSTOM-TODO.md:1235-1236).

---

## 0. Executive summary

- **The lever.** Re-choose the per-(layer, role) expert formats over the pack's EXISTING format
  menu, guided by our routing data (expert profiles) and imatrix-weighted requantization of only
  the recut tensors, targeting ~3.0-3.2 bpw effective on the routed experts (today: 3.331).
  Monetizes the two banked terms: the expert bytes at the VRAM roofline (T12-MEM, 87-92% of GDDR7
  peak on the hits line, SVG-CUSTOM-TODO.md:914-916) and the expert-cache slot count (9,337
  slots / 17.71 GiB = the VRAM ceiling; +512 slots = +2.1 pt hit, SVG-CUSTOM-TODO.md:912-913).
- **Recommended v1 recipe (RECIPE-α, "cold-12"):** recut exactly the 12 layers that are cold in
  BOTH the static and the learned profile to (gu IQ2_S, d Q2_0), COPY everything else. Experts
  46.84 → 42.97 GiB (**−8.3%**), effective 3.331 → **3.056 bpw**, avg blob 1.952 → 1.790 MiB,
  slots ≈ 9,337 → **≈10,129 (+792)**. 36 of 144 expert tensors requantized; every other tensor in
  the model byte-identical (`--tensor-type ...=COPY`).
- **Honest pricing, correcting T17F's envelope.** The +6-10% M1 in SVG-CUSTOM-TODO.md:1131-1136
  assumed unsaturated slot credit. At the standing 97.7% hit rate the slot term is nearly
  saturated (miss exposure 0.3-0.7 ms warm, T13 correction SVG-CUSTOM-TODO.md:973), so the honest
  decomposition is: bytes term −8.3% of a 3.26 ms hits line in a 14.2 ms window ≈ **+1.9-2.0%
  M1** direct; miss-pool/PCIe/staging terms scale with blob size too (+0.5-1%); conc gains more
  than M1 because its windows pull wider expert unions (F-183 direction, SVG-CUSTOM.md:5963).
  Realistic standing-recipe outcome: **+2-4% M1, +3-6% conc**, with the 262k-slot-risk absorption
  (§6.1) as a separate unpriced win. Outer bound: the same-quant 2x2 measured the pure quant
  effect at +45% M1 on the stock engine (Q2_0 205.9 vs IQ3_S 141.5, F-234a SVG-CUSTOM.md:8052) —
  our standing recipe already banks most of the miss-regime that made that number big.
  ADOPT BAR for a single-boot read stays the campaign +4%/-4% law; treat +2-4% as a compounder,
  not a milestone.
- **Tool gap, answered (T17F open question 3).** The repo has NO quantizer of its own, but the
  pinned fork `third_party/llama.cpp` (v0.1.44.1-126-gadf5f25f checkout of the strata tree) has
  arch `qwen4exp` end to end: `src/llama-arch.cpp:43`, `src/models/qwen4exp.cpp`,
  `tools/quantize` with `--allow-requantize --imatrix --tensor-type-file --keep-split`, and
  `tools/imatrix`. The fork's quantizer knows the pack's type 42 ("Q2_0 ... 2.25 bpw quantization
  (group 64)", tools/quantize/quantize.cpp:36; `GGML_TYPE_Q2_0 = 42`, ggml/include/ggml.h:432).
  No new quantizer script is needed; two thin driver scripts (chooser + tensor-type-file writer)
  are specified in §4.3.
- **Source decision (the spec's main judgment call).** No BF16/Q8_0 source exists on disk (only
  the IQ3_S 84 GB and Q2_0 66 GB family members; searched /home, /mnt/HQ, /mnt/STORE,
  /mnt/SCRATCH). True requant from FP16 needs a 188-372 GB download and exits the 1-2 CPU-day
  budget. v1 therefore requantizes FROM the existing IQ3_S shards (dequant → requant, one extra
  generation of error, imatrix-weighted, confined to 36 cold-layer tensors), with the family-graft
  path (ISTA-DASLab's own first-generation low tensors) as the quality fallback and the BF16
  download as the out-of-budget escalation (§4.6).
- **Falsifier is one boot** and costs no battery: build the v1 pack, boot it, run the G2 logit
  canary (quant_quality.py vs the current pack as reference) + one probe-lane A/B (M1/pair/conc/
  M3b vs a same-night ctl boot). The recut is confined to cold layers, so the canary is expected
  to pass; a fail costs hours, not weeks.

---

## 1. What the pack is, and what the loader accepts (code-verified)

Artifacts today:
- Shards: `/home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`
  (54.82 + 28.80 GB; arch `qwen4exp`, `general.file_type 26`, tokenizer embedded,
  `qwen4exp.expert_count 512`, `expert_feed_forward_length 640` — GGUF header dump this session).
- Pack dir: `/home/svg/LLM/Custom-data/packs/iq3_s/` — `native_experts.txt` (v3, 48 rows),
  `index.txt` (1,079 tensors), `dense.bin` (1.47 GiB), `conversions.json` (**zero** converted
  tensors — the GSQ-RCO file already stores every small projection in the engine's form),
  `tokenizer/`. No `experts.bin`: the engine reads experts straight from the GGUF shards.
- Launch: `--pack .../packs/iq3_s --native <shard1> --ple-gguf <shard2>` (boot line,
  custom-iq3s-adaptmtp.log:5; standing config `custom-iq3_s-8081.json`).

**Mixing granularity — per (layer, role), not per expert.** `native_experts.txt` v3 columns are
`layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]`
(Tools/iq_pack.py:474-477 writes it; src/kernels/cpu/expert_layout.cpp:430-480 parses it).
`gu_type` covers gate AND up (they must share type — iq_pack.py:452 refuses otherwise); `d_type`
is down. One blob per expert = `[gate rows | up rows | down rows]` with per-layer STRIDE
`blob_bytes` (expert i of layer l at `offset[l] + i*blob_bytes[l]`; contiguity enforced,
expert_layout.cpp:481-488). Per-expert formats are impossible without a format change; the
decision variable is 96 cells: 48 layers x {gu, d}.

**Blob-size law (exact, verified against all 7 combos in the pack).** `native_fmt()`
(src/kernels/cpu/native_expert.cpp:50-93): `gu_row = row_size(gu, 2560)`, `up_off = gu_row*640`,
`down_off = 2*up_off`, `bytes = down_off + d_row*2560`. In bits-per-element this is
`blob = 409600*bpw_gu + 204800*bpw_d`, and the loader REFUSES a row whose `blob_bytes` column
disagrees with the recomputed value (expert_layout.cpp:439-442). Menu (all already shipped in
this pack — i.e. kernel-proven on both CPU pool and GPU decode):

| type id | format | bpw/elem | role use today |
|---|---|---|---|
| 22 | IQ2_S | 2.5625 | gu (20 layers) |
| 18 | IQ3_XXS | 3.0625 | gu (17) |
| 21 | IQ3_S | 3.4375 | gu (10) |
| 23 | IQ4_XS | 4.25 | gu (L47 only) |
| 42 | Q2_0 (GSQ-RCO, group 64) | 2.25 | d (9 layers) |
| 20 | IQ4_NL | 4.5 | d (39 layers) |

Current mix (counted from native_experts.txt this session): (18,20) x14, (22,20) x15, (21,20) x9,
(18,42) x3, (22,42) x5, (21,42) x1, (23,20) x1 → total 50,292,326,400 B = 46.84 GiB of experts,
avg blob 2,046,400 B, effective **3.3307 bpw** (matches data/gguf_fingerprints.json `qwen/IQ3_S`:
experts_bpw 3.331, total 83,606,638,080 B).

**Legality = ggml-cpu has a dot product.** `native_fmt` requires `vec_dot` + `from_float` for the
type and its activation type (native_expert.cpp:52-63). Kernel coverage: AVX-512 multi-token
`iq512_supported = {16,17,18,21,22}`; AVX2 adds 23 (`iq256_supported`, native_expert.cpp:40-46);
K-quant down {12,7,8} legal via kq256 (native_expert.cpp:48; docs/QUANTS.md:35-41); type 42 runs
the fork's Q2_0 kernels (src/kernels/cpu/expert_layout.cpp:355-368). **v1 stays inside
{16,17,18,21,22,23,42,20} so no kernel question can arise.**

**The engine validates a pack before it trusts it — this is what a new pack must satisfy.**
`check_experts_gguf` (src/core/expert_source.cpp:4340-4391), run on every boot before the arena
allocation (expert_source.cpp:4591), requires for EVERY (layer, role):
1. tensor `blk.N.ffn_{gate,up,down}_exps.weight` exists in the named shard;
2. its GGUF type == the type in native_experts.txt (expert_source.cpp:4358, :4366);
3. shape `[n_embd, n_ff, 512]` for gate/up, `[n_ff, n_embd, 512]` for down (:4360-4361);
4. per-expert payload == `per[3] = {up_off, up_off, blob - down_off}` (:4351, :4372);
5. the tensor's absolute data start == the `gate_off/up_off/down_off` columns (:4374-4376).
So a re-quant CANNOT rewrite tensors in place (offsets shift); it must write new shards AND a
regenerated native_experts.txt. `tools/iq_pack.py` does exactly that derivation from a GGUF's
headers (iq_pack.py:430-488), which is why §4 uses it as the assembler. It refuses packs whose
small projections are not in the engine's FORM (iq_pack.py:73-86) — irrelevant here: the dense
side is COPYed byte-identical, and conversions.json is already empty.

**Cache-slot law (the yield term).** `--expert-cache auto` fills slots in expert-profile rank
order until VRAM `room` is exhausted; for a native pack each slot is charged
`ceil256(blob_bytes(layer))` and the cache opens per-slot-sized (generate.cpp:5658-5672,
`cache.open_sized`). The boot prints the banked ledger line:
`expert cache 9337 slots, 17.71 GiB of VRAM` (custom-iq3_s.log:35; auto line :34). Slots scale
inversely with the mean blob along the ranking (which is layer-uniform: 138-240 of 512 pairs per
layer inside the top 9,337, computed from the profile this session):
`slots_new ≈ 9,337 x (2,046,400 / mean_blob_new)` — the ±45-slot slack of this model vs the
banked 9,337 (model gives 9,292) is boot-to-boot free-VRAM variance, not model error.

---

## 2. Sensitivity model (what data we have, and the fallback)

### 2.1 What the profile files actually contain (writer code read)

Format (writer src/core/expert_cache.cpp:170-209; reader expert_cache.cpp:72-139; parser
tools/make_profile.py:33-44): magic `STRP`, then five u32 `{version=1, n_layers, n_expert,
n_ranked, n_ranked}`, then `n_ranked` (u16 layer, u16 expert) pairs **in rank order** (most-routed
first), then an int32 rank-per-pair table. Size 24 + 24,576x4 + 24,576x4 = 196,632 B — matches all
three 512-expert profiles on disk. **A profile is an ORDINal ranking of all 24,576 (layer, expert)
pairs; it does NOT persist per-expert token counts** (the adaptive tier's `heat` counts order the
sort at save time, expert_cache.cpp:141-168, and are then dropped).

- `data/expert-profile.bin` — the shipped static prior (in use: the launcher's
  `--expert-profile data/expert-profile.bin`, boot line custom-iq3s-adaptmtp.log:5).
- `data/expert-profile-learned.bin` — what the adaptive tier learned serving this box
  (`--expert-profile-save ... --expert-profile-save-every 30` = every 30 MINUTES, not requests —
  generate.cpp:954-957; T17B's "every 30 req" at SVG-CUSTOM-TODO.md:1100-1101 is a misquote).
  Fresh copy banked 2026-10-18 (data/, ls this session); campaign/t17-lp-profile.bin is the
  t17-leg copy.
- `data/expert-profile-coder.bin` — **48x256 geometry (12,288 pairs): belongs to the pruned Coder
  variant; `read_expert_profile` refuses it for this model (ne mismatch, expert_cache.cpp:104-112).
  NOT an input. Rejected here.**

Raw per-pair counts CAN be regenerated any time with one boot of `--dump-routing` (the trace
format make_profile.py:46-58 parses: per record layer + k routed ids + k weights). That is the
fallback if magnitudes are ever needed; the chooser below works on ranks only.

### 2.2 The model

Three signals, in priority order:

1. **Routing mass (ours, measured).** Layer sensitivity `S(l)` = fraction of the layer's 512
   pairs inside the top-9,337 of the learned profile. Cross-checked static vs learned: the 12
   coldest layers AGREE in 10 of 12 positions (static coldest: 15,3,43,31,46,40,45,39,44,16,41,47;
   learned: 15,3,43,31,46,40,45,39,44,47,23,16 — computed this session). Spread is real: resident
   share per layer runs 138-240/512 (27-47%) vs 194.5 mean.
2. **Role prior (external, corroborated).** down_proj carries the sensitivity (MxMoE, T17F dossier
   S10, campaign/research/T17F-training-quant-context.md:100 + :151); the pack's own allocation
   already pays down more bits than gate/up (39 layers at IQ4_NL 4.5 vs gu at 2.5-3.4).
3. **Coherence guard (GEMQ's hazard).** Routing drift is the failure mode (GEMQ: >40% reroute at
   1.5-bit; calibration-domain hazard, T17F:56-58). Two consequences: (a) never recut a layer that
   is hot in EITHER profile (protect the union of the two resident sets, 16,558 distinct pairs —
   computed this session; note the pair-level union protects the hot experts themselves, the
   band rule below protects whole layers); (b) the calibration corpus for the imatrix must match
   the serving mix (GEMQ's GSM8K-collapsed-under-C4 lesson) → repo prose (the F-236-promoted,
   content-matched corpus harvest-spec §3.1 assembles; it already exists as depth-lane-real.py's
   pool: 45 files, 675k tokens) + optionally the /mnt/HQ md slice at <= 25% exactly as the harvest
   diversifier is gated (harvest-spec.md:106).

**Fallback if profiles were missing/unreadable:** derive bands from a single fresh
`--dump-routing` boot over the probe corpus (one boot, the T17B fast-warm class), or fall back to
the layer-band prior (MxMoE: first/last layers + down most sensitive) and halve the recut set.
Never needed for v1: both 512-expert profiles are on disk and agree.

---

## 3. Chooser rubric, budget arithmetic, slot yield

### 3.1 Rubric (v1, RECIPE-α)

Decision per (layer, role), evaluated top-down:

- **R0 protect:** layer in hot-12 of EITHER 512-expert profile → keep both roles at current types
  (COPY). Hot-12 today: 0,1,2,6,7,13,18,21,22,29,34,38 (learned) — static agrees on 11.
- **R1 cold band:** the 12 layers coldest in the learned profile: **3, 15, 16, 23, 31, 39, 40,
  43, 44, 45, 46, 47** (static's cold-12 differs by exactly one member, 23 ↔ 41 — both profiles
  rank 23 in their coldest 13; every R1 layer is outside the hot half of BOTH) → set
  (gu IQ2_S, d Q2_0) — the cheapest legal cell.
- **R2 dial (OFF for v1):** mid-24 layers with gu above IQ3_XXS → gu IQ3_XXS (touches 8 layers:
  17, 25, 27, 28, 35 + 3 already-at-18 are no-ops). Turns on only if the quality gate passes with
  margin and more slots are wanted.
- **R3 never upgrade in v1:** up-tiering down (IQ4_NL → Q8_0 = 8.5 bpw) costs more bytes than the
  entire cold cut saves; the menu has no cheap upgrade cell. Recorded as a rejected branch, not an
  open question.
- Types stay inside the shipped kernel set (§1) so no kernel/CUDA-graph question can arise; the
  `native_gu_mt_min` grouping rule (native_expert.cpp:124-140) applies unchanged.

### 3.2 Worked example on real native_experts.txt rows

Rows 47, 40, 3 of the actual file:
```
47 23 20 48929177600 2662400 53891822080 54341015040 53418222080   (IQ4_XS gu / IQ4_NL d, blob 2,662,400)
40 21 20 40894464000 2329600 45334348544 45698549504 44861567744   (IQ3_S  gu / IQ4_NL d, blob 2,329,600)
 3 22 20  2896691200 1971200 4573970560 4846165120 4101189760     (IQ2_S  gu / IQ4_NL d, blob 1,971,200)
```
All three are cold-band layers (S: L47 0.330, L40 ~0.30, L3 0.285-0.291 vs 0.375 mean). Rubric R1
sends each to (22,42): blob `409600x2.5625 + 204800x2.25 = 1,049,600 + 460,800 = 1,510,400`.
- L47: 2,662,400 → 1,510,400 = **−1,152,000 B/expert** (−512 x 1.125 MB = −576 MB, the single
  biggest line; the pack's one IQ4_XS layer is cold — ISTA spent its most expensive gu on our
  coldest layer).
- L40: −819,200 B/expert; L3: −460,800 B/expert.
Full v1 delta table (computed this session, all 12 layers):
3,15,23,39: −460,800 each (down-only requant, gate/up COPY); 16,31,43,46: −665,600; 40,44,45:
−819,200; 47: −1,152,000.

### 3.3 Budget arithmetic (bpw) — verified by script against the real file

`blob(gu,d) = 409600*bpw_gu + 204800*bpw_d` (§1). Per-expert params 3 x 1,638,400 = 4,915,200.

| recipe | experts total | effective bpw | avg blob | Δ vs today |
|---|---|---|---|---|
| current pack | 46.84 GiB | 3.331 | 1.952 MiB | — |
| **RECIPE-α (v1: cold-12 → (22,42))** | **42.97 GiB (−4.15 GB)** | **3.056** | **1.790 MiB** | **−8.3%** |
| α + R2 dial (+mid-8 gu → IQ3_XXS) | 42.60 GiB | 3.030 | 1.783 MiB | −9.1% |
| RECIPE-β (quality-first: gu-only cuts, down untouched) | 45.6 GiB | ~3.24 | 1.90 MiB | −2.8% |
| (reference: full 12/24/12 band swap) | 41.60 GiB | 2.958 | 1.733 MiB | −13% — REJECTED for v1: cuts hot-layer types wholesale |

RECIPE-β is the pre-agreed retreat if the G2 canary flags the down cuts: it never touches a down
tensor, keeps MxMoE's role prior intact, and still clears the top of the 3.0-3.2 window.

### 3.4 Slot yield and its credit

Slots for the same free VRAM, calibrated model `slots = 17.71 GiB / mean_blob` (reads 9,292 for
today's pack vs the banked 9,337 — the 45-slot gap is boot-to-boot free-VRAM variance, so deltas
carry ±45 slack):
- v1: **≈10,129 slots (+792)**; with R2: ≈10,216 (+879); β: ≈9,590 (+253).
- Hit-rate credit at the T12-MEM slope (+512 slots = +2.1 pt, SVG-CUSTOM-TODO.md:913): +3.25 pt
  nominal — but the standing hit rate is 97.7% and saturates; the honest marginal credit is
  +1.5-2 pt at 131k and, critically, the ENTIRE T15B 262k risk (§6.1).
- What the credit is NOT: a repeat of F-183's +4.9% conc. That measurement added cache BYTES
  (reserve 2000→800 MiB, cache 19→20.3 GiB) and paid an M1 −6.9% overhead (SVG-CUSTOM.md:5954-
  5968). v1 adds zero cache bytes — the overhead term does not exist here; only the finer-blob
  effect remains.

### 3.5 Byte-term arithmetic (the honest M1 decomposition)

Standing window ≈14.2 ms; expert hits line 3.26 ms (2,495 hits x 2.05 MB measured, T12-MEM
SVG-CUSTOM-TODO.md:914-916 = 23% of the window at 87-92% of roofline). v1 cuts hit bytes 8.3% →
−0.27 ms/window → **+1.9-2.0% M1**; miss-side terms (pool 0.3-0.7 ms, PCIe share, staging) scale
with blob size → +0.5-1% more; prefill reads all pairs per 32k chunk (T12-MEM item 7) so prompt
throughput gains a proportional slice of its expert-traffic share. conc/pair gain more than M1
(their windows pull wider unions; F-183 direction). Net expectation band: **+2-4% M1, +3-6% conc**
— inside T17F's +6-10% envelope, below its center; bank the decomposition, not the envelope.

---

## 4. Tooling runbook (exact commands; gaps and fallbacks named)

### 4.0 Workspace

```
RQ=/home/svg/LLM/Custom-data/requant            # new: all artifacts, ~165 GB transient peak
RQ_PACK=/home/svg/LLM/Custom-data/packs/iq3_s_rq1
```
Everything under /home: 705 GB free; peak usage ≈ old shards (read) + merged intermediate (83.7,
only if 4.5 needs it) + new shards (~80) + new pack (1.5) ≈ 165 GB. ≥ 500 GB stays free — the
harvest's 163.8 GB (harvest-spec §6) still fits afterwards. Nothing under /home/svg/LLM/
strata-custom-2 is written except campaign configs and results.

### 4.1 Build the fork's tools (CPU-only; the one real build)

The fork has never been built here (no binaries found; only Strata's own engine is in build/).
```
cd /home/svg/LLM/strata-custom-2/third_party/llama.cpp
cmake -B build-rq -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build-rq --target llama-quantize llama-imatrix gguf-split -j32
```
~5-15 min. (A CUDA build would speed 4.2 but costs a 20-40 min compile; CPU-only fits the budget,
see §8.) Verify: `build-rq/bin/llama-quantize --help` shows `--allow-requantize`,
`--tensor-type-file`, `--keep-split` (all present in tools/quantize/quantize.cpp:120-146).

### 4.2 imatrix from OUR serving mix

Corpus: the SAME seed pool the harvest uses (harvest-spec §3.1: repo prose 2.21 MB ≈ 675k tokens;
depth-lane-real.py:30-47 concatenation format) + optional /mnt/HQ md slice ≤ 25% (harvest-spec
:106). NOT wiki/C4-only (GEMQ hazard; T17F:58 "our calibration corpus should be OUR serving mix").
 llama-imatrix default chunk 512 tokens → ~1,300-2,600 chunks from ~675k-1.35M tokens. Two
bridge scripts are needed (both trivial, ~40 lines, tools/ requant_corpus.py + requant_chooser.py
— §4.3):
```
python tools/requant_corpus.py --out $RQ/calib.txt          # emits the raw corpus text
build-rq/bin/llama-imatrix -m $RQ/merged.gguf -f $RQ/calib.txt \
    -o $RQ/imatrix.gguf --chunks 2600 -c 512 -t 32 --no-ppl
```
GATE before trusting it: the imatrix file must contain entries for all 144 expert tensors
(48 x ffn_{gate,up,down}_exps.weight). This exercises the fork's qwen4exp forward pass
(src/models/qwen4exp.cpp); if an op is unimplemented or the entries are missing, STOP — go to
§4.6 Path-1 (family graft), do not requantize unweighted.

### 4.3 Chooser + tensor-type-file (the ~80 lines of new tooling, no engine code)

`tools/requant_chooser.py`:
1. Read `data/expert-profile.bin` + `data/expert-profile-learned.bin` (parser =
   make_profile.py:33-44); compute S(l) per §2.2; emit the hot/cold bands and the per-(layer,
   role) type table (default RECIPE-α; `--recipe beta|alpha+r2` for the variants).
2. Read both shard headers (gguf_reader.py — header-only, as quant_formats.py does) and write
   `$RQ/tensor-types.txt` with **1,224 explicit lines**: every expert tensor gets its chosen
   `iq2_s`/`q2_0`/...; EVERY other tensor gets `=COPY` ("only copy tensors, no quantizing",
   quantize.cpp:74, accepted as a --tensor-type value at :584/:603). Explicit-everything is what
   makes the dense side, PLE table (IQ4_NL, 28.8 GB), token_embd (IQ4_XS) and head provably
   byte-identical — llama-quantize's default arch rules never fire.

### 4.4 Quantize + assemble

```
# (only if 4.5 needs it) merge:
build-rq/bin/gguf-split --merge models/Qwen3.8-...-IQ3_S-00001-of-00002.gguf $RQ/merged.gguf
build-rq/bin/llama-quantize \
    --allow-requantize --imatrix $RQ/imatrix.gguf --keep-split \
    --tensor-type-file $RQ/tensor-types.txt \
    $RQ/merged.gguf $RQ/rq1.gguf IQ3_S   32
# first run WITH --dry-run to print the expected size (should read ~79-80 GB for v1)
```
- `--allow-requantize` is mandatory (source is IQ-quantized) — its WARNING is the §4.6 source
  caveat, bounded here to 36 cold-layer tensors.
- `--keep-split` preserves the 2-shard layout so the config keeps `--native`/`--ple-gguf` (if
  split handling misbehaves: quantize the merged file without --keep-split and drop `--ple-gguf`
  from the config — single-file --native is supported; the pack just names one shard).
- Assemble the pack from the NEW shards (iq_pack.py regenerates native_experts.txt from the
  headers, satisfying every check_experts_gguf clause automatically):
```
python tools/iq_pack.py --gguf $RQ/rq1-00001-of-00002.gguf --out $RQ_PACK
```
dense.bin/index.txt regenerate byte-identical (conversions.json was empty; the float tensors are
untouched), and the engine self-verifies the rest at boot.

### 4.5 Boot + loader verification (free failures)

```
cp custom-iq3_s-8081.json custom-iq3s-rq1-8081.json
#   deltas ONLY: --pack -> $RQ_PACK ; --native -> $RQ/rq1-00001... ; --ple-gguf -> $RQ/rq1-00002...
#   --expert-profile-save -> data/expert-profile-learned-rq1.bin   (protect the standing learned
#   file from new-pack traffic — the rollback-cleanliness clause, §7)
bash campaign/t17-leg.sh custom-iq3s-rq1-8081.json rq1-boot0
```
Loader acceptance to verify in the log: `check_experts_gguf` passes silently; the boot prints
`native pack: .../iq3_s_rq1 experts (largest blob 1.51 MB)` and the new ledger line
`expert cache NNNN slots, 17.71 GiB` — NNNN is the measured slot yield (predicted ≈10,129).
A refusal here (type/shape/offset mismatch) is a pack-assembly bug, caught for free.

### 4.6 Source paths (decision record)

- **Path 2 (v1, PRIMARY): dequant-requant from the local IQ3_S shards**, as above. Cost of the
  extra generation: confined to 36 cold-band tensors; imatrix-weighted; the yardstick canary (§5)
  measures exactly this before anything else touches the pack.
- **Path 1 (quality fallback): family graft, zero requantization.** The GSQ-RCO family
  (ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, per data/gguf_fingerprints.json) ships the SAME
  checkpoint at IQ2_XS (experts 2.348 bpw; mix IQ1_M/IQ2_S/IQ2_XXS/Q2_0) and IQ3_XXS (2.842).
  Survey their per-layer menus WITHOUT downloading via
  `tools/quant_formats.py --hf ISTA-DASLab/... <rev> IQ2_XS` (HTTP range requests only,
  quant_formats.py:70-77). Where a family file already holds a needed (layer, role) at the wanted
  type, graft that tensor (first-generation error, no compounding). Requires the ~68 GB IQ2_XS
  download and a small graft variant of iq_pack.py (read three model sets, pick per cell).
- **Path 3 (out of budget): true requant from a high-precision source.** No BF16/Q8_0 of this
  checkpoint exists locally (searched /home, /mnt/HQ, /mnt/STORE, /mnt/SCRATCH this session);
  Unsloth's Q8_0 file is 188.2 GB, BF16 ~372 GB (186B-class params incl. the 51.2B PLE table).
  Best quality, days of download; only if the campaign later wants the 2.9-bpw tier.

---

## 5. Validation ladder (one-boot falsifier first; batteries only after the canary)

Law in force: bits-changing arm ⇒ coupled-MTP acceptance battery + 2x MMLU (T16 #9,
SVG-CUSTOM-TODO.md:1090-1092); adopt bar +4% with no priority metric regressing past −4%
(gates-spec §0); F-225: never compare across boot positions/sessions.

| step | instrument | pass/fail | cost |
|---|---|---|---|
| V0 | `--dry-run` size print + iq_pack assembly + boot loader check (§4.4-4.5) | binary (pack bug) | ~30 min |
| V1 **G2 logit canary** (the T11.R5 G2, adapted): `python tools/quant_quality.py run --config custom-iq3s-rq1-8081.json --out $RQ/rq1.tsv` then `compare` with the CURRENT pack's run as REF (one ctl run first). Teacher-forced top-1 agreement + KL(current‖new) + PPL on data/quant_quality/v1 (5 x 1,000 tokens; the implemented logprob surface T11.R5 asked for, docs/QUANTS.md:56-63) | top-1 ≥ 97% AND mean KL ≤ 0.05 AND KL p99 ≤ 1.0 (calibrated against the field rows: Q8_0→IQ3_S is 0.144/1.975 — a cold-layer recut must add a FRACTION of one format step, not a step; QUANTS.md:73-81) | 2 x ~20-30 min |
| V1b routing-shift check: one `--dump-routing` boot per pack over the same 100k-token probe; per-layer top-10 agreement (GEMQ monitor; the harness is make_profile.py:46-58's parser) | mean agreement ≥ ~97% (= today's miss scale); no layer < 90% | 2 boots, ~10 min each |
| V2 **probe-lane A/B in one boot session** (the falsifier's M3b): `bash campaign/t17-leg.sh custom-iq3s-rq1-8081.json rq1-arm` + ctl leg same night; M3b is informational-only here (bits change by design — gates-spec §2 G2's M3b discipline applies to the DRAFTER, not the pack), the read is M1/pair/conc vs the same-night ctl band | M1 within −4% floor; conc expected positive; NO adoption claim from one boot (F-231's +1.81% lesson) | ~1 h |
| V3 **quality floor:** 2x MMLU-Pro (matched arm/ctl, same session, gates-spec G3 commands) + mimic via mimicbench2 on the arm boot | MMLU mean ≥ 95.0 floor law, mimic ≥ 87.9 | ~1.5-2 h |
| V4 **No-Regression table, all three lanes:** five-boot protocol per lane (M1/pair/conc means vs the 3135-re-banked baselines — T16 M0 requires the re-bank before ANY absolute milestone claim) + M3b + acceptance battery per F-230 law | adopt iff M1 ≥ baseline − 0 AND the +4% bar where claimed; conc/pair no-regress | ~3-4 h serial |
| V5 | stamp + bank (§7), then and only then the T14B harvest launches on the new pack | — | ~15 min |

Abort rules: any V1 fail → retreat to RECIPE-β (gu-only) or Path-1 graft and re-run V1 (cheap:
the recut set shrinks, no battery was spent). Any V3 fail → the pack is dead; rollback §7; the
T14B harvest proceeds under the harvest-spec's PRE-REQUANT stamp clause (harvest-spec.md:380-383).

---

## 6. Interactions

### 6.1 VRAM/slot math feeds the 262k switch (T15B/T16 W3)

T15B's standing risk: at 262k the KV grows and the expert cache drops 9,337 → ~8,630 (−7.5%) =
F-183's ±500-slot band = −4.9% conc exposure (SVG-CUSTOM-TODO.md:1080-1081). Re-derived with v1:
`10,129 x (8,630/9,337) = 9,369 ≥ 9,337` — **the re-quant fully absorbs the 262k slot loss at
zero cost to the switch**; the 262k launcher refresh (stale, T16 W3) should re-read its slot line
against 9,369, not 8,630. This also compounds with the parked position-ladder work (deep-ctx
decode 226-243 t/s warm, SVG-CUSTOM.md:8007): smaller blobs cut that lane's PCIe/page-cache
traffic proportionally.

### 6.2 T17E C4 domain-shift guard

C4 (divergence-triggered un-pin/re-save, variance class, T17E dossier + SVG-CUSTOM-TODO.md:1124-
1126) exists to protect the profile/cache under routing drift. A re-quant is a small deliberate
routing shift (GEMQ hazard, V1b measures it) — the same class C4 guards. Sequencing: C4 stays
QUEUED; v1's mitigations are (a) the fresh `--expert-profile-learned-rq1.bin` path (§4.5), so the
adaptive tier re-learns on-pack without touching the standing artifact, and (b) V1b's agreement
gate. If V1b shows a layer < 90% agreement, that layer's profile prior is stale post-requant —
pin C4's re-save-cadence pull-forward into the adoption commit.

### 6.3 F-230 does NOT apply — precisely

F-230 (SVG-CUSTOM.md:7884-7912) measured DENSE-MMQ: a prefill-PATH numerics change (Q8-activation
cuBLASLt GEMM vs FP16-dequant) that shifted ONLY the prompt's KV (~1.8e-4 KL) while decode kept
the old bits — the asymmetric case. The coupled exact-match verifier then flips borderline
accepts: −2.8% M1 with no offsetting win. The re-quant is the OPPOSITE shape: prefill and decode
read the SAME new expert bits, the drafter binds to the same shifted backbone (T14B trainer
§: the drafter trains ON the serving pack — the whole point of the sequencing law), so no
prefill/decode asymmetry and no draft/target divergence can arise (T17F's own resolution,
campaign/research/T17F-training-quant-context.md:63). What DOES carry over is F-230's mechanism:
the acceptance baseline is hypersensitive to bit distance → hence V2's acceptance accounting and
the T14B pack-coupling (harvest-spec.md:381-383: "the pack's prefill bits shift the hiddens
themselves").

### 6.4 Other couplings (each one line)

- **Conversation cache:** the fingerprint hashes native_experts.txt + sampled source files
  (expert_source.cpp:4290-4310; conversation_file.cpp:938) → the cache invalidates on pack swap
  automatically and correctly. No action.
- **Learned-profile slot-fill:** the cache fill walks the profile in rank order (generate.cpp:
  5660-5665) — unchanged; only blob sizes shrink.
- **MTP drafter:** the current rt drafter was harvested/distilled on the OLD pack's numerics; its
  acceptance may shift a little (F-230 mechanism). V2's M3b reads it; the T14B retrain then
  re-anchors on the new pack (the sequencing law's purpose). If V3 passes but V2's M3b drops
  > ~1.5pp, note it as expected-until-r18 rather than a blocker — the retrain is scheduled anyway.
- **prefill auto:32768 / kv int8 / pcie dma:** untouched; no config delta beyond §4.5's paths.
- **262k launcher (stale):** do NOT point the old launcher at the new pack until its T15B arm
  refresh lands (T16 W3) — one variable per battery.

---

## 7. Rollback (and the stamp)

**Invariant: nothing existing is written.** The current pack dir, both current shards, and the
standing learned profile are read-only inputs to every command above. Rollback =
`custom-iq3_s-8081.json` back as the config on 8081 (one restart) — byte-identical by
construction. The conversation cache re-fingerprints back automatically.

**Stamp procedure (banked BEFORE the first new boot):**
```
mkdir -p campaign/results/requant-stamp && cd /home/svg/LLM/Custom-data
sha256sum models/Qwen3.8-...-IQ3_S-0000{1,2}-of-00002.gguf \
          packs/iq3_s/native_experts.txt packs/iq3_s/index.txt \
          packs/iq3_s/dense.bin packs/iq3_s/conversions.json \
  | tee /home/svg/LLM/strata-custom-2/campaign/results/requant-stamp/pre-pack-sha256.txt
cp data/expert-profile.bin data/expert-profile-learned.bin \
   /home/svg/LLM/strata-custom-2/campaign/results/requant-stamp/
```
(harvest-spec §2 expects exactly these fields in its manifest: pack file sha256s + config hash +
env hash.) The same command against the NEW artifacts post-adoption completes the pair; the
harvest manifest then cites the post stamp. Boot-line slot ledger (`expert cache NNNN slots,
17.71 GiB`) for both packs goes into the same directory as `slots-ledger.txt`.

---

## 8. Time and disk budget (honest wall-clock, 9950X3D + NVMe, box otherwise idle)

| stage | estimate | basis / risk |
|---|---|---|
| Fork build (CPU targets) | 5-15 min | llama.cpp core + 2 tools, -j32; NEW build dir, nothing overwritten |
| Corpus emit | ~5 min | files already on disk (depth-lane-real pool) |
| **imatrix pass, 675k-1.35M tok, CPU-only** | **1.5-5 h** | qwen4exp forward in the fork at ~40-120 tok/s class on 32 threads; NOT NVMe-bound. The one genuinely uncalibrated number — hence the §4.2 entry-coverage GATE on a 10-chunk smoke run first (fail fast, ~2 min). Optional CUDA fork build cuts this to <1 h at +20-40 min compile |
| Chooser + tensor-types | <5 min | header reads |
| Quantize v1 (36 tensors, ~5 GB in / 4.15 GB out) + COPY the rest (~79 GB) | 1-2.5 h total | IQ requant with imatrix is compute-bound; COPY streams at NVMe speed (84 GB read + 80 GB write ≈ 10-30 min at 1.5-3 GB/s). --max-buffer-size 8192 default fits 184 GB RAM trivially |
| iq_pack assembly | 10-20 min | memmap reads of both new shards + 1.5 GB dense write |
| V0-V2 (canary + A/B) | ~2-3 h | §5 |
| V3-V4 (quality floor + five-boot) | ~5-6 h serial boots | §5; the campaign's serial-batteries law applies |
| **CPU-work total (V0-V2)** | **~4-8 h = well inside 1-2 CPU-days** | T17F's envelope holds |
| **Full ladder to adoption** | **1.5-2 days elapsed** | dominated by serial boots, not CPU |
| Disk peak | ~165 GB of 705 GB | old (read-only) + merged 83.7 (transient, deletable after 4.4 — keep until the stamp) + new ~80 + pack 1.5; ≥ 500 GB always free; the harvest's 163.8 GB fits after |

---

## 9. Contradictions encountered, and their resolutions

1. **T17F's +6-10% M1 vs the standing-recipe window arithmetic (+2-4%).** Not an error in either:
   the envelope priced the slot term at the F-183-era miss regime (T12-MEM slope applied
   unsaturated); T13's re-bank moved standing-warm miss exposure to 0.3-0.7 ms
   (SVG-CUSTOM-TODO.md:973). Resolution: ship the decomposition (§3.5), keep +6-10% as the
   outer envelope, and let the five-boot means price it. The 262k absorption (§6.1) is the
   unsquandered part of the slot term.
2. **"imatrix sweep over the MIMIC corpus" (T17F:64) vs MIMIC absent** (harvest-spec §3.1
   re-verified twice). Resolution: calibration corpus = the harvest's own seed pool (repo prose +
   gated HQ slice) — which is ALSO the content-matched serving mix F-236 promoted. The two
   artifacts then share provenance, which is exactly what "drafter trains on the serving
   distribution" wants.
3. **T17F "expert-profile-save-every 30" as requests (SVG-CUSTOM-TODO.md:1100) vs the code's
   MINUTES** (generate.cpp:954-957, default 10; the launcher passes 30). Cosmetic, corrected
   here because the learned profile's freshness claim (Oct-18 file) depends on it.
4. **expert-profile-coder.bin as a third sensitivity source** — rejected: 256-expert geometry,
   refused by read_expert_profile for this model (expert_cache.cpp:104-112).
5. **Down-proj sensitivity (MxMoE) vs cutting down to Q2_0 in the cold band.** Resolved by
   structuring the retreat: RECIPE-α spends the down cuts ONLY where the routing data says cold;
   RECIPE-β (gu-only, down untouched) is pre-authorized on any V1 wobble. The G2 canary
   arbitrates, as T17F:100 says it should.

## 10. Open questions

1. Does the fork's llama.cpp qwen4exp forward pass produce sane imatrix entries on the first
   smoke run (§4.2 gate)? Decides Path 2 vs Path 1 in the first 10 minutes of real work.
2. Does `llama-quantize --keep-split` handle the 2-shard input cleanly, or is the merge
   intermediate required (§4.4 fallback)? One --dry-run answers it.
3. Exact CPU tok/s of the fork's qwen4exp prefill on this box (§8's 1.5-5 h band). Priced by the
   smoke run; the CUDA-build option exists if the CPU number embarrasses itself.
4. Does V2's M3b read the drafter's old-pack conditioning visibly (§6.4)? Expected shift is
   small and self-healing at r18; the threshold (>1.5pp) just keeps it honest.
5. (Post-adoption, T16 W3 input) Does the refreshed 262k launcher re-derive its slot expectation
   from 9,369 (§6.1)? One line in that spec, pending this pack's boot ledger.

## 11. Sources (all read this session, 2026-10-19)

Code: src/kernels/cpu/expert_layout.cpp:376-493 (v3 parse, contiguity, blob check);
src/kernels/cpu/native_expert.cpp:34-140 + include/.../native_expert.hpp:16-57 (blob law, kernel
coverage, mt rule); src/core/expert_source.cpp:1103-1199 (GGUF-in-place mapping),
:4290-4310 (fingerprint), :4329-4391 (check_experts_gguf), :4569-4606 (from_gguf decision);
src/core/expert_cache.cpp:72-209 (profile read/write/rank); src/program/generate.cpp:5470, 5600-
5616, 5645-5686 (slot fill law), :954-957, :2295-2297 (profile-save flags); tools/iq_pack.py
(header doc, :430-488 expert table, :608-614 atomic write); tools/quant_formats.py:25-32, 70-77;
tools/quant_quality.py (whole; the G2 surface); tools/make_profile.py:1-58 (profile/trace
formats); tools/quantscope.py:117-124 (type-42 identification); tools/quant_bound.py:21-32
(routed-bytes model); campaign/battery-config.sh:1-46 (M3b line, env-source law).
Data: /home/svg/LLM/Custom-data/packs/iq3_s/native_experts.txt (48 rows parsed, all formulas
verified); data/expert-profile.bin + -learned (parsed: 24,576 ranks; layer shares computed);
data/expert-profile-coder.bin (rejected, 48x256); data/gguf_fingerprints.json (family bpw table);
GGUF shard-1 header dump; boot logs custom-iq3_s.log:34-35, custom-iq3s-adaptmtp.log:5-11
(slot ledger, launch line).
Campaign record: SVG-CUSTOM-TODO.md:909-935 (T12-MEM), :952-965 (T12-GPU), :969-1005 (T13),
:1075-1085 (262k risk), :1130-1148 + campaign/research/T17F-training-quant-context.md:9-65,
:97-110, :151-163 (T17F item 1 + GEMQ/MxMoE/imatrix sources), :1215-1236 (T17 verdict),
:1270-1291 (T14B order); SVG-CUSTOM.md:7884-7912 (F-230), :7936-7938 (deploy law), :7963-8013
(F-233 + scoreboard), :8015-8072 (F-234/234a matrix), :8074-8135 (F-235/236), :5920-5981
(F-183), :5815ff (F-181); campaign/T14B-specs/harvest-spec.md (§2 stamp, §3.1 corpus, §6 storage,
§8.1 sequencing, R3); campaign/T14B-specs/gates-spec.md (§0 anchors/bars, §2 G0-G5);
campaign/research/T17E-moe-serving-leaps.md:51-52, :205-213 (C4).
Toolchain: third_party/llama.cpp — src/llama-arch.cpp:43, src/llama-model.cpp:330,
src/models/qwen4exp.cpp, tools/quantize/quantize.cpp:36, 73-74, 120-146, 584-603,
src/llama-quant.cpp:858, ggml/include/ggml.h:432, ggml/src/ggml-quants.c:74; fork git describe
v0.1.44.1-126-gadf5f25f.
Methods note (searches that came up empty, so nobody repeats them): NO BF16/F16/Q8_0 GGUF or
safetensors of this checkpoint anywhere on the box (find over /home, /mnt/HQ, /mnt/STORE,
/mnt/SCRATCH at depth 3-4); no prebuilt llama.cpp fork binaries; no experts.bin in any active
pack dir; no in-repo imatrix artifact; "MIMIC corpus" confirmed absent a third time.
