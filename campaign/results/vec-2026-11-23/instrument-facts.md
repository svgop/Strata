# Instrument facts for the control-vector work — measured 2026-11-23

Every line below was established by running the engine (`engine/strata`, engine 0.1.46, the
champion binary of 2026-11-20 15:34) or by reading its source at the cited line. They replace
beliefs the 2026-11-20 and 2026-11-26 dossiers carry.

## 1. `--dump-layers` cannot run on the deployed pack

The deployed model is the native IQ pack `packs/iq3_s` (`native_experts.txt` present →
`expert_layout().native`). Three independent gates close it, each confirmed by boot:

| Gate | Evidence |
|---|---|
| native pack requires `--spec >= 2` and `--prefill CHUNK` | `src/program/generate.cpp:3891`; without them: `is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`, rc 2 |
| the residency table is built only when no layer dump is open | `generate.cpp:7697` (`... && layer_dump == nullptr && half_dump == nullptr`) |
| the speculative loop refuses without the table | `generate.cpp:18017`: `--spec needs the device residency table`, rc 2 — reproduced with `--max-new 256` and with `--max-new 1` |
| the sequential loop exits before its first iteration on native packs | `generate.cpp:17770` `if (native_pack) { spec_pos = pos; break; }`; the only layer-dump write is `:17867`, inside that loop |

Consequence: `tools/persona_capture.py` (2026-11-21) could never produce a row. It also passed a
`generate` subcommand the binary does not take (`unknown argument: generate`, rc 2 in 0.1 s), and on
a failed boot it logged the failure and continued, then wrote an empty result and exited 0.

## 2. The legacy pack dumps every position

`packs/q2_0` runs the sequential path. Command (from the repo root):

    engine/strata --pack /mnt/DATA/LLM/Models/GGUF/strata/data/packs/q2_0 \
      --ple-gguf /mnt/DATA/LLM/Models/GGUF/GSQ-RCO/qwen3.8-flash-next/gsq-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
      --gpu 0 --tokens-file F --max-new 2 --seed 1 --temperature 0 --max-context 4096 \
      --expert-cache 1000 --vram-reserve-mib 8000 --dump-layers OUT

- rc 0; an 85-token prompt with `--max-new 4` gave 88 rows = P − 1 + M; 2,007,040 bytes per row
  (49 slots × 4 streams × 2560 × f32); layout `rows × 49 × 4 × 2560`, little-endian f32.
- Slot 0 = input embedding; slot k = residual after block k−1. All prompt positions are dumped, so
  a response placed in the prompt is captured with no generation.
- Cost: 17 s wall, 34 GB maxrss for the first boot; ~12–15 s per boot with a warm file cache.

**Store defect.** Slots 45–48 are non-finite in every row (matches the launcher's own line: `no
VALIDATED store ... the canonical pack is corrupt, T68`). Slots 0–44 were finite for every prompt
tried. The pack's generated text is therefore garbage and is never used.

**Confound, stated.** This is a 2-bit quant of the model. A direction measured here is an
approximation of the deployed model's; its worth is decided by validation on `iq3_s`.

## 3. Index rule — proven by arming one layer

Two legacy boots on the same 85-token prompt, `--max-new 2` (86 rows): (a) no vector; (b)
`--control-vector-scaled .../qwen3.8-flash-next-refusal-r.gguf:1.0 --control-vector-layer-range 22 22
--cvec-mode project --cvec-dir per-layer` (engine line: `layers 22..22 (1 steered)`).

| Dump slot | 0 … 22 | 23 … 44 |
|---|---|---|
| positions changed by the vector | 0 / 86 | 86 / 86 |

So `direction.<l>` acts on the residual AFTER block l = dump slot l+1.

    dump slot k  ->  tensor direction.<k-1>      (k in 2..48)

Slot 1 (after block 0) is not steerable (`direction.0` is dropped, `generate.cpp:1972`; the range
is clamped to ≥ 1, `:1988`); slot 0 is the embedding. Source agrees: the kernel indexes `s_l[layer]`
and `dir + layer*n` by its raw argument (`src/kernels/cuda/cvec.cu:53,56`) and every call site passes
the block index (`src/core/layer.cpp:1639-1650`, `src/prefill/prefill.cpp:4770-4828`,
`src/core/verify.cpp:2371-2378`). `tools/persona_vector.py --method dump` writes slot k to
`direction.<k>` — one block late — and must not be used for measured rows.

## 4. Residual norms are small (first measurement)

Per-stream mean ‖h‖ over the 85 positions of one prompt, legacy pack, finite slots:

| slot | 0 | 10 | 18 | 22 | 27 | 33 | 37 | 40 | 44 |
|---|---|---|---|---|---|---|---|---|---|
| mean ‖h‖ | 0.23 | 1.66 | 2.03 | 2.25 | 2.94 | 3.91 | 6.16 | 8.06 | 10.72 |

A unit row added at scale s is therefore s/‖h‖ of the residual per layer: at slots 19–27, 0.08 is
3–4 % and 1.0 is about 40 %. Dose lists must be set from this, not from the 0.02–0.4 range the
r01 sweep used.

## 5. What the loader preserves

`generate.cpp:1951-2036`: `data[l]` = Σ over files of (file scale × `direction.<l>`). In project
mode `s[l] = ‖data[l]‖` and `dir[l] = data[l]/‖data[l]‖`; in add mode `s[l] = 1` and `dir[l] =
data[l]`, raw. Per-layer magnitude survives in both modes, so per-layer calibration needs no engine
change. One mode and one summed direction apply per boot: two files at the same layer are added
before any mode is applied — a project-mode refusal vector and an add-mode persona vector cannot
share a boot.

## 6. The per-request switch

Under `--serve`, the body key `experimental_speed_projection` (true or absent = armed, false = off)
switches a loaded vector per request. The flag is one device int shared by every batch slot
(`cvec.cu:18,137-150`), so ON/OFF probes are valid only strictly sequentially. Conversation and
prompt caches are keyed by the cvec state (`generate.cpp:14097,14586`), so cache state does not
cross between the two arms.

## Where the raw evidence lives

`/home/svg/LLM/Custom-data/persona-vectors/capture/` — `probe-timing/` (the native-pack refusals,
the first legacy dump, the norm measurement) and `index-rule/` (the two armed/unarmed dumps and
their engine logs). Not in git.
