# The shadow expert store

**Status:** architecture, shipped and standing on the Qwen lane.  Owner: `src/core/shadow_experts.cpp`,
`include/strata/core/shadow_experts.hpp`.  Adopted F-359/F-360 (2026-10-27); placement normalized 2026-10-29 (F-374).

## What it is

The engine splits every routed expert pick across three tiers: VRAM (the expert cache), PCIe (streamed from the
pinned host arena), and the CPU lane.  The CPU lane is the slow one and it is **byte-bound**: its speed is how
many bytes of expert weights it must pull from DRAM per entry, not how fast it computes.  On the IQ3_S pack a
blob is ~2,046,400 B and an entry costs ~150 µs.

The shadow store is **the same model at a lighter quantization** (Q2_0: 1,382,400 B a blob — 33% fewer bytes)
mapped beside the running pack.  When an expert misses VRAM and lands on the CPU lane, the pool computes it from
the shadow instead of the native pack: ~75 µs an entry instead of ~152.

Only the CPU-miss path reads it.  The GPU, the PCIe share, the prompt path and every cache decision are untouched.
The ~3% of picks that take this path therefore carry Q2_0's quantization error instead of IQ3_S's; measured quality
on the gate: MMLU-Pro 95.71 (floor 93.3).

## Why it exists

Not a micro-optimization: it is the only lever that reduces the CPU lane's bytes without changing the model's
serving quantization, which is fixed by the pack.  It was worth +7.7% pair / +8.4% conc when adopted (F-360), and
+5.5% M1 / +4.2% pair / +3.5% conc from the tmpfs placement alone (F-367).

## Where it lives — artifact vs placement

Two different things, and conflating them is the mistake this document exists to prevent:

| | path | what it is |
|---|---|---|
| **Artifact of record** | `<models-dir>/packs/q2_0/` | a complete, publishable pack: `experts.bin` (32 GB), `experts.json`, `index.txt`, `manifest.json`, `tokenizer/`.  Built by `tools/iq_pack.py` from the published Q2_0 GGUF shards.  Same form as any other pack. |
| **Placement cache** | `/dev/shm/strata-shadow-q2_0/` | a byte-identical copy on tmpfs.  **Pure placement, not a second artifact.**  RAM-resident pages are THP-eligible and never evicted, so the pool pays no page faults (F-365: 2,635,199 minor faults per probe leg on the page-cached copy). |

`/dev/shm` is tmpfs and is **wiped on every reboot**.  `campaign/shm-shadow.sh` rebuilds the placement cache
(12 s for 37 GB) and is a no-op when the copy is intact; `launch-custom-iq3s-8081.sh` calls it before every start,
so the lane never silently loses the mechanism again.

The launcher resolves the store in three tiers, so the lane always boots:

1. `/dev/shm/strata-shadow-q2_0` — placement cache present: the fast path.
2. `<models-dir>/packs/q2_0` — canonical artifact, page-cached: correct, slower misses.
3. unset — no shadow: the native store serves every miss.  Outputs are byte-identical to an unshadowed run.

`STRATA_SHADOW_EXPERTS` overrides the choice; `STRATA_SHADOW_EXPERTS=""` disables the shadow entirely.

## Requirements

- **RAM.** The copy is as large as the pack's `experts.bin` (32 GB for Qwen3.8's Q2_0).  Budget it beside the
  engine's own pinned arena — on a 64 GB PC take the canonical artifact and skip the placement cache.
- **The pack must be the same model, same geometry, same layer count.** `arm()` validates `experts.json` against
  the running pack's geometry (`n_layers`, experts per layer, blob bytes, layer offsets) and refuses with one
  line if they disagree; the engine then runs unshadowed.  A Hadamard-folded pack cannot be shadowed (the
  canonical Q2_0 form cannot carry it).
- **The lighter quant must be materially lighter.** The mechanism is a byte-count cut, so it pays only when the
  shadow's blob is meaningfully smaller than the native one and the CPU lane is byte-bound.  On DeepSeek-V4.1
  the CPU lane already runs at 93-96% of the box's DRAM ceiling on *every* quant type the pack uses, and the pack
  is already mixed-quant by sensitivity (IQ1_M/IQ2_XXS/IQ2_XS, 1.75-2.31 bpw) — there is no lighter artifact to
  swap to, so dsv41 has no shadow.  Measured 2026-10-29, `ds4_gg_bench`: 44.8 / 45.6 / 47.1 / 45.7 GB/s for
  IQ1_M / IQ2_XXS / IQ3_XXS / IQ3_S against a 48-50 GB/s plain read.

## Installer

`Q2_0` is a first-class `--model` choice in `setup.py`, so a user installing the Q2_0 pack already has the
artifact.  A user on `IQ3_S` (the quality tier) can add the shadow by installing Q2_0 beside it — the two packs
coexist, and the launcher's `STRATA_SHADOW_EXPERTS` points the miss path at the lighter one.  When publishing
a pack set, ship the Q2_0 pack as the shadow companion of the IQ3_S/IQ3_XXS tiers; it is the only extra artifact
the architecture needs.

## Rollback

`export STRATA_SHADOW_EXPERTS=""` — no mapping, no arm, byte-identical outputs to an unshadowed run.
