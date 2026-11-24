# Router-visibility audit — slot-9 row vs r02 band (T2 falsifier, offline, zero boots)

Question (RARE, arXiv 2608.21236, ported per the low-level audit): is the steering perturbation
visible to the MoE routers? Metric: ratio = ‖R_l·v̂‖ / (‖R_l‖_F/√2560) per block — the direction's
router-logit leverage against the expected value for a random unit vector. Routers read from
`blk.<l>.ffn_gate_inp.weight` (BF16 [512 experts, 2560], unquantized in the PLE GGUF shard 1).
A ratio of 1.0 = statistically random (invisible); 2+ = strongly router-visible.

## Result

| vector | acted-on blocks | mean ratio | max |
|---|---|---|---|
| r02 band rows (direction.18–26) | 19–47 | **0.99×** (per row 0.97–1.01) | ~1.0 |
| r03 slot-9 row (direction.8) | 9–47 | **2.21×** | 3.25 (block 12) |

Per-block profile for slot 9: 2.81 (blk 9), 2.54 (14), 2.52 (20), 2.64 (27), 2.19 (35), 1.14 (44),
1.25 (47) — strong through the 9–35 band, fading late. Ratio > 1.2 on all blocks 0–47 except
39/43/44/46 (0–8 do not see the row; it acts after block 8).

## Two consequences

1. **r02's damage was NOT routing corruption.** Its band rows are router-invisible — the damage at
   dose 0.35 came through the residual/read-out channel (the 4× read-out dose of the shared-add
   geometry, audit 2409.19606). Consistent with the r02 dose table: facts fell with truncation
   (output channel), not with route-driven incoherence.
2. **Slot 9's row is router-visible (2.2×) — a risk specific to slot 9.** If the running slot-9
   ladder shows facts damage at doses r02 survived, the router channel is the suspect, and the T2
   projection is the contained fix.

## T2 projection, measured and built

P = I − Rᵀ(RRᵀ)⁻¹R against block 12's router (the max-visibility block):

| variant | norm kept | vis after (mean/max) |
|---|---|---|
| none | 1.000 | 2.21 / 3.25 |
| P9 | 0.744 | 1.04 / 1.48 |
| **P12** | **0.729** | **0.84 / 1.17 ≈ random** |
| P9+12 | 0.587 | 0.75 / 1.37 |
| P9–14 concat | 0.000 (null space empty) | — |

The P9–14 result is the mechanism: six routers' row-spaces fill all 2560 dims — the routers share
one dominant semantic subspace, and a single block's null space already escapes it globally.
(Steering at the ⅓–⅔ depth band, audit T4, aligns: block 12 sits at depth 0.25, the CAA mid-band.)

Built: `/mnt/DATA/LLM/Models/GGUF/vectors/qwen3.8-flash-next-persona-r03-assertive-s9-p12.gguf`
(direction.8, F32[2560], row norm 1.0276 = same calibration as the plain row; provenance embedded).
Arms only as the NEXT leg after the plain ladder verdict — one variable per leg.
