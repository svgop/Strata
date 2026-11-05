# UD3.1 Qwen on stock — the verdict, 2026-11-06

The ask: test the UD3.1 Qwen sets on strata (stock) instead of porting first. Done:
setup.py --model UD-IQ2_XS from the local 76-GiB set (data/models/UD3.1/Qwen3.8-Flash-Next),
engine 0.1.57 at tip, pack built at data/packs/unsloth-ud-iq2_xs, 262144 ctx, q4_0 KV.

## Measured

| metric | UD-IQ2_XS (stock 0.1.57) | our fork IQ3_S (ported, same box) |
|---|---|---|
| MMLU-Pro 70q | **97.14 ± 1.99** | 94.29 ± 2.77 |
| M1 solo | 202.0-203.8 | 224.0 |
| M2 pair | 178.1-190.0 | 216.7 |
| M2 6-conc | 180.6-186.2 | 295.3 |
| arena | 42.3 GB | 80.8 GiB |

Isolations: --spec 6 (matching our depth) moved nothing (202/190/181 vs 204/178/186);
decode hit rate 99.0-99.5% (the miss path is innocent); stock's OWN IQ3_S pack on the same
engine reads ~186/~192/~338 — the UD set is slower than BOTH engines on IQ3_S.

## Verdict

1. **The UD3.1 quality claim is real on this box**: equal-or-better MMLU-Pro at less than
   half the bytes of IQ3_S. The dynamic-quant lineage holds — the key reference point for
   the re-bake wave (the quant-rebake digest's hot/cold allocation thesis, validated).
2. **At IQ2_XS size it is a memory-economy play, not a speed play**: the mixed-type expert
   kernels (Q2_0 / IQ4_NL / Q5_0 downs) at 99% hit rate are pure-window compute and run
   10-39% below IQ3_S on this card. For a box that CAN hold IQ3_S, the small UD sets buy
   RAM, not tokens/s.
3. **The port question for our fork resolves to the BIG sets**: UD-Q3_K_XL (89.5 GB,
   IQ3_S-class bytes) or UD-IQ4_XS-v31 — a potential QUALITY upgrade at comparable size
   and speed. That needs the download + the 0.1.57 type-mix kernel port (F-370-class) —
   priced as the follow-up, not blocking anything.

The MMLU gate gained STRATA_GATE_PORT (env) for multi-port gates; the UD config on the
stock tree is spec 6 now (the recipe variable isolated), boots clean for future legs.
