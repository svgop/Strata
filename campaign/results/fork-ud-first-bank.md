# Fork-UD first bank, 2026-11-06 (1553a614 + this file)

The fork now serves UD3.1 natively (the port: Q5_0 dot fix + parity pins; the rest of the
contract was already ours). First fork-UD measurement, 2-leg means, same box/session as
the GSQ means (229.1 / 220.7 / 301.6):

| lane | fork GSQ IQ3_S | fork UD-IQ2_XS | stock UD (best) |
|---|---|---|---|
| solo | 229.1 | 171.8 | ~203 |
| pair | 220.7 | 185.1 | ~195 |
| conc | 301.6 | **292.3** | ~200 |

MMLU-Pro on fork-UD: 92.86 +/- 3.08 (in-class with fork-GSQ 94.29 +/- 2.77).

The headline: the conc collapse that held stock's UD at ~200 is GONE on our engine
(292 vs stock's best ~210 - our W1 controller does not under-window UD). What remains
is the solo/pair conversion tax, named by the kernel study + dispatch counters:
1. The HC read runs the Q8_0-COPIES form (UD's Q6_K dense side) instead of bf16 rows -
   a per-layer conversion copy per window. Load-time fix: convert Q6_K dense -> bf16
   rows (the architecture support GSQ has had).
2. The final head runs its Q8 fallback form - preconvert at load.
3. The dense Q8_0 (attn_qkv/ssm_out, ~2 GB/pass) has no IL path (only K-quants are
   whitelisted in native_mmvq_il_supported).
Those three are the "UD much faster than GSQ" levers; until then UD-IQ2_XS on the fork
is the memory-economy play at conc-parity, not the speed win.
