# t22 verdict — stock v0.2.11-3 (engine 0.1.57) vs custom, 2026-11-05

The ask: pull upstream (v0.2.7 → v0.2.11-3, 1,282 commits) and run our models there —
has it caught up to us? Method: t22's interleaved legs (stock 8080 / custom 8081, warm
request both sides, custom waits for prewarm), same window, box load settling ~5.5→~2
through the session, probe-lane on each port.

## The lanes (same window, 2026-11-05 evening)

| lane | stock 0.2.11 (2 legs) | custom (4 legs) | custom bank (Nov-2) |
|---|---|---|---|
| M1 solo | 191.4 / 180.9 → **186.2** | 229.2 / 202.4 / 191.3 / 230.8 → **213.4** | 232.2 |
| M2 pair | 187.4 / 196.2 → **191.8** | 217.8 / 192.8 / 192.6 / 197.7 → **200.2** | 222.3 |
| M2 6-conc | 337.4 / 339.4 → **338.4** | 289.0 / 271.3 / 263.1 / 264.3 → **271.9** | 315.4 |

**Verdict: not caught up — and ahead on one lane.** Custom holds solo (+14.6%) and pair
(+4.4%) on the spec-decode single-stream path (our W1 controller + arc). Stock's 6-conc
is decisively ahead: +24.5% over today's custom mean and **+7.3% over our clean-window
bank** — two tight legs (337.4/339.4), the best conc number ever measured on this box.

KV diagnostic: custom@q4_0-KV read conc 278.9 — inside the int8 spread (263-289). The KV
format does not explain stock's edge; their conc win is engine work (the 0.2.8-0.2.11
batched-decode wave).

## Boot-contract findings (each cost a leg to learn)

Stock 0.2.11 refuses/fatals where our fork serves:
1. `--expert-profile <missing file>` → **exit 1** (fatal), not a warning.
2. `--resident-experts` now REQUIRES `--mmap-experts` + a static profile (0.2.1-era arg
   no longer maps; dropped from the stock config).
3. At 262144 with `--kv int8` + `--spec 6`, the verify tier refuses: no VRAM left for the
   profile-filled expert tier. Their sanctioned path on a 32-GB card: `--kv q4_0` +
   `--expert-profile data/expert-profile.bin` (the shipped base, all 24,576 pairs ranked).
   Our fork runs this exact 262K/int8 recipe standing (int8 + resident experts + 49 GiB
   conversation cache + prewarm).

Stock's cvec loader arms our vector correctly at the asked range (1..47, 47 steered).

## The harvest list (ranked)

1. **The conc-lane class** — import/study first: the auto CPU/PCIe miss-share rule
   (213f1a39: ceiling 0.95, k falls 1.85@30GB/s→0.8@12, `pcie_share_rule.hpp` + test),
   FNX_PF_ROWS 8192 / PF_HCSPLIT 2 / PFX_ATTN_I8 NVIDIA defaults (6983b09b), the batch
   planner's per-lane draft-round pricing (0748e927). Target: our batch window's miss
   path and admission pricing.
2. **Unsloth v3.1 quants for our exact model** (9ed3c778: UD-IQ2_XS, UD-Q3_K_XL,
   UD-IQ4_XS-v31, ...) — re-bake material for the quant wave; loader compatibility is an
   F-370-class study first (fused `_exps` layout law from the UD-IQ1_M verdict).
3. **DFlash2 draft source** (c850b979, greedy-identity proven) — a competing design for
   the decode-control pair; study before building.
4. Q5_0 expert-dot fix (aa678f10) — correctness; we run no Q5_0 experts (low).
5. Mac/Metal dsv41 vision — not applicable on this box.

Instrument notes: battery-lib's GPU sweep is REPO-scoped by design — a stock-tree engine
holding the GPU survives it; dual-checkout work must use t22's nvidia-smi-sweep cleanup.
Stock legs need no prewarm (none exists upstream — still true at 0.2.11); both sides got
one warm request before probing.
