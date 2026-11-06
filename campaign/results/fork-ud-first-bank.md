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

## The smalls' attribution (2026-11-06, quiet-box interleave, same binary)

STRATA_SMALL_UNR=0 control (both constant-trip arms off): 206.1/182.9/269.5 vs the armed
means 233.1/224.8/303.5 -> the mmvf router fast path + the GDN a/b constant-trip passes
are worth +13.1% solo / +22.9% pair / +12.6% conc ON GSQ - the largest single GSQ lane win
since W1, from two bitwise kernel-form changes ported from upstream's BW-DENSE line.

## The decode-controller composition (2026-11-06, honest negative)

L1 estimator AUC on OUR drafter: 0.676 tie-corrected over 3,149 clean windows (kill line
0.65 - survived, but a third weaker than vLLM's shipped 0.83). L5 composition (all three
arms) vs ctl, interleaved: armed-1 214.4/215.8/286.4 vs ctl-1 224.6/211.3/289.8 (adjacent,
mid-download); the later legs (armed-2 191.2, ctl-2 199.7) were depressed by the 88-GB
download's writeback - the back half of the interleave is confounded. What is clear: the
predicted +8-15% conc did NOT materialize (SURVTOPK's conc-kill condition trending true).
ARMS PARKED default-off (zero standing cost; the observe() extrapolation deletion is the
only unswitched change - zero engine call sites, bit-identical). A quiet-window attribution
ladder (L2-L4) remains priced if wanted.

## The IL-Q8 verdict + law 17's first case (2026-11-06 night)

The 3+3 interleaved A/B (law 17, loads recorded per leg, box at load ~4-7 through the
window): armed 195.0/214.3/281.6 vs control 197.0/206.6/277.2 = -1.0/+3.7/+1.6% - inside
the noise envelope, the +4% bar not cleared. DEFAULT FLIPPED OFF (bitwise kernels, zero
quality risk; STRATA_MMVQ_IL_Q8=1 re-arms; the tuned emitted table - the micro's real cell
wins vs the mirror - is the revisit path). The full arc is the law's named case: the first
armed leg read 92.6 t/s (a Chrome session at load 19 through the leg) - a fake -55%
"regression" that would have killed a bitwise win without the operator's catch.

The window-graph package: landed sound (compile + on-card legality probe) but INERT here -
the wrap gates on ar_on() (all-resident), which no 42-81-GB pack can reach on 32 GB; banked
as the enabler, with the doorbell-family flag-wait conversion named as the local path.
cond_node's micro deadlocks on its own protocol (spins on a flag the host sets later) -
tool bug, parked.

## The IQ4_XS gap, decomposed (2026-11-06, same-hour logs)

Conc-class windows (4.76 vs 4.70 rows): IQ4_XS 15.89 ms vs GSQ 13.59 ms (+17%). The
routing union is IDENTICAL (28.55 vs 28.81 VRAM hits/layer-window), the CPU-expert term
is identical (2.15 vs 2.05), the GPU-reach wait is identical (0.46 vs 0.42) - the entire
gap is GPU compute per hit (+21%/hit). The lever is the mixed-type expert kernel forms
(gu 18/23 x down 6/20/42), not routing, not tiers, not the miss path. The blob sizes are
2.38 MB (IQ4_XS layer 0) vs GSQ's 2.66 - per-expert bytes are SMALLER; the deficit is
kernel-form quality, not bytes. Deep study in flight (digest owed).

## P1 (Split<6>, the Q5_0 decode-once down) - verdict by law-17 interleave, NOT adopted (2026-11-06 night)

Parity: 0 failures, bitwise (the exact Q5_0 chain mirrored into load/apply). The 3+3
interleave on IQ4_XS (loads 3.5-8.2 recorded; NOTE the box was degraded - BOTH sides read
20-30% under the banked values, so only the RELATIVE deltas carry): solo +4.8 / pair +2.9 /
conc -11.2. The decode-once form wins serial windows and loses the 6-column case; law 13
forbids the lane loss. kSplit<6> back to false; the traits stay in-tree for the follow-up
design: a per-NC dispatch (multi for small-column windows, per-entry for conc) keeps both
sides' wins. The digest's P1 estimate (+4-7 all lanes) was half right - the serial half.
The quiet-box re-run of BOTH P1 forms + the per-NC dispatch is the next kernel window's
first item; P4 (config pins: --pcie-frac/--pcie-mode/--resident-experts parity with GSQ)
is config-only and rides the same window.
