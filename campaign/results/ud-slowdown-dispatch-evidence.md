# The UD slowdown, named by the engine's own counters (2026-11-06)

Instrument: STRATA_DISPATCH_REPORT on stock 0.1.57, identical traffic (1 warm + 1 conc
round) on both packs, clean-exit JSONs (dispatch-ud-iq2xs.json / dispatch-iq3s-gsq.json).

| fact | UD-IQ2_XS | IQ3_S (GSQ) |
|---|---|---|
| eager kernel launches | **2,262** (T1 999 / T2 686 / T3 272 / T4 147...) | **13** (T1 7 / T4 6) |
| hc read form | `weights Q8_0 copies=1` (a Q8_0 copy per read) | `copies=0` (bf16 rows direct) |
| final head | `q8=1` (Q8 form) | `q8=0` |
| experts path | `pcie_mode=2` | `pcie_mode=0` |
| window form | doorbell groups, all_resident=0 - but EAGER | same doorbell groups, CAPTURED |

Reading: the UD set's per-layer forms (the Q8_0-copies HC read forced by its Q6_K dense
side, the Q8 head, the mixed-type experts) fall OUTSIDE the graph-capturable set, so its
windows run one kernel launch at a time - ~175x the launch count of the same traffic on
IQ3_S. That launch overhead, plus the per-layer Q8_0 copy, is the 45% conc deficit (and
most of the solo deficit). The fix class: make the UD forms capturable (capture-safe
kernels) or convert at LOAD into the graph-friendly forms (Q6_K dense -> bf16 rows so
the HC read takes copies=0; preconvert the head) - load-time architecture support, the
thing GSQ has had for years.

## CORRECTION (same day, the kernel-study agent's counter reading, accepted)

The "windows run EAGER" line above misreads the counter: graph_launch(T, kind) fires only
on CAPTURED-graph paths - the 2,262 launches are 2,262 captured window GRAPHS, and the
T1/T2/T3 dominance means the UD run's windows averaged T ~1.96 ROWS (acceptance collapse),
not per-kernel launches. The mechanism the histogram actually names: the stock setup's UD
config drafts with the FULL 106,299-token vocabulary (--mtp data/mtp/rt) while every GSQ
config drafts the "en" 40,525-token subset (mtp-en/rt) - acceptance collapsed, the
~3.5 GB dense+HC fixed traffic amortized over ~2 rows instead of ~4.5 (~2.45 GB/token vs
1.57), which is the measured ~45%. The eager-vs-captured framing below is wrong; the
vocabulary finding is stronger AND config-only to fix. The HC Q8_0-copies and Q8-head
observations stand as secondary kernel-form deficits (see
.docs/research/digest/2026-11-06-ud-slowdown-kernels.md).
