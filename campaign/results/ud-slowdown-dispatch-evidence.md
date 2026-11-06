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
