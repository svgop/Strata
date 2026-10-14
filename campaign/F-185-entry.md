
### F-185 — R9 OVERLAP-UNBLOCK TESTED: rejected — the three "vestigial" arms are worth more than the F-67 overlap they block (2026-10-14 ~11:00, goal round 15)

**THE HYPOTHESIS (round-8 study).** The launcher kept ADAPT_LAG=2 / DMA_BATCH=1 /
VERIFY_DEVICE_PLAN=1 from the pre-v2 Windows battery; each trips the refusal chain
(generate.cpp) that disarms the adopted F-67 ADAPT_OVERLAP combo. Windows two-boot evidence had
the overlap at +4-9% concurrent. R9 = drop the three, arm the overlap.

**THE ARM ENGAGED CORRECTLY.** Boot line confirmed for the first time on this box: "adaptive
overlap: the adaptive tier's copies land while the next window runs" (vs the old refusal note).

**THE A/B (back-to-back boots, 3+3 probes, on the F-31-adopted recipe):**

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| R9: overlap armed (three arms dropped) | 160.6 | 167.8 | 193.4 | 79.5% |
| control: standing (three arms + BATCH_MTP=1) | **173.8** | **176.7** | **204.3** | 80.0% |
| Δ (R9 vs control) | **−7.6%** | **−5.0%** | **−5.3%** | flat |

**VERDICT: REJECT.** Arming the overlap costs 5-8% across the board HERE: what it disarms is worth
more than what it arms. Mechanism: STRATA_VERIFY_DEVICE_PLAN is the E-6 per-layer device plan —
the machinery that lets fully-resident layers skip the host round-trip entirely (round-9's budget
showed it covering ~half of layer-windows); STRATA_DMA_BATCH batches the swap submits; the
overlap's win (copies landing behind the next window, worth ~0.25 ms/window of adapt wait) is
smaller than their combined contribution. The Windows v2 battery's +4-9% was measured WITHOUT
these arms (different recipe, different balance); on the merged 0.1.44 line they are not vestigial
— they are load-bearing. Launcher restored to standing (the control IS standing); the armed
variant parked at /tmp/launch-r9-armed.sh for the record.

**THE BONUS OBSERVATION: the standing recipe's current-window means are the best concurrent
numbers of the whole session — pair 176.7 / 6-conc 204.3 / M1 173.8** (the drift is up; these are
window numbers, not anchor claims — but they show the F-31-adopted recipe's concurrent class:
pair +62% over the Windows bar of 108.8, 6-conc +46% over 140.2).

Sources: campaign/results/r9-ovl{1,2,3}.txt, r9-ctrl{1,2,3}.txt.
