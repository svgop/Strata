
### F-176 — R7 CPU AFFINITY, THE AUX-CPUS ARM: auto and 16-23 both land inside noise — REVERTED (2026-10-13 ~23:20, goal round 7)

**THE HYPOTHESIS (R7 investigation, same evening).** The engine's host thread pins to CPU 0 and the 15
pool workers to cores 1-15, but every thread created *after* host pinning inherits the host's single-CPU
mask: 7 threads hard-stacked on CPU 0 (futex waiters, the adapt job thread, prefill helpers), and the
CUDA driver's threads float over all 32 CPUs including the workers' cores. The fork ships the fix:
`--aux-cpus` / `STRATA_AUX_CPUS` (`include/strata/platform/aux_cpus.hpp`), whose own header documents the
pathology on a 16-worker box ("the host thread was preempted 5x as often; the per-layer wait for the
host's flag was 60-155 us instead of 8"). The anchor log showed the plan computed (`CPUs 16`) but left
`off at start`. Placement-only change — "no output of the engine depends on where a thread runs" — so
no MMLU gate needed.

**THE TEST (operator-specified protocol: baseline probes → launcher edit → same probes).** Baseline =
3x probe-lane on the standing server (up since 21:33, warm; NOT fresh boot — recorded as a caveat).
Arms = one env line in `launch-custom-iq3s-8081.sh` + restart + 3x probe-lane each. Placement verified
per boot from /proc: R7a put 13 aux threads on mask=16 (the host's SMT sibling — the only CPU `auto`
could take, since all 16 physical cores carry host or worker); R7b spread 12 aux threads over 16-23
(the V-cache CCD siblings).

| arm (3-run means) | M1 | M2 pair | M2 6-conc | M3b |
|---|---|---|---|---|
| baseline (warm, no aux) | 173.7 | 153.2 | 194.6 | 81.4% |
| R7a `STRATA_AUX_CPUS=auto` (plan {16}) | 171.4 (−1.3%) | 146.6 (−4.3%) | 192.1 (−1.3%) | 80.3% |
| R7b `STRATA_AUX_CPUS=16-23` | 176.5 (+1.6%) | 149.3 (−2.5%) | 194.8 (+0.1%) | 81.1% |

Sources: `campaign/results/r7-aux{1,2,3}.txt`, `r7b-{1,2,3}.txt` (baseline numbers from the session log,
same probe, same night, back-to-back).

**THE VERDICT: no gain, honest negative, reverted.** R7a stacked all aux threads on one logical CPU that
SMT-shares the *busy host's* physical core — a small active regression (pair −4.3%, at the noise edge,
consistent direction on all three metrics). R7b fixed the pileup and recovered it, but every metric
stayed inside the ±4% noise band. The documented pathology is real on paper but not this box's
bottleneck: the aux threads are mostly blocked (futex/event waiters), and at 79-82% acceptance the
draft path — not host preemption — sets the rate. Launcher restored to the standing battery verbatim;
revert verified by the boot log reading `aux cpus: ... off at start`.

**TWO DRIFT OBSERVATIONS WORTH THE RECORD.** (1) Tonight's no-aux baseline (173.7/153.2/194.6, warm)
and the fresh-boot R7b arm both sit 4-9% below the locked anchor means (190.3/159.4/203.7) — the whole
evening drifted vs the anchor night independent of any arm. No conclusion drawn from it here (that is
what the Lock Law's matched-window rule is for), but the next adoption round should re-run the
anchor-protocol fresh boot before claiming anything against the anchor. (2) GPU IRQs (nvidia MSI-X 230/232)
land on CPUs 19/27 — SMT siblings of workers 3/11, scattered by irqbalance; that piece of R7 was never
in this arm (needs the sudo step, R7c).

**WHAT THIS CLOSES AND WHAT IT LEAVES.** Closed: the env-var aux-cpus arm in both forms — the cheap
half of R7 is done and negative; do not re-run without a new mechanism. Left open: R7c IRQ steering
(sudo, user's paste block) and R7d (taskset whole-process to V-cache CCD + `--pool-workers 7`), the
latter demoted — it takes real cores away from the tier and risks the pair/conc means. The queue
resumes at R5 (STRATA_SPEC_GUMBEL).
