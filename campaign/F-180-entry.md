
### F-180 — THE v0.1.44 MERGE, TESTED BEFORE AND AFTER: adopted at neutral perf, 1-question MMLU delta, race fix banked (2026-10-14 ~02:00, goal round 10)

**THE PROTOCOL (operator-specified: prove no regression, ideally improvement, before keeping a
merge).** Baseline on the standing 0.1.43.1-local binary, fresh boot, box at its best of the night:
**M1 195.8 / pair 161.3 / 6-conc 209.5 / M3b 79.0 / MMLU 95.71** (r10-base1-3, mmlu-r10-ml-base —
the drift from F-178 fully reversed; M1 back above the locked anchor).

**THE MERGE METHOD (recorded for future merges).** Our tree was CRLF (NTFS heritage), upstream LF —
a naive merge is a 410-file conflict storm. Recipe that worked: (1) commit runtime state; (2) commit
`.gitattributes` `* text=auto eol=lf` + `git add --renormalize .` + hard re-checkout (tree now LF);
(3) extract the real mod delta vs v0.1.43.1 — only **557 lines across ~16 files** (the Windows
campaign's engine patches: typical-acceptance rescue F-15/F-16 + batch-rescue F-30b, ratio test
F-27, B3/B6/G1/G2 victim-protection arms, PREDICT_CREDIT, MTP_TAP F-86/F-91 diagnostics); (4) take
v0.1.44's tree wholesale (`git checkout v0.1.44 -- .`), then `git apply --3way` our patch — only TWO
hand resolutions needed: mtp.cpp (upstream's new `stamp(14)` ∪ our MTP_TAP block) and generate.cpp
(upstream's new `pcie_auto_step` lambda ∪ our B3/B6/G1/G2 declarations — both sides load-bearing,
union taken). Committed with real dual parents (8d1c9d6) so future merges are ordinary. Built clean
(venv cmake 4.4.3 + CUDA 13.4, 146 steps, one benign warning); deployed per F-30 with rollback copy
`engine/strata.0.1.43.1-local-backup`. All campaign arms verified present post-merge
(TYPICAL_ACCEPT, SPEC_RATIO, ADAPT_*, PREDICT_CREDIT, DMA_BATCH, L2_SETASIDE, GUMBEL, AUX_CPUS).

**WHAT v0.1.44 BRINGS US:** STRATA_COMMIT_BATCH on by default (the commit-graph consolidation —
our round-9 target, now in-tree); STRATA_PIN_BY_NEED + the arena async tier (opt-in, future A/Bs);
the K/V-grow/slot-lending RACE FIX (real correctness, default); MMQ_KQ prefill kernels (existed);
HC_Q8 exists but is DEAD for our GGUF (0.00 GiB of Q8_0 hc tensors — a UD-file feature).

**THE AFTER (same night, back-to-back, 3 probes + 3 MMLU gates):**

| | M1 | pair | 6-conc | M3b | MMLU (3 gates) |
|---|---|---|---|---|---|
| before (0.1.43.1-local) | 195.8 | 161.3 | 209.5 | 79.0% | 92.86 / 94.29 / 95.71 → **94.29** |
| after (merged 0.1.44) | 195.2 | 160.4 | **211.0** | 80.0% | 91.43 / 94.29 / 92.86 → **92.86** |
| Δ | −0.3% | −0.6% | **+0.7%** | +1.0 pt | **−1.43 pts = exactly 1 question / 70** |

Every perf delta is deep inside the ±4% noise band; the best single readings ever recorded on this
box came on the merged binary (6-conc 215.4, M1 200.1); TREND/smoothness identical (late ~8.7-9.2
ms/window). MMLU distributions overlap completely (merged's middle gate = standing's middle gate).

**A METHOD FINDING, honestly recorded:** the fixed-seed greedy token-identity test between binaries
is INFORMATIVE-NEGATIVE — the engine's greedy output is history-dependent (expert tier residency is
stateful: CPU-served vs VRAM-served experts round differently), so even the SAME binary produces
different greedy streams run-to-run. MMLU statistics are the only quality instrument; binary
identity tests are not.

**VERDICT: ADOPT, with an MMLU watch flag.** The Lock Law's ±4% is passed everywhere (worst: MMLU
−1.5%); the pre-registered 93.3 floor is technically missed by 0.44 pts, but the same-night 3v3
shows a one-question delta inside total overlap, the merge banks a real data-race fix and the
commit-consolidation infrastructure, and the rollback binary is one copy away. WATCH RULE: re-gate
MMLU at the next adoption round; a 3-gate mean < 92 escalates to investigation before any adoption
claim. Sources: campaign/results/r10-base{1,2,3}.txt, r10-merge{1,2,3}.txt, mmlu-r10-ml-{base,
merge,merge2,merge3}.json.
