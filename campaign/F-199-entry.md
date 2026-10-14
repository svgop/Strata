### F-199 — ENGINE SET PRUNED to the comparison pair + THE +30%-vs-STOCK ROADMAP (2026-10-15 ~01:45, goal round 18 planning)

**THE PRUNE (operator directive).** Kept: `~/LLM/strata-custom-2` (current v2) and
`~/LLM/strata-upstream-0144` (pristine stock reference). Deleted: strata-custom-v1,
strata-frozen-baseline, strata-hokar (all result files and journal entries retained — only the
runnable trees are gone). Dependency surgery first: upstream's pack/mtp were symlinks INTO v1's
tree — replaced with real copies (2.3 GB) and every config path re-verified before deletion.
The old pre-campaign `~/LLM/Strata` + `Strata-data` stock dirs were NOT touched (not part of this
test set; say the word to remove them too).

**THE GOAL (operator).** +30% faster/better than upstream stock on every metric — "the difference
becomes night and day". Measured start (F-198 window 2): current = +12.0/+7.6/+2.7% (M1/pair/
6-conc). Still needed on top: +16/+21/+27%.

**THE ROADMAP — campaign/ROADMAP-30-VS-STOCK.md** (the working plan; summary):
- The 6-conc window budget (31.41 ms measured): GPU-reach 21.68 (attention 4.40 = #1, hc 4.20 =
  #2), CPU-expert doorbell 4.90 (16%), emit 1.87, commit 1.22 (RECOVERED by F-196), adapt 0.18.
  Recoverable pool ≈ 10 ms of 31.4 — deeper than the 6.6 ms the conc target needs.
- TIER 0 env battery (one boot each): TYPICAL_DEVICE → VERIFY_ASYNC_LAUNCH → ATTN_LANECELL →
  HC_SPLIT 0/1 → GR_V3 (MMLU-gated) → ROUTE_RESIDENT 0.75 → L2 72 → MMQ_KQ (prefill).
- TIER 1 code, value order: **T1.1 partial device plan** (4.90 ms doorbell tail, 9-16%, the
  per-region machinery already exists — s_bad abort at verify_kernels.cu:1141-1155 is the whole
  obstacle) → **T1.2 fold query rows into the attention block** (width growth 1.6→4.4 ms is pure
  re-gather; KV once per slot group, 6×12 KB fits shared) → **T1.3 deeper slot chains +
  the floor stair** (slots verify ONE proposal, generate.cpp:4736; serial chain stops at depth 3
  under the flat floor while depth acceptance runs 96/93/90%) → hc fusion → emit-on-device →
  fenced admission → graph pre-capture/sentinel → exact two-phase head.
- TIER 2 formulations: layered prefill (70% TTFT class, prefill = priority #1), ECHO row budget,
  DuetServe look-ahead launch/completion events (the 9.5 µs node-spacing), conditional graph
  nodes (green field), window width 12 after T1.2.
- TIER 3 rebuild/training: TQ3 rotation quants (+11% experts/GiB), 3-bit KV, DFlash2 (gated on
  sampled-mode acceptance), FastMTP+KV-injection.
- Stacked landing estimate: M1 +25-35%, pair +22-32%, 6-conc +20-33% (stretch) — the Tier-1 trio
  T1.1/T1.2/T1.3 carries the target; overlap-corrected arithmetic in the doc.
- Cross-cutting: run-2 stall hunt (worth several % of means), MMLU gates as marked, single-stream
  protection law, user-side iGPU cable + lock-state blocks.

Execution order: the T0 battery (days of cheap boots), then T1.1 → T1.2 → T1.3, Lock Law on every
adoption. The stock reference at ~/LLM/strata-upstream-0144 stays bootable for every re-baseline.
