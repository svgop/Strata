# Frontier quality/evaluation audit (2026-11-13) — llama.cpp KL workflow / lm-eval / LiveBench vs our gates

Batch 2 of the 18-agent frontier wave. Headline correction to the brief: the instruments EXIST
(tools/quant_quality.py, tools/kl_yardstick.py — top-32 restricted KL, F-230-anchored bands, prompt-path
scoring; frontier-grade). THE GATE DOES NOT: nothing invokes them at adoption, and the statistical gate that
does run cannot resolve what it decides.

## Ranked findings
1. **The KL yardstick is unwired and pins the serving path off** — compare() prints and exits 0; no
   thresholds in code; t20-gates never runs it; the PROVISIONAL bands (top-1>=97%/KL<=0.05) would reject
   EVERY pack we serve (IQ3_S: 87.69%/0.144; UD-Q6_K_XL: 94.61%/0.033; NVFP4 adopted at 90.5%/0.085) — band
   vs practice disagree; FIXED pins --kv fp16 + drafts off = the yardstick structurally cannot score the
   arms that change served outputs (KV/verify/routing/batch arms). Would have caught F-230, the NVFP4 band
   violation, HF-#33-class upstream corruption. FIX: wire into t20 with delta-vs-same-pack-baseline bands +
   one committed Q8_0 reference TSV per model revision. Low cost.
2. **The 70-question MMLU gate mathematically cannot resolve adoption-sized deltas** — at 94.3%: 1σ=2.77pp,
   Wilson 95% CI [86.2,97.8], MDE ~11pp at 80% power; temp 0.1 unseeded; past 1-3pp "wins" were coin flips
   (this INVALIDATES prior 1-3pp decisions more than it catches bugs). FIX: stratified 512-1,024q from the
   12,032q test split, temp 0, seeded; paired McNemar resolves ~1.5-2pp. 7-15x wall (still <1h).
3. **bit-check is 3 prompts with NO comparator** (manual eyeball); several live arms have no end-to-end
   quality instrument at all. The DEEPSEEK3 digit bug would have flipped bit-check p2's hash ("13:37",
   "84.5", "102.3") — if anything diffed it automatically. FIX: compare subcommand + 50-100 prompt bank +
   auto-run in t20. Low.
4. **CI has zero quality/parity gates** — serve/ unittests with a fake engine only; the parity binaries are
   hand-run; the tokenizer oracle that caught \p{M} exists only as a comment. FIX: CPU-runnable parity +
   vendored-ids tokenizer oracle in CI.
5. **Content-channel degeneration undetected live** — loop recovery exists (default off, reasoning-only);
   the spaced-digit class shipped to users. FIX: run coverage on the content channel, metric-only first;
   named detectors (digit-spacing, non-finite, emoji-run).
6. **Determinism envelope unstated** — bit-identity is valid only same-box/binary/batch-shape/config;
   cross-boot hash comparisons produce false alarms (the GSQ bank audit class). FIX: state the envelope;
   record config+batch with each hash.
7. **Quality has no bank** — perf has provenance-audited lanes.csv; quality is loose JSONs. FIX: kind=quality
   rows in lanes.csv (score, n, temp, floor, boot, commit, ref-TSV hash).
8. **No post-pack quality reference** — SHA pins prove download fidelity, not weight sanity; nothing pins
   the served pack to known-good after conversion. The committed reference TSV IS the pin. Note: Q8_0 ref is
   quant-on-quant (blind to shared defects); commit one FP16 root reference per family if RAM allows.
9. **Eval freshness** — static validation split contamination-blind; LiveBench-class rotation is the 2026
   practice. FIX: one rotating post-cutoff slice per cycle. mimicbench2 (self-authored) is fine by design.

## Already exceeding
Top-32 restricted KL + acceptance-anchored bands (nobody public ties KL to spec-acceptance cost like F-230);
prompt-path scorer (public QA measures decode only); bitwise parity with NEGATIVE CONTROLS ("a pass without a
failing disabled-control means nothing" — rare discipline); tokenizer-as-reference with the \p{M} oracle
lesson; golden hash on every perf probe for free; loop RECOVERY not just detection; mimicbench2's agent-depth
axes; per-request floor transport (the matched-window discipline applied to quality).

## One-line synthesis
Instruments frontier-grade, process is not: wire what exists into t20 with recalibrated bands + the 512q/
temp-0 upgrade closes the two largest gaps at near-zero new engineering.

Sources: llama.cpp perplexity README + PR #5076; Unsloth dynamic-v2/3.0; GSQ paper + ISTA-DASLab packs;
lm-eval-harness #1947; MMLU-Pro paper; LiveBench (+paper); Artificial Analysis LiveCodeBench; Thinking
Machines nondeterminism; production golden-set + drift-monitoring references.
