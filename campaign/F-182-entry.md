
### F-182 — THE BALANCE QUESTION: capacity is free, threads honestly idle, calibrate moot — the dynamic-context and CPU-utilization studies (2026-10-14 ~05:30, goal round 12)

**THE OPERATOR'S QUESTIONS.** (1) Scale to 262k for real use, 131k for baselines — agreed. (2) Can the
context window be dynamic — grow as needed, no fixed cost when small? (3) Should we leverage ≥80% of
CPU threads? (4) Does `--calibrate` give gains? Plus: why only ~6k experts cached?

**THE 6K-EXPERTS ANSWER.** The 1M boot's slot sessions carved **9.09 GiB** of the expert cache's
VRAM (262k: 3.79 GiB; 131k: 2.91 GiB) — that boot cached ~6k experts instead of 10,182. Not a fault;
capacity pricing in plain numbers.

**CAPACITY IS FREE AT SHORT USAGE — MEASURED BACK-TO-BACK.** 262k vs 131k, same night, same binary:
M1 177.0 vs 177.4 (−0.2%), pair 149.0 vs 150.2 (−0.8%), 6-conc 190.6 vs 193.2 (−1.3%) — statistical
zero (an apparent −9% earlier in the evening was the drift swing, both configs reading ~177 at that
hour vs the 01:00 peak of ~195). **The engine already does "declare big, pay per depth": streaming
KV means cost tracks USED context, not declared capacity.** The only fixed cost is the slot-session
arena (0.88 GiB going 131k→262k) — noise at our hit rates. The user's daily driver = the 262k
launcher; baselines stay 131k. (Third 131k control by a mis-fired pool-workers boot: 176.7/147.4/
189.3 — consistent.)

**TRUE DYNAMIC CONTEXT (rope-on-grow): NOBODY DOES IT, AND IT BREAKS.** The scan (kvcached/OSDI'26,
vAttention/ASPLOS'25, vLLM/SGLang/TRT-LLM, llama.cpp paged-KV RFC #21961, InfiniteHiP, InfLLM-V2)
is unambiguous: no engine re-scales rope mid-conversation — KV cached under the old encoding is
incompatible; silent corruption unless K is re-rotated (checkpoint/restart or 1.6×-prefill re-index
are the only shipped mechanisms). Within one rope setting, capacity-elasticity is solved two ways:
software paging (~3% overhead) or CUDA VMM (kvcached: reserve VA, map physical per use —
production-adopted by Google/LinkedIn/Red Hat). OUR [code] PATH: make `--kv-grow`'s VMM coexist
with the streaming+batch recipe (it currently refuses kv-resident/batch/resident-experts) — that
eliminates the slot-arena cost entirely and makes 1M cheap. Our streaming-KV + top-2051-index
design is validated as the published best practice (InfLLM/InfiniteHiP/TRT host-cache); our 173.9
t/s @244k is competitive with anything published on one GPU.

**80% CPU UTILIZATION: THE EVIDENCE SAYS NO — AND THE IDLE CORES ARE THE HONEST READING.** The
CPU-threads scan (KTransformers SOSP'25, Fiddler, fMoE, llama.cpp #8684 maintainer-confirmed,
#11765 3-cores-per-DDR-channel rule, TRT-LLM n-gram blog, vLLM v0.14 async scheduling): high CPU
utilization is the design goal of RESIDENCY-INVERTED systems (experts on CPU DRAM with AMX — we
have 98% VRAM residency, 2% CPU traffic, and Zen5 has no AMX). At GPU-bound offload "there is no
mechanism for helper threads to do useful work; they'd just spin" (llama.cpp maintainer). Our
15 workers are already ~2.5× past the 2-channel bandwidth saturation point (~6 threads) for expert
math. **What the cores CAN do (evidence-backed, queued):** (1) host-side n-gram drafting feeding
the same GPU verify — TRT-LLM measured +96% b1 / +64% b4 on cache-warm traffic, lossless, composes
with MTP — NOTE: the engine's PLE (n-gram embedding) path may already be a partial implementation;
investigate before building. (2) De-stall admission/orchestration (the 30-56 ms sync'd admissions =
2-5 windows of stall; matches round-9's fenced-admission item). (3) Prefill support roles
(tokenization, prefix-cache, staging). Chasing utilization for its own sake = spin-waiting or
demoting work from a 1792 GB/s device to a ~100 GB/s one — both measured/literature-confirmed net
negative.

**--calibrate: MOOT AND DANGEROUS.** It is a SETUP-harness auto-tuner (15-30 min; A/Bs
--pcie-frac, --spec-min-p, --pool-workers, adapt cadence; keeps >3% winners) — running
`setup.sh --calibrate` would rewrite the config args and DESTROY the standing recipe (Launch Law).
Its knobs: pcie-frac 0.35 and spec-min-p 0.5 are already campaign-measured optima (F-126,
tau-curve-1); **--pool-workers 8 was A/B'd tonight: null** (182.8/145.3/190.6 vs 177.4/150.2/193.2
control — mixed inside noise, as predicted at 2% CPU traffic; config reverted, standing verified);
adapt cadence stays queued (interacts with the learned profile). Nothing it could measure is worth
the config clobber.

Sources: `campaign/results/262k-short{1,2,3}.txt`, `131k-ctrl{1,2,3}.txt`, `pw8{,b}-{1,2,3}.txt`;
agent scans archived in the session record. No adoption; standing recipe unchanged and healthy.
