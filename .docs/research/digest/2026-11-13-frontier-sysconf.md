# Frontier systems-conference harvest (2026-11-13) — OSDI/SOSP/ASPLOS/FAST/NSDI/EuroSys 24-26 vs our engine

Batch 3 final of the 18-agent frontier wave. ~40 papers screened; everything the first twelve digests cover
skipped. Venue corrections banked (Fiddler ICLR'25; Preble ICLR'25; AdaSpec=SoCC'25; AdaServe=EuroSys'26;
no Andes-KV paper; no Grand Slam; PowerInfer SOSP'24). Fun fact: OSDI'26 carries a paper literally named
"Strata" (Stanford/NVIDIA hierarchical context caching).

## The three genuinely new mechanism classes
(a) **TIER-AWARE ADMISSION** — Bidaw (FAST'26: -3.58x latency, storage-aware ordering + eviction
prediction), LMetric (OSDI'26: hyperparameter-free hit-rate x load admission score), JITServe (NSDI'26:
runtime-observed batch composition), Agentix (NSDI'26: program-level scheduling for agents). ALL FOUR land
in serve/server.py:1872 `_admit_next` — a combined S-M port, order-only, low risk.
(b) **CODED/TIER-SELECTIVE BYTES** — MoE-APEX (ASPLOS'26: jointly which experts offload + at what
precision; the tier-selective version of the TQ2_0 lever — price as ONE combined arm with the KL gate);
CacheGen (SIGCOMM'24: KV codec 3.5-4.3x smaller, 3.2-3.7x faster fetch — on the T66 disk tier).
(c) **SPEC-DECODE PER-REQUEST ECONOMICS** — AdaServe (EuroSys'26: +74% goodput, per-request draft trees
from drafter probabilities) + AdaSpec (SoCC'25: analytical acceptance/cost model, +66%) — the systems-venue
legitimization of the F-352-legal batch-lane adaptive-depth re-open; spec::CostModel exists, is not
load-aware.

## Other ranked findings
- Strata-OSDI'26 (5x TTFT): our PARTIALs are real (doorbell decouples GPU from host service; k==0 PLE
  gather = their compute/IO overlap) — the gaps are the decode KV-miss resolve (11.6% at 262K,
  synchronous) and tier-blind admission; arm KV_PREFETCH first (env-only).
- CacheSlide (FAST'26, position-shifted segment reuse 3-5.8x): the beyond-radix reference for agentic
  traffic (tool-result injections); the 80%-exists mount path + position-delta metadata; HIGH port.
- SolidAttention+ECHO (FAST'26/OSDI'26): selector-driven KV retention + lossless prefetch for NSA-class —
  our indexer scores are the unique input; retention policy KL-gated.
- Goodput instrumentation (DistServe's paired-SLO metric): we collect the TTFT half only; S port — the
  enabler for the admission family.
- ServerlessLLM/PPC-FAST'26 (loading trees, programmable page cache): hygiene rank on pack load.
- FineMoE (sub-expert offload): overlap-first (ADAPT_OVERLAP gap is the same class, cheaper).
- DynamoLLM (energy): honest negative for throughput (memory-bound lane; SM modulation buys watts);
  take only if power becomes an objective.

## Negatives banked (with reasons)
Llumnix/Parrot/TESSERACT/Splitwise/StreamEP/SwiftSpec (multi-instance/multi-GPU classes — single box, law
15); PowerInfer/Fiddler (our lane IS the mechanism, further); DirectKV (NVLink-C2C fabric class — on PCIe
our KV reads would compete with the expert lane for the same 50 GB/s); FastServe/MLFQ preemption
(BYIELD-cooperative keeps work; kill-and-recompute strictly worse); LLM-42 determinism (we hold the fast
path they retrofit: bit-exact position-keyed Philox); ServeGen (bench utility, mimicbench2 upgrade later);
unified-memory hardware classes (our shape is its own architecture); GB200/NVL72 studies (no new sm_120
fact).

## Cheapest sequence (their synthesis, folded)
goodput instrumentation -> KV_PREFETCH arming + tier-aware admission -> AdaSpec cost model on the batch
lane -> MoE-APEX/TQ2_0 combined dual-precision arm behind the KL gate.

Sources: DistServe OSDI'24; Parrot; Llumnix; ServerlessLLM + PPC FAST'26; Splitwise; CacheGen SIGCOMM'24;
Fiddler; PowerInfer; Preble; Strata OSDI'26; DirectKV; JITServe; Agentix NSDI'26; SolidAttention FAST'26;
Bidaw FAST'26; CacheSlide FAST'26; MoE-APEX ASPLOS'26; SwiftSpec ASPLOS'26; AdaServe EuroSys'26; AdaSpec
SoCC'25; LLM-42 SOSP'26; StreamEP SOSP'26; FineMoE EuroSys'26; DynamoLLM MICRO'24; Autellix; venue programs.
