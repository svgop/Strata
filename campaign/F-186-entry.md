
### F-186 — THE NEXT-LEVEL QUEUE v3: two frontier scans + R9 rejected + batch-6 adopted (2026-10-14 ~12:30, goal round 15 study)

**R9 (overlap-unblock) TESTED AND REJECTED** (F-185): arming the F-67 overlap costs 5-8% — the
three "vestigial" arms are load-bearing (VERIFY_DEVICE_PLAN = E-6 device planning over ~half the
layer-windows, DMA_BATCH, ADAPT_LAG). Round-9's "ungated resident share" item is REFRAMED: extend
E-6 to PARTIALLY resident layers (keep the flag, plan the resident share on device, gate only the
miss share) — do not remove it.

**BATCH WIDTH 6 ADOPTED** (F-185 addendum): the systems scan's top item — the 6-client probe was
slot-starved (queued 2 on 4). 6-conc 216.0 (+5.7%, the highest concurrent mean ever), pair flat,
slot acceptance 67.8%. In both configs.

**THE FORMULATIONS SCAN (ranked, full text in the session record):**
1. **FlashInfer SM120 stack** — fused GDN decode step for our exact model family (v0.7.0),
   persistent fused MoE with recompile-free decode-batch changes (v0.6.17), ragged-Q XQA for spec
   decode (v0.6.16, variable accepted lengths without padding — the "verify sublinear in rows"
   primitive for the 6-conc width problem). Port kernels, not ideas. [code]
2. **ECHO** (arXiv 2604.09603) — global verification budget across rows: deepen confident rows,
   truncate+reallocate lazy ones; sparse confidence gating at 3-4 calibrated depths; training-free;
   +8-14% at high concurrency over fixed EAGLE-3. Formalizes our 6-conc width problem. [code]
3. Routing-aware per-expert mixed precision (arXiv 2604.06515) — COUNTERINTUITIVE: RARE experts
   need the bits (router-norm-ranked), beating hot-boost (PMQ). Same-bytes quality; our
   expert-profile data already has the routing stats. [code+analysis]
4. DFlash v2 (KV-injection ablation: injection alone 4.8 acc vs 3.5 — applies to our MTP head via
   light retraining; full DFlash = new drafter). [training]
5. CSD calibrated rescue (training-free, LOSSY — conflicts our quality law; parked).
6. FlexDraft's finding: parallel draft+verify at batch<=2, sequential above — matches our
   pair-vs-6-conc split; the batch-adaptive strategy switch is the idea. [code]
TRAP CONFIRMED: 2026 spec papers report greedy-only; discount acceptance claims ~25-30% at temp 0.7
(OmniDraft's T=1 data: 1.57x→1.27x) — matches our Gumbel/min-p experience.

**THE SYSTEMS SCAN (ranked):**
1. Batch width (DONE — adopted above). Sentinel-slot graph bucketing (SGLang #27988) as the
   follow-on: fixed slot count + zero-length sentinels instead of per-layout graphs. [code]
2. DuetServe scheduler (arXiv 2511.04791): decode-first token budget + LOOK-AHEAD k-step graph
   launch (pre-launch next step's graph while current drains — attacks the 9.5 µs node spacing).
   [code]
3. Layered prefill (arXiv 2510.08055, MLSys'26): layer-group interleaving replaces token-chunk
   prefill — kills the per-chunk expert-reload tax (+39% off-chip traffic on MoE); up to 70% TTFT
   / 41% e2e. Prefill is a stated priority. [code]
4. Env: clock lock + power cap (TensorRT practice; Blackwell clock-transition hard-lock reports) —
   the R8 user step, now double-justified; driver 615.71.09 carries the Xid-109 fix with residual
   reports — keep the watch. [env/sudo]
5. CUDA 13.x grab bag: launch-completion events (submit next graph during drain),
   cudaMemcpyBatchAsync node collapse, CUDA 13.2 graph-host spin-wait; NVIDIA's llama.cpp trick:
   QKV as MULTIPLE STREAMS INSIDE ONE GRAPH (+35% MoE TG class — a node-density lever we have not
   exploited). [code]
6. sm_120 kernel truths: TMA supported (wins at prefill tile sizes, NOT decode gemv); clusters
   ≤8 supported but no gemv win; NO tcgen05/TMEM/wgmma/NVLink — datacenter Blackwell recipes do
   not port; 99KB smem, 48 warps/SM. DO-NOT-BOTHER list: tcgen05-class, DSMEM-gemv, NVFP4 requant,
   PD disaggregation, ik_llama offload, n-gram depth >2 (S=2 saturates on GDN hybrids, mlx-lm
   measured).

**THE OPEN LANE (both scans converge):** nobody ships a per-row entropy-gated arbiter between the
neural MTP chain and an S=2 n-gram drafter under sampled traffic with GDN-state rollback — all
components have 2026 production evidence (vLLM dynamic-spec control plane, mlx-lm's rollback
recipe, ECHO's budget math, our coupled verifier + per-slot state). A [code]-only project with a
credible 10-20% concurrent win.

**QUEUE v3:** R16 batch-8 probe (+ sentinel-slot graphs design) → R17 FlashInfer kernel ports
(fused GDN step first — our exact architecture) → R18 ECHO-style verify budget + ragged packing →
R19 layered prefill → R20 DuetServe look-ahead launch + launch-completion events + QKV-multistream-
in-graph → R21 FastMTP+KV-injection training → R22 rare-expert mixed precision (quality axis) →
R8 clocks (user step) whenever ready. Standing recipe now: merged 0.1.44 + BATCH_MTP=1 + batch 6.
