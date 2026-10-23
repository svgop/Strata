# T37 — PATHWAYS & MEMORY: the milliseconds between tiers (host↔device pathways, OS/host scheduling, cold-start, launch overhead, stability)

Date: 2026-10-23. Researcher C, wave T37. READ-ONLY session: no boots, no builds, no probes; the only
write is this file. HEAD = bf746391 ("features doc: the lifecycle era at full depth"). Consumed, not
repeated: T35-memory-hierarchy, T35-elastic-refinement, T35-serving-latency, T26-6, T26-3, T26-1,
T27-1, T27-3, T33, F-191, F-232, F-251, F-259..F-266, F-275, F-276. Every internal claim carries a
file:line or log:line read THIS SESSION at HEAD; external claims carry numbered sources with a
confidence grade ([MEASURED-OURS] / [CORROBORATED] / [HYPOTHESIS]).

---

## §1 OUR-STATE VERIFICATION (anchors at HEAD bf746391, logs re-read)

**The lifecycle era is standing exactly as F-276 banked it.** Launcher
(`launch-custom-iq3s-8081.sh:25-30`): `STRATA_ELASTIC_SESSIONS=1`, `STRATA_ELASTIC_PARK=1`,
`STRATA_ELASTIC_WATERMARK=1`, `STRATA_ELASTIC_TIMED_SCAN=1`, `STRATA_VMM_CHUNK_MIB=32`,
`STRATA_SERVE_TELEMETRY=1`. Boot line confirms all five (`custom-iq3_s.log:32685`): "chunk unit
32.0 MiB (STRATA_VMM_CHUNK_MIB; 24 driver calls per arena moved), watermark on (24 chunks, 0.75 GiB
kept in the spare), timed scan on, park-on-demote on". Demote lines at HEAD show the adopted anatomy:
"demoted in 46.3 ms (park 44.2, unmap 2.1, 24 chunks to the spare pool …)" (:31282-class; second
sample park 25.4/unmap 0.4); restores "restored 84 tokens (live) in 9.4 ms" — the 9-15 ms contract
holds. Complement: "mapped pinned cache complement ready: resident 36.82 GiB … in 8 s" (:31225).
Batch anatomy on the newest boot: "13.11 ms/window = run 11.82 (GPU-reach wait 0.54 + CPU experts
2.21) + commit 0.34 + emit 0.35 + adapt wait 0.60; … 319.5 rows/s; slot drafts accepted 69.1%" at
avg 4.99 rows (:32637-class line). All scoreboard numbers in the assignment reconcile with F-276
(M1 215.2/228.3 best, pair 205.3, conc 317.1) and with T26-6 (cold 272.0 → warm 299.5 wall).

**Corrections of prior-dossier claims banked this session (the T35 tradition — four more):**

1. **T26-1's unknown #1 (attn_gate types) is RESOLVED — mid_chain is unconditionally unblocked.**
   Read directly from the GGUF v3 headers of both source shards
   (`/home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`, 1,223
   tensors in shard 1): **attn_gate.weight = 10 Q4_K + 14 Q6_K + 4 Q5_K + 8 IQ4_XS** (36 layers) —
   every one inside basalt `mid_chain.cu:444` `type_ok {12,13,14,21,23}` (= Q4_K/Q5_K/Q6_K/IQ3_S/
   IQ4_XS; enum verified at `third_party/llama.cpp/ggml/include/ggml.h:398-413`). The same parser
   reproduced F-251's qkv (22 Q6_K/10 Q5_K/2 Q4_K/2 IQ4_XS) and ssm_out (29 Q6_K/4 Q5_K/3 Q4_K)
   exactly — independent confirmation of both F-251 and the new read. The pack `index.txt` type
   column (8 = Q8_0) is confirmed misleading for the dense tensors, the same F-251 index-column
   misread class. **The generic triple runs on every GDN layer today; no re-quant exists.**
2. **The assignment's "PCIe 5 at --pcie-frac 0.55 (adopted, +2.8% M1 pooled over 16 legs)"
   contradicts the record and the config.** The standing config reads **0.35**
   (`custom-iq3_s-8081.json:28-29`), and the ledger is unambiguous: adopted F-259
   (SVG-CUSTOM.md:8800-8815), falsified by the F-263 FINAL proof battery and reverted
   (:8927, :8945 — "frac 0.55" in the nine-falsification night ledger), consolidated into the
   dead-end removals at F-265 (:8974). T35-memory-hierarchy's own hierarchy diagram carries 0.35.
   Nothing after F-265 re-adopts it. "PCIe shaping past 0.55" dissolves as an item (see §3 V8).
3. **UseNodePriority verified MISSING at HEAD; anchors refreshed.** No
   `cudaGraphInstantiateFlagUseNodePriority` anywhere in `src/`; the three instantiate sites pass 0
   (`verify.cpp:2776` commit, `:3738`, `:3745` window/batch — T27-1's :2628/:3588/:3595 have
   drifted). Priority streams exist (`verify.cpp:1016-1029`,
   `cudaStreamCreateWithPriority(..., top)`). The XS arm stands unchanged in substance.
4. **The eunomia figures are real but were over-transferred.** Fetched the primary
   (eunomia.dev, "When CPU Noise Slows Down GPU Inference", 2026-05-24) this session: the **15.3 ms
   median penalty applies ONLY to the 62 of ~86k launch pairs (0.1%) whose interval contained a
   context switch, on a NON-graph engine emitting ~950 launches/token** (qwen3.cu, FP32 GGUF,
   24-core host, kernel 6.15). Overall scheduler impact in their clean run: 1.2% of 79.5 s. The
   churn-class numbers DO transfer as box-level cases: stress-ng CPU load −8.8%, iperf3 −2.8%,
   fio −0.3%, combined −20.5%; taskset 0-3 + nice −10 recovered +7.6 of the 8.8 points (residual
   1.9%), context switches −96.3%. **Our engine replays a whole window as ONE graph launch (~2-3
   host launches/window) — the per-launch-pair exposure is ~10³× lower than their workload; the
   correct mapping of their mechanism onto us is per-WINDOW: a context switch inside the service
   loop (GPU-reach spin, doorbell poll, staging) or between graph replays lands 1:1 in a 13 ms
   window.** The lever is real; the magnitude band must be re-derived from window-gap telemetry,
   not from their 15.3 ms.

**Cold-start anatomy re-anchored at HEAD** (`expert_source.cpp`): `pin_depends_on_reclaim` at :101
(`free_now < bytes + reserve`); `drop_file_cache` lambda at :3484-3491 (madvise(MADV_DONTNEED) on
the engine's own shard mappings + `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` whole-file);
fired at :3519-3521 under `STRATA_PIN_GUARD` default 1 (:3511-3514); **guard=2 sets `paced=true`
and SKIPS the drop entirely** (:3516-3518) — paced registration reclaims incrementally via kswapd
instead; the paced path takes 2 MiB pages (:3576-3579, huge_alloc), the one-shot path keeps
cudaHostAlloc with the printed note (:3535-3543). Cold-boot cost re-read from the log: a drop boot
pays "complement ready … in 20 s" (:14376; warm boots 8 s, a second sample 18 s at :28028), then
"file tier I/O this request: the OS read **38,602 MB** from storage (1 major fault) for 22,137 MB
of expert reads … decode waited **1,984.4 ms** in 10,863 fetches" on request 1, with every later
request at 0.0 MB (:14379-14381). Note the arithmetic: 4,804 MB/s × 4.5 s ≈ 22.1 GB = the
profile-prefill read of that boot — **the cold re-read lands substantially INSIDE the boot (GPU-
cache prefill at NVMe speed + the complement copy), with request 1 absorbing the I/O tail**; the
drop released 89-91.6 GiB of cache to free 35.9-40.5 GiB really free (:14374, :16080, :28021).
THP state on this box, read this session: `enabled = [madvise]`, **`defrag = [madvise]`**,
HugePages_Total=0, swappiness 10, 1 NUMA node, V-cache CCD = CPUs 0-7/16-23 (98,304K L3), fast CCD
= 8-15/24-31 (32,768K) — `/sys/devices/system/cpu/cpu{0,8,16,24}/cache/index3` read directly;
matches the assignment and T35.

**PDL surface at HEAD**: 37 PDL references across exactly 4 files (dense_tc.cu 16, native_mmvq.cu
10, native_bf16.cu 8, verify_kernels.cu 3 — grep counts). The gate is per-device + per-kernel
PTX-checked (`pdl_supported`, verify_kernels.cu:1976-2000) and the capture side already enforces
kernel→kernel-only programmatic edges via the capture-info query (`pdl_launch_ok`, :2003-2050;
CUDA 13 path at :2034-2037). `STRATA_DF_PDL` remains unset in the launcher → standing value 0.
Our GDN decode spine at HEAD (`verify.cpp:1522-1624`): qkv `mm` (:1522/:1538) +
`gdn_conv_ab_multi` (:1553-1557, the 143-amdk conv+SiLU+L2+a/b fold) + z `mm` (:1571) +
`gdn_step_norm_multi` with QFUSE q8_1 (:1613-1618) + [`quant_in` only on the batch_rec path,
:1622] + out `mm` (:1624) ≈ **5-6 critical-stream nodes/layer**. Diet gates confirmed in-tree and
OFF: `STRATA_LFUSE` (verify.cpp:97), `STRATA_FUSE_WAIT` (:317-346), `STRATA_DF_PLE`,
`STRATA_DUMMY_NODES` (:1431, the per-node-cost instrument), `STRATA_AUX_CPUS` (generate.cpp:602,
1103). WSTAT emission at `generate.cpp:11766`; telemetry plumbing at prometheus.py:132-136 and
:204-213, server.py:968-969/:1536/:2078. Telemetry counters live (log + t36-lifestic.log:
"[metrics …] TTFT p50/p99 375ms/2395ms ITL p99 35ms … golden dbb294a9 … clock 2955MHz").

---

## §2 EXTERNAL SWEEP (graded; primary source on every claim)

**Cold-start / page cache**
- [CORROBORATED] `posix_fadvise(2)` (man7, current 2026): DONTNEED frees cached pages **for the
  specified range** (page granularity); WILLNEED is a *non-blocking* readahead that the kernel may
  shrink under pressure. The whole-file form our drop uses (`fadvise(fd,0,0)`) is the broadest
  possible scope; range-scoped drops are first-class API.
  https://man7.org/linux/man-pages/man2/posix_fadvise.2.html
- [CORROBORATED] fadvise+mincore selective-drop practice (oetiker/insights; glandium's drop-rate
  caveats; borg #907's page-alignment gotcha): partial-range drops and cache-preserving drop loops
  are an established pattern. https://insights.oetiker.ch/linux/fadvise ;
  https://glandium.org/blog/?p=1262 ; https://github.com/borgbackup/borg/issues/907
- [CORROBORATED] THP admin docs (kernel.org): THP backing modes; `defrag` (not `enabled`) is what
  decides synchronous-compaction behavior on fault.
  https://docs.kernel.org/admin-guide/mm/transhuge.html
- [MEASURED-OURS (family)] Strata issue #771 (engine 0.1.39→0.1.40): MADV_HUGEPAGE under
  `defrag=madvise` + fragmented RAM put every arena-fill fault into synchronous compaction —
  **startup 25 s → 432 s** (~18,304 faults/s at ~50 µs kernel work, compact_fail==compact_stall);
  isolated A/B: plain mmap fill 4,440 MB/s vs 216 MB/s with the madvise. Fixed by not requesting
  MADV_HUGEPAGE when unnecessary + `STRATA_NO_ARENA_THP`; the thread's follow-up notes the fix is
  incomplete under `enabled=madvise` and proposes **fill-then-MADV_COLLAPSE (Linux 6.1+)**.
  https://github.com/Niko1221/Strata/issues/771 — **this box runs defrag=[madvise]** (read this
  session), so the condition is live here for any hugepage-on-fault path.
- [CORROBORATED] vLLM cold-start decomposition (arXiv:2606.07362, sleep-mode blog, Run:ai
  streamer) — carried from T27-3 A9: our prewarm2 shape (cache re-warm + capture-forcing replay)
  is the converged remedy for a dedicated single-model box; nothing cheaper has appeared.
- [HYPOTHESIS → falsifiable] vLLM #40807: copies captured into CUDA graphs require pinned host
  tensors — our graphs already bake only device pointers (the F-272 capture audit), so the class
  does not bind. Recorded as a watched constraint on any future host-visible node in graphs.
  https://github.com/vllm-project/vllm/issues/40807

**Host scheduling / CCD placement**
- [MEASURED (primary, theirs)] eunomia CPU-noise study — figures and rescoping in §1.c4.
  https://eunomia.dev/blog/2026/05/24/measuring-cpu-noise-in-gpu-inference-with-ebpf
- [CORROBORATED] gpu_ext (arXiv:2512.12615): eBPF hooks in the GPU launch path cut LC-launch p99
  95% under GPU-side timeslice contention (2 LC + 4 BE processes); hook overhead <0.2%; hosts
  include an RTX 5090. Relevant as (a) confirmation that launch-path delay is a first-class loss
  class, (b) the GPU-side scheduler is a different layer than our CFS problem.
  https://arxiv.org/html/2512.12615v1
- [CORROBORATED] CPU-induced slowdowns in multi-GPU LLM inference (arXiv:2603.22774): CPU
  starvation (tokenization, control plane, launch serialization) costs TTFT 1.36-5.40x on starved
  configs; adding cores fixes what pinning cannot when cores are insufficient — on OUR box cores
  are abundant (16 SMT threads idle), so the transferable finding is their §VI-A note that
  pinning/affinity/isolation are the right tools **when cores suffice**, which is our case.
  https://arxiv.org/html/2603.22774v1
- [CORROBORATED] PeeR (OSDI'26) — eBPF preemption for latency-critical host programs; class
  confirmation only. https://www.usenix.org/system/files/osdi26-carin.pdf
- [CORROBORATED] IRQ-affinity/isolation practice: NVIDIA forum isolcpus thread + the eunomia
  follow-ups (isolcpus/nohz_full/IRQ smp_affinity as the escalation ladder).
  https://forums.developer.nvidia.com/t/how-to-use-cpu-isolation-to-reduce-network-latency/206672

**Launch overhead / PDL / graphs**
- [MEASURED (primary)] CUDA PDL programming guide (v13.4.2, current): PDL works in graphs via
  stream capture (programmatic edges, port `cudaGraphKernelNodePortProgrammatic`) or edge data;
  `cudaGridDependencySynchronize` blocks until primary results are **flushed**; the doc's explicit
  correctness warning — secondary blocks "might launch before data written by the primary kernel
  is visible" and MUST use `cudaGridDependencySynchronize` — is exactly the TRT-LLM quant-kernel
  NaN class; sm_90+ required. No performance numbers promised by the doc.
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- [CORROBORATED] TRT-LLM PDL casebook (carried T27-1, re-cited): PDL default-on since PR #9695
  with a per-kernel WAR-disable list (quant kernels, TinyGEMM, trtllm-gen routing NaN, dsv3 MoE)
  and an on-disk per-kernel accuracy record — the discipline our kill-switch design copies.
- [CORROBORATED] UseNodePriority=8 semantics (CUDA runtime docs, carried T27-1): per-node priority
  replaces the launching stream's priority inside the graph.
- [CORROBORATED] Event Tensor (MLSys'26, carried T27-1): non-graph kernel boundaries cost 5-10 µs;
  our graphs already avoid that class; inside graphs the levers are node count and edge type.
  https://arxiv.org/abs/2604.13327
- [CORROBORATED] llama.cpp PDL adoption discussion (Aug 2025): practical PDL for a
  llama-family engine exists upstream of us; nothing transferable beyond confirmation.
  https://github.com/ggml-org/llama.cpp (issue thread "Programmatic Dependent Launch", 2025-08)
- [MEASURED-OURS] F-191 (SVG-CUSTOM.md:6446-6464): the ABAB that closed PDL as a no-op — and the
  resolution this session: at F-191 the armed surface was 4 files/~25-40 nodes of a ~2,050-node
  window; T26-3 (c) priced a partial chain at −0.01..−0.05 ms, **inside F-191's own ±1.2% noise
  floor**. F-191 was a correct measurement of a too-partial chain, not of the chain band.

**Driver / VMM**
- [MEASURED-OURS] 29-51 µs per cuMemMap-class op on this GSP driver (F-276; T35). [CORROBORATED]
  No batched VMM map API exists through CUDA 13.4 (toolkit release notes scanned this session;
  CUDA 13.0's additions were host-side cuMemCreate/cudaMallocAsync + batched *memcpy* error
  reporting, not batched map). Op-count reduction remains the only cure; 32 MiB superchunks is the
  standing form. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- [CORROBORATED] vAttention v3 ~40 µs/call, no batched API (carried T35 S10) — consistent.

**Memory-tier refinements**
- [MEASURED (primary)] **SplitZip** (arXiv:2605.01708, 2026): lossless KV compression for PD
  transfer — BF16 **1.324×** (exponent-stream fixed-length coding), FP8 **1.136×** (E5M2 top-16) /
  **0.933× (expansion)** on E4M3 top-8; the paper's own bound: "FP8 offers less redundancy than
  BF16 because the original representation is already smaller… compression ratio is also bounded
  by the redundancy in floating-point exponent values". **No integer-KV results; int8 quantized
  codes have no exponent field at all** — the redundancy class SplitZip exploits does not exist in
  our parked int8 KV. https://arxiv.org/html/2605.01708v1
- [CORROBORATED] The lossless-KV field population (Ecco: group-wise non-uniform quant + Huffman;
  speculative-KV-entropy-coding blog ~4× on fp16-class KV; CacheGen 3.5-4.3× carried from T35) —
  every verified member either (a) compresses fp16/fp8 KV, or (b) changes what compute reads
  (F-230-gated). https://dl.acm.org/doi/10.1145/3695053.3731024 ;
  https://fergusfinn.com/blog/kv-entropy-coder
- [CORROBORATED] Hugepage/TLB mechanism on the host path (Farshin IOTLB wall, carried T35 S9;
  kernel THP docs above; nelhage's THP-regression caution) — 2 MiB pages recover TLB-miss class
  losses; THP is not free under fragmentation (#771 is the measured family instance).
- [CORROBORATED] PCIe transfer shaping 2025-26: the field's PCIe work is multi-GPU EP comm
  (arXiv:2609.40093, expert-parallel on PCIe); single-GPU mapped-pinned miss paths are unchanged
  since the DMA_BATCH/overlap classics (NVIDIA overlap-transfers blog remains the canonical
  mechanism; pinned memory is the prerequisite everywhere). Nothing new to import.
  https://arxiv.org/html/2609.40093v1 ; https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc

**Telemetry practice**
- [CORROBORATED] vLLM metrics semantics (ITL vs TPOT; histogram buckets) + 2026 production
  monitoring practice (PromQL alert recipes on latency histograms; p99-spike attribution to
  scheduling/queueing). Our /metrics already exports the vLLM-named histograms plus the true
  per-token gap — the gap between our instrument and vLLM practice is alarm rules, not metrics.
  https://docs.vllm.ai/en/latest/design/metrics ; https://akrisanov.com/vllm-metrics ;
  https://www.digitalocean.com/community/conceptual-articles/when-your-vllm-p99 ;
  https://clickhouse.com/resources/engineering/llm-inference-latency

Source count: ~34 external+family sources across 4 categories (specs/docs; literature;
code+history incl. our logs, GGUF headers, issue #771; practitioner) plus the internal corpus.

---

## §3 IMPLEMENTER-READY CANDIDATES, VALUE-RANKED

Each: mechanism → expected band (arithmetic) → cost-anatomy read (F-232 law) → falsifier-first
design → composition + law-4b/9 scoping → rollback. Sub-4% candidates ride the five-boot
quiet-window law; event-class candidates (V1, V5) gate on event telemetry, not means.

### V1. Host isolation of the serving pair (zero-code) + the CCD_SPLIT redo — the variance lever

- **Mechanism.** Eunomia-verified (primary, theirs): a context switch in a launch interval leaves
  the GPU idle and the delay lands whole in the next completion; taskset+nice recovered 7.6/8.8
  points under CPU churn (§1.c4). Our shape: the service loop's host thread spin-waits GPU-reach
  0.54 ms/window mean and polls doorbells; server.py parses every line; one 15 ms descheduling
  event inside a 13.11 ms window doubles it and prints as a WSTAT max / token-gap outlier. The
  box is a desktop (co-resident browser/IDE = the combined-churn case, their −20.5%).
- **Recipe (9950X3D, verified topology: V-cache CCD 0-7/16-23 96 MiB, fast CCD 8-15/24-31 32 MiB).**
  1. Engine-internal split: `STRATA_CCD_SPLIT=1` (in-tree, pool.cpp:467-551) — the 15 GEMV pool
     workers move to the fast-CCD primaries (8-15; they stream DRAM, are cache-agnostic), the host
     thread to CPU 0 on the V-cache CCD (spin/park flags stay hot in 96 MiB L3). Reads /sys, says
     so, falls back safely.
  2. Aux + server: `--aux-cpus 16-23` (generate.cpp:602 — non-pool non-host threads) and
     `taskset -c 16-23` on server.py, so the Python loop never shares a core with the host spin
     thread or the desktop's preferred cores.
  3. Priority: `renice -n -5` on the serving pair (negative nice needs privileges: sudo-list, or
     systemd-run --user with CPUWeight as the unprivileged form). Escalations (only if the
     falsifier shows residue): GPU/NVMe MSI-X IRQ affinity to fast-CCD siblings 24-31
     (/proc/irq/*/smp_affinity_list, sudo), then isolcpus/nohz_full (kernel cmdline, reboot).
  4. Leave rtkit/RLIMIT_RTTIME alone for now (SCHED_FIFO is the last rung; our boot log already
     carries the rtkit warning — T35-serving-latency's observation stands).
- **Expected band.** Honest: variance reduction, not a mean shift. Clean-idle box ≈ 0; under real
  desktop activity the bound is eunomia's churn class (their −8.8% single-source, −20.5% combined)
  applied to our ~28 ms round and 13 ms window. Detect at the event level: token_gap p99,
  window_ms_p99, TREND early/late spread — never means.
- **Cost anatomy (F-232).** Done: pool.cpp:467-551 (the split is placement-only, "no output bit
  depends on where a thread runs" — comment :472); aux wiring generate.cpp:5926-5943-class;
  verify.cpp:1016-1029 (priority streams already highest). No new code for legs 1-3.
- **Falsifier-first.** One 3v3 pinned-vs-standing **under a controlled churn generator** (4×
  stress-ng threads + a browser-class load) reading token_gap_seconds p99, WSTAT p99/max, and
  admission_wait p99. If gap p99 does not separate under churn, the lever is dead on this box
  (desktop too quiet) — killed for one boot's cost.
- **Composition + laws.** Composes with everything (placement-only). Law 9: pure REMOVE (deletes
  waits caused by co-scheduling; no mechanism added). Law 4b: scoped by construction — the arm
  pays only where churn exists. **CCD_SPLIT battery redo priority: SECOND** — the label bug that
  ate the isolation leg is engine-side and needs the battery; the zero-code legs above price the
  same mechanism family first, and the redo then composes on top. Rollback: `taskset`/`renice`
  unpinned; `STRATA_CCD_SPLIT=0`.
- **Grade.** Mechanism [CORROBORATED (primary)]; our magnitude [HYPOTHESIS until the churn 3v3].

### V2. PDL: the per-pair kill switch + accuracy record, then the chain re-arm — F-191 resolved

- **Mechanism.** Two facts re-open PDL. (a) F-191's null is now explained: the armed surface was
  4 files / ~25-40 of ~2,050 window nodes → priced −0.01..−0.05 ms, below F-191's own ±1.2%
  verdict floor (T26-3 (c) partial-chain arithmetic + this session's 37-reference grep). The
  measured no-op was a measurement of a fragment. (b) The window changed since (COHORTS, dm fold,
  async launch, lifecycle): inside a captured graph, node-to-node handoff is the graph scheduler's
  per-node serialization — our measured marginal 0.48-0.95 µs/node — and programmatic edges skip
  exactly that, which is why diet and PDL multiply (T26-3 (e) sequencing).
- **Design (adopt before extending the chain).**
  1. Per-family kill switch: extend the STRATA_DF_PDL gate with `STRATA_DF_PDL_OFF=<families>`
     consulted in `pdl_launch_ok`/`pdl_supported` per kernel family (the per-kernel PTX check at
     verify_kernels.cu:2003-2050 is the natural hook; a name→family map is ~20 lines). Quant-class
     pairs (native_mmvq's quantize_q8_1 :2462-class, gu_swiglu q8 stores) are WAR-listed by
     default — TRT-LLM disabled exactly this class after real NaN regressions, and the CUDA doc's
     visibility warning (:§2 primary) is the mechanism.
  2. Accuracy record: per-pair golden hash + M3b at every armed boot, appended to a checked-in
     `campaign/pdl-accuracy.json` (family, date, hash, M3b, verdict) — the TRT-LLM protocol.
  3. nsys overlap probe per pair is mandatory (the GB10 no-overlap precedent — PDL behavior is
     per-chip; carried T27-1 A3.6).
- **Expected band.** Extension of 2-3 pairs: −0.02..−0.1 ms/window (sub-noise alone — the point
  of legs 1-2 is discipline, not speed). Full chain (12 more files, post-diet): −0.5..−1.5
  ms/window ≈ +2..+6% M1 [HYPOTHESIS, the T25-5/T26-3 envelope re-affirmed].
- **Cost anatomy.** verify_kernels.cu:1976-2050 (gate + capture-edge law) read this session; the
  37-site inventory per file; F-191's own table (SVG-CUSTOM.md:6446). Done.
- **Falsifier-first.** The kill switch itself is exercised free by the existing 2-3-pair + nsys
  falsifier (T25-5 T-1, never run). Cheapest kill of the RE-PRICE: arm STRATA_DF_PDL=1 once on
  today's standing recipe with the nsys probe — if inter-kernel overlap is still zero on the
  37 armed edges (driver/chip refusal), the whole chain dies for one boot.
- **Composition + laws.** Rides the diet (V3) — sequence tranche-1 gates first so the chain has
  more contiguous kernel→kernel spans. Law 9: PDL is a FOLD of launch-tail waits into execution;
  the kill switch keeps every fold revertible per family. Law 10: M3b + golden hash per boot.
  Rollback: `STRATA_DF_PDL=` unset (global) or `STRATA_DF_PDL_OFF=family` (surgical).
- **Grade.** Partial-chain explanation of F-191 [MEASURED-OURS arithmetic]; the chain band
  [HYPOTHESIS].

### V3. Launch-map diet phase 2, tranche 1: the three in-tree gates + UseNodePriority

- **Mechanism.** F-204's top-10 remains the phase-2 spec; the adopted work (FUSE_HITS, POOL_DIET,
  dm fold F-270, norm_fast) consumed item 9's core and part of 5. What remains ON OUR GRAPHS at
  HEAD: (1) standalone q8_1 quantize nodes (QFUSE covers the single-session form; batch_rec still
  quantizes — verify.cpp:1620-1623), (2) the shared-expert chain 6→3 (`STRATA_LFUSE`, verify.cpp:97,
  implemented + bitwise-checked per F-204, never armed on CUDA), (3) separate wait+copy doorbell
  forms (`STRATA_FUSE_WAIT`, verify.cpp:317-346: HIP fused, CUDA behind the env, F-205 un-gated),
  (4) router chain 4→2, (5) hc read residue, (6) the batch-window QSA per-row suite (the −330 S=6
  prize; needs new kernels), (7) PLE row loop (`STRATA_DF_PLE`, in-tree, bitwise-equal), (8) QSA
  small-pairs, (10) misc. UseNodePriority: still missing (§1.c3).
- **Tranche 1 = four one-boot falsifiable arms, zero new kernels:** LFUSE (−144 nodes/window),
  FUSE_WAIT (−144), DF_PLE (−56), UseNodePriority flag 8 on the three instantiate sites
  (verify.cpp:2776, :3738, :3745; ~3 lines). Node-term arithmetic at the measured marginal
  (0.48-0.95 µs/node): −344 nodes ≈ **−0.17..−0.33 ms/window**, plus each arm's execution delta
  (LFUSE deletes side-stream fork/join ceremony; FUSE_WAIT deletes a round trip per group).
- **Cost anatomy.** verify.cpp:97/:317-346 (gate shapes), :1431 (STRATA_DUMMY_NODES — the
  per-node-cost instrument for the A/B), :2776/:3738/:3745 (flags=0), F-204 inventory (read this
  session). One gap flagged: F-204's node deltas were counted at the pre-dm graph; re-count per
  arm from the capture listing (`STRATA_VERIFY_NODES=2` exists, HIP-side listing noted at :2539)
  before banking any delta.
- **Falsifier-first.** One ABAB per arm (or one composed tranche boot with per-arm env flips in
  separate legs); read WSTAT p50/p99 + M1, M3b, golden hash. UseNodePriority's specific watch:
  priority inversion against doorbell spin nodes (T27-1 C10) — nsys the shared-expert fork.
- **Composition + laws.** Tranche 1 precedes the PDL chain (V2) by design. Law 9: LFUSE/FUSE_WAIT/
  DF_PLE are REMOVE/FOLD items already written — arming costs nothing. Rollback: each `=0`/unset.
- **Grade.** Node deltas [MEASURED-OURS (F-204) pre-dm, needs re-count]; per-arm speed
  [HYPOTHESIS until arms].

### V4. mid_chain + dec_gdn_rec ports — the last banked kernel-pathway items

- **Mechanism.** mid_chain (basalt, 553 lines + hpp; `kMidChainMaxT=4`, mid_chain.cu:444/:507)
  runs qkv/z/out projections + the recurrence + norm as ONE persistent kernel per GDN layer for
  windows T ≤ 4 — **our mean window is 2.47 accepted rows, so the fused form covers the M1/pair
  mix**, with the unfused spine as the T ≥ 5 fallback. Our replaceable spine (§1): qkv mm + z mm +
  rec/norm (+ quant on batch_rec) + out mm ≈ 4-5 nodes/layer → 1-2. dec_gdn_rec (407 lines,
  12 PDL refs, 4-blocks/value-head cluster + DSMEM butterfly, bitwise, no type restriction)
  folds conv+gates+rec+norm+q8_1 — overlapping what our spine already fuses (conv_ab covers
  conv+gates; step_norm covers rec+norm+q8_1): its honest remainder is 2 nodes/layer (T26-1's
  amendment).
- **Expected band.** Node term: mid_chain −3..4 nodes × 24 GDN layers × ~0.7 µs ≈ −0.05..−0.07
  ms/window; execution term (x/qkv/hb/z DRAM round trips kept in SMEM: ~0.1-0.2 MB/layer saved):
  −0.2..−0.5 ms/window. Total **−0.3..−0.6 ms/window ≈ +1.5..+3% M1** [HYPOTHESIS] — larger than
  dec_gdn_rec's +0.5..1.5% now that mid_chain is unconditionally unblocked (§1.c1). dec_gdn_rec
  stays the first rung: pure-bitwise, smaller, and its parity rig is the on-ramp (T26-1 ladder).
- **Cost anatomy.** Ours: verify.cpp:1522-1624 (spine counted, this session); geometry N=2560/
  C=10240/HV=48 (T26-3 header). Donor: mid_chain.cu:444 (type_ok), :507 (gate incl. hard geometry
  check `n_embd==2560 && h_v==R_HV && h_k==R_HK && C==R_C`), :479-481 (device cover check: SMs ≥
  kMidCtas, smem optin ≥ SMEM_MAX — GB202 satisfies; basalt ships on a 5090), MID_TRIPLES
  (14,14,14)/(23,21,12) + generic runtime-type form (:450-456). **One pre-port read owed: confirm
  R_HK/R_HV/R_C equal our GDN geometry in the donor headers** — if they differ, the port is dead
  in five minutes (the gate exits).
- **Falsifier-first (cheapest first).** (0) The donor `mid_chain_supported()`/parity kernel run
  STANDALONE on our card with our type triples (generic path) — no engine build; (1)
  `dec_gdn_parity` T=1..8 bitwise (the harness both ports share); (2) T=2-4 microbench vs our
  fused spine; (3) then the engine arm behind `STRATA_MID_CHAIN=1` (default off), ABAB + M3b +
  golden hash. Mid_chain is NOT bitwise on projections dequant→fma order unless the donor's row
  order matches our mmvq's — the parity harness decides; if not bitwise, it is an M3b/KL-gated
  candidate (F-230 law), which would drop it below every bitwise item.
- **Composition + laws.** dec_gdn_rec's PDL preamble pairs with V2's chain; mid_chain is
  T-scoped (T≤4 fused, T≥5 spine) — law 4b-clean by construction (no lane pays). Rollback: env
  off; the fallback path is the standing code.
- **Grade.** Unblocking [MEASURED-OURS (GGUF headers)]; bands [HYPOTHESIS]; bitwise-ness
  [UNKNOWN until the harness].

### V5. Cold-start structural options (post-prewarm2)

The adopted mitigation (prewarm2.sh: dd re-warm + capture-forcing replay, battery-wired at
campaign/batteries/t32b-dm-fiveboot.sh:48) works; the remaining structure:

- **V5a. The STRATA_PIN_GUARD=2 diagnostic boot (zero-code, prices three questions).** Guard=2
  skips the wholesale drop (expert_source.cpp:3516-3518) and pins in kswapd-paced steps that take
  2 MiB pages (:3576-3579) — i.e. **it is simultaneously the bounded-re-read arm AND the
  hugepage-middle-form arm (V6)**. What one back-to-back boot pair reads: (i) "really free" line
  absent? (ii) complement-ready seconds vs the 8 s / 18-20 s pair (watch kcompactd — the #771
  compaction-stall class is live under this box's `defrag=[madvise]` if huge_alloc faults into
  fragmentation); (iii) request-1 file-tier MB (38,602 on the drop boot) and WSTAT of leg 1.
  Honest mechanism note: the paced pin reclaims ≈ the shortfall (~38-40 GiB) which kswapd will
  mostly take from the shards' cache anyway — **the engine-side re-read barely moves; the win is
  (a) ~50 GiB of non-model page cache preserved (system-level), and (b) the 2 MiB-page path**.
  Scoreboard expectation ≈ 0; diagnostic value high. Rollback: unset.
- **V5b. Shortfall-scoped drop (S code, only if V5a's paced pin is too slow).** Replace the
  whole-file `fadvise(fd,0,0)` with a range loop (fadvise takes offset+len; page-align per borg
  #907) dropping shard ranges in reverse-profile-heat order until `free_now ≥ bytes+reserve`.
  Same system win as V5a without the paced-pin cost; keeps the one-shot driver path. Falsifier:
  the same three reads on one boot. Law 9: FOLD (one loop replaces a whole-file flush + a later
  dd re-warm of what wasn't needed).
- **V5c. In-engine post-pin warm pass (S-M code).** Fold prewarm2's dd stage into the engine as a
  boot thread after the complement-ready line (the same move-after-the-pin shape as the original
  prewarm fix): the engine owns the shard list, the 8 I/O threads, and the ordering — the external
  stage and its orchestration failure mode (the battery label-bug class) disappear. The replay
  burst stays external. Benefit: every boot gets the I/O half of prewarm2 automatically at ~+8-10 s
  boot wall; speed delta ≈ 0 vs prewarm2 (it does the same reads). This is a robustness/simplification
  item (law 9: REMOVE an external stage), not a scoreboard item.
- **What the remaining cold-gap costs post-T36.** With prewarm2 wired into batteries: the cold
  leg's +10% first-leg class becomes prewarm2's ~+3-5% five-boot band (F-266 adoption) minus
  prewarm2's own ~10-20 s. The unrecoverable residue is the boot's own re-read (profile prefill +
  complement copy ≈ 36-57 GB at NVMe speed ≈ +10-12 s) — avoidable only by not dropping (V5a/V5b
  trade that against system cache, not against engine speed).
- **Grade.** Mechanism [MEASURED-OURS (log:14374-14381, :28028)]; V5a-V5c values [HYPOTHESIS /
  structural].

### V6. Hugepages for the complement — the middle form exists (paced path), #771-conditioned

- **Mechanism.** The one-shot `cudaHostRegister` of ~37 GiB stalls (F-276 banked negative); the
  PACED path (guard=2) takes 2 MiB pages in-tree — that IS the middle form (staged registration
  of hugepage-backed memory), no new code. Value: the CPU-pool GEMV tier's IOTLB/dTLB recovery
  (Farshin class, T35 lever 3: +5-15% staging path ≈ 0.3-0.5 ms/window on the 2.21 ms CPU-experts
  term) + pin PTE count 9.35M → ~18k + shorter page-table RAM.
- **The condition.** #771 (family-measured, this box's defrag setting): MADV_HUGEPAGE on fault
  under fragmented RAM = synchronous-compaction stall (25 s → 432 s in the worst family case).
  The falsifier in V5a covers it (clock the pin, watch kcompactd); the modern fallback if it
  stalls is fill-then-`MADV_COLLAPSE` (Linux 6.1+; this box runs 7.2.10) — an S-code change
  confined to the huge_alloc helper.
- **Falsifier-first.** V5a's boot IS the falsifier (pin seconds + pool GB/s from the batch line +
  ms_pool term). Rollback: `STRATA_HUGEPAGE_ARENAS=0` / guard unset.
- **Grade.** Mechanism [CORROBORATED]; our magnitude [HYPOTHESIS]; stall risk [MEASURED-OURS
  (family #771)].

### V7. Telemetry-driven tail hunt — the standing program (design, Python-only)

The instrument is banked (WSTAT p50/p99/max at generate.cpp:11766 → /metrics window_ms_*;
`strata:token_gap_seconds` true per-gap histogram, prometheus.py:132-136; admission_wait; elastic
counters). What's missing is the standing program:

- **Alarms (probe-metrics/battery-side, per leg):**
  1. *Slow-detect*: leg token_gap p99 > 1.5× the rolling 10-leg median → flag the leg (catches
     eunomia-class box noise and lifecycle regressions; the mean-of-gaps era's blind spot).
  2. *Window-p99 regression*: five-boot pooled WSTAT p99 tracked per recipe; +5% over the standing
     pool = investigate before any adoption claims (the tail counterpart of the sub-4% law).
  3. *Event canary*: park D2H (40-51 ms) or promote must NEVER appear inside a token gap
     (timed-scan contract); one correlation pass per battery over demote-line timestamps vs WSTAT
     max windows is the regression tripwire for A3.
  4. *Golden hash* (exists) + clock/temp scrape (exists, t36 log shows it riding) stay.
- **Hunt list (first targets, with what the true gaps expose):**
  1. The gen-1 post-idle window outlier class (t36-lifestic.log: "[30] … gen 1 → 31.0 ms/window"
     against 7-13 ms neighbors) — correlate with demote/restore lines; if it is a restore
     (9-15 ms) + first-window class landing on one token, price moving the restore under the
     arriving prompt read (law-9 MOVE).
  2. GPU-reach wait tails inside windows (mean 0.54 ms; the p99 is inside WSTAT p99 — decompose
     with the existing wait-stats on one diagnostic boot before touching anything).
  3. Admission_wait p99 at 9-wide ramps (the 2,395 ms TTFT p99 anatomy: queueing + serialized
     admissions; T35-elastic's <3% elastic share is pre-park numbers — re-read with the new
     counters).
  4. Decode-fetch wait after the demote-adjacent hit-rate dip (93.1% vs 99%+ class, T35-elastic
     Q3) — now measurable per window instead of inferred.
- **Grade.** Design [implementer-ready]; all reads exist server-side today (verified prometheus.py
  + server.py anchors, §1).

### V8. PCIe — item dissolved and reframed (correction §1.c2)

Standing = 0.35 (config, ground truth); 0.55 is ledger-dead (F-263 FINAL, F-265). The only live
PCIe items, both previously banked: (a) `tools/calibrate.py` frac sweep in a quiet window
(zero-code, T35 lever 8 — expected ~0 by F-263's own law); (b) the PIPED-for-cohorts port with
`--pcie-mode kernel` (banked +6-8% conc class, M cost, gate at batch_read.hpp:48 — a serving-lane
item, not this dossier's). External sweep found nothing new for single-GPU mapped-pinned miss
paths in 2025-26 (§2). Verdict: keep 0.35; do not re-arm 0.55 without a new five-boot pooled
result; the miss path is <1% of window (log:32637: PCIe 0.13 ms/layer-window, 0.4% of entries).

---

## §4 NEGATIVES — researched and rejected this session (with reasons)

1. **CONV_COMPRESS structure-aware lossless coding: KILL, permanently.** F-276 kept it open
   ("structure-aware codec owed"). The external bound closes it: SplitZip's lossless exponent-
   coding gets 1.324× on BF16 and 1.136×/0.933× on FP8, and states the ratio is bounded by
   floating-point EXPONENT redundancy — int8 quantized codes have no exponent field, so the
   class of structure SplitZip/CacheGen/Ecco exploit does not exist in our parked int8 KV rows.
   Measured 1.01× (F-276) is AT the honest bound; decompression would ADD host time to the 9-15 ms
   restore for ≤1% fewer bytes. Re-open condition: the parked state becomes fp16/fp8 (it will not;
   kv int8 is standing).
2. **KV-shadow compression rider: same kill** (same int8 data, same bound; the shadow is 1.55 GiB
   flat — T35's correction — so the capacity argument never existed either).
3. **A driver-side batched cuMemMap: does not exist through CUDA 13.4** (release notes scanned);
   op-count reduction stays the only cure; 64 MiB superchunks (12 ops/arena) would save ~0.5 ms
   per lifecycle event that already runs background at 2.2-8 ms — park at 32 MiB; re-price only if
   arena-churn rate rises by an order of magnitude.
4. **Transferring eunomia's 15.3 ms to our launch pairs: rejected.** Their workload emitted ~950
   host launches/token with no graphs; ours replays windows as single graph launches — the
   frequency of exposed intervals differs by ~10³. The transferable unit is per-window/per-gap
   variance (V1's telemetry reads), not their per-pair figure.
5. **gpu_ext's GPU-side timeslice scheduler: not our layer.** It schedules GPU work between
   processes; we are one process on one GPU. Its host-hook result (<0.2% overhead) and its RTX
   5090 host are noted; no port.
6. **isolcpus/nohz_full as the FIRST isolation move: rejected.** Kernel-cmdline + reboot cost,
   desktop collateral, and the eunomia data show taskset+nice recovers 7.6 of 8.8 points; the
   escalation is held until V1's falsifier shows residue.
7. **SCHED_FIFO for the service thread now: rejected.** rtkit/RLIMIT_RTTIME plumbing + desktop
   risk for a gain the nice tier likely captures; last rung, not first.
8. **PDL blanket-global adoption: stays rejected** (T27-1 N11 reaffirmed with the CUDA doc's own
   visibility warning as the mechanism record) — V2's per-family switch is the only sanctioned
   form.
9. **"Pin before the drop" boot ordering: rejected on mechanism.** The drop exists to CREATE the
   really-free RAM the one-shot driver pin requires; ordering cannot remove it. The substitutes
   are V5a/V5b (change the drop's scope or replace it with paced reclaim).
10. **Dropping the drop via STRATA_PIN_GUARD=0: stays dead** (the #1250 OOM-kill guard comment at
    expert_source.cpp:3498-3503; T26-6 a.3). Guard=2 is the safe form of the same intent.
11. **vLLM #40807-class host copies in graphs: no action** — our capture audit (F-272) already
    bakes device pointers only; recorded as a constraint, not a candidate.
12. **Watch-only:** Blackwell fp32-accum unlock (none exists; F-252 watch), llama.cpp #28248 RFC
    (still draft), PeeR-style eBPF host preemption (interesting, no object here).

---

## CONTRADICTIONS AND THEIR RESOLUTION

- **Assignment ("frac 0.55 adopted, +2.8% pooled") vs config+ledger (0.35, falsified):** resolved
  for the record — config is ground truth; F-263 FINAL/F-265 are the verdicts. Main loop should
  confirm no operator decision post-F-265 re-armed it outside the journal; nothing in the tree or
  log suggests one (boot line would print "pcie_frac 0.55").
- **F-191 (PDL no-op) vs the chain band:** resolved — the F-191 arm priced −0.01..−0.05 ms
  (partial chain), inside its own ±1.2% floor; both stand.
- **T26-1 "36 GDN layers" vs T26-3's geometry ("GDN 24L"):** unresolved count; settle by counting
  gdn_idx entries at capture time before sizing V4's per-layer arithmetic (the band quoted uses
  24; at 36 the node term scales ×1.5 and mid_chain's band becomes −0.45..−0.9 ms/window).
- **prewarm2's cold-boot re-read (44-55 GB "inside requests 1-2", T17B) vs today's log (the boot
  itself pays it; request 1 absorbs the tail):** today's log is more precise (per-request file-tier
  lines + the 4,804 MB/s prefill arithmetic); T17B's phrasing predates per-request attribution.
  The prewarm2 value proposition is unchanged (it re-warms what the drop released and forces
  captures), but "the requests pay" should read "the boot+first request pay" in future quotes.

## OPEN QUESTIONS

1. GDN layer count on our pack: 24 or 36? (One grep at capture; sizes V4.)
2. Donor mid_chain geometry constants (R_HK/R_HV/R_C) vs ours — the five-minute kill check.
3. Is mid_chain bitwise vs our mmvq row order on q6_k/q5_k/q4_k/iq4_xs reads? (Parity harness.)
4. Does the paced pin (guard=2) hit the #771 compaction stall on a fragmented back-to-back boot?
   (V5a's single boot answers; kcompactd + pin seconds.)
5. What fraction of WSTAT max windows carry a context-switch signature on this desktop (the V1
   churn 3v3 answers it; eunomia's cuda_sched_trace is the reference design if we ever want the
   eBPF view).
6. Whether the cold boot's 38.6 GB request-1 read is fully attributable to the profile prefill
   (the 4,804 MB/s × 4.5 s arithmetic says yes) — confirm once on the V5a boot by reading the
   prefill-rate line and the file-tier line together.

## IMPLICATIONS

- The value-ranked queue for the next batteries: **V1 falsifier (one 3v3, zero code) → V3 tranche
  (four one-boot arms) → V2 kill-switch + nsys re-arm → V4 ports (dec_gdn_rec rung, then
  mid_chain, both with the shared parity rig) → V5a/V6 diagnostic boot → V7 alarms wired into
  probe-metrics before any of the above** (they make every other verdict cheaper and keep the
  lifecycle honest).
- Two assignment framings were corrected against the record (frac 0.55; eunomia magnitude) —
  future waves should quote the config and the primary figures, not the brief.
- The mid_chain unblocking is the wave's principal new fact: the last two banked kernel-pathway
  items are now both implementable with no re-quant and a shared falsifier.

## SOURCES

Internal (read this session at HEAD bf746391): src/core/verify.cpp (:97, :317-346, :1016-1029,
:1431, :1522-1624, :2539-2544, :2776, :3738-3745); src/kernels/cuda/verify_kernels.cu (:1976-2050);
src/core/expert_source.cpp (:101, :615-616, :3484-3543, :3576-3579, :5897-5913); src/kernels/cpu/
pool.cpp (:460-551); src/core/vmm.cpp (:15, :65-75, :189, :226-232) + include/strata/core/vmm.hpp
(:7-26); src/program/generate.cpp (:602-603, :1103, :11766); serve/prometheus.py (:13, :46, :132-136,
:204-213); serve/server.py (:959-969, :1536, :2078); launch-custom-iq3s-8081.sh (:13-51);
custom-iq3_s-8081.json (:28-29); custom-iq3_s.log (:14374-14381, :16080, :25077, :26544, :28021-28028,
:29738, :31203-31225, :31282-class, :32637-class, :32685); campaign/results/t36-lifestic.log (tail);
campaign/prewarm2.sh; campaign/batteries/t32b-dm-fiveboot.sh (:48); third_party/llama.cpp/ggml/
include/ggml.h (:398-413); /home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}
(of-00002.gguf GGUF v3 headers, parsed this session); /home/svg/LLM/data/references/basalt/src/
kernels/cuda/{mid_chain.cu (:444-520), dec_gdn_rec.cu (:1-30)} + mid_chain.hpp (:15-18); /sys and
/proc (lscpu, cache index3, THP enabled/defrag, meminfo — 2026-10-23).

External: eunomia CPU-noise (2026-05-24) https://eunomia.dev/blog/2026/05/24/measuring-cpu-noise-in-gpu-inference-with-ebpf ·
gpu_ext arXiv:2512.12615 · CPU-induced slowdowns arXiv:2603.22774 · PeeR OSDI'26
https://www.usenix.org/system/files/osdi26-carin.pdf · CUDA PDL guide
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html ·
CUDA toolkit release notes https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html ·
CUDA 13.1 announcement (carried T27-1) · UseNodePriority runtime docs (carried T27-1) · TRT-LLM PDL
casebook (carried T27-1) · posix_fadvise(2) https://man7.org/linux/man-pages/man2/posix_fadvise.2.html ·
oetiker fadvise https://insights.oetiker.ch/linux/fadvise · glandium https://glandium.org/blog/?p=1262 ·
borg #907 https://github.com/borgbackup/borg/issues/907 · THP docs
https://docs.kernel.org/admin-guide/mm/transhuge.html · Strata #771
https://github.com/Niko1221/Strata/issues/771 · SplitZip arXiv:2605.01708 · Ecco
https://dl.acm.org/doi/10.1145/3695053.3731024 · Speculative KV coding
https://fergusfinn.com/blog/kv-entropy-coder · CacheGen arXiv:2310.07240 (carried T35) · Farshin
IOTLB (carried T35 S9) · vAttention v3 arXiv:2405.04437 (carried T35 S10) · vLLM #40807 ·
vLLM metrics docs https://docs.vllm.ai/en/latest/design/metrics · akrisanov vLLM monitoring
https://akrisanov.com/vllm-metrics · DigitalOcean p99
https://www.digitalocean.com/community/conceptual-articles/when-your-vllm-p99 · ClickHouse LLM
latency https://clickhouse.com/resources/engineering/llm-inference-latency · EP-over-PCIe
arXiv:2609.40093 · NVIDIA overlap-transfers blog
https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc · NVIDIA isolcpus forum
https://forums.developer.nvidia.com/t/how-to-use-cpu-isolation-to-reduce-network-latency/206672 ·
llama.cpp PDL thread (2025-08) · Event Tensor arXiv:2604.13327 (carried T27-1).

Banked campaign consumed: F-191 (:6446), F-204, F-232 (:7942), F-251/F-252 (:8552), F-259
(:8800), F-263 FINAL (:8921), F-265 (:8972), F-266 (:8986), F-275 (:9191), F-276 (:9221);
T26-1, T26-3, T26-6, T27-1, T27-3, T33, T35 ×3.

## METHODS NOTE

Productive: the two GGUF header parses (resolved a named open question without a boot); the log
re-reads of drop boots (the request-1 arithmetic that re-attributed the cold re-read to the boot);
the pool.cpp/verify_kernels.cu gate reads; first-hand fetches of the eunomia primary, the CUDA PDL
guide, SplitZip, issue #771, and arXiv:2603.22774. Unproductive / do-not-repeat: searching the
eunomia figures in gpu_ext/the LPC slides/dev.to (the blog post of 2026-05-24 is the only primary;
the 15.3 ms lives there and only there, scoped to 0.1% of launch pairs); "SlabQuant" (already
recorded unfindable, T35); CUDA 13 batched-map searches (confirmed absent — stop re-searching
until 14.x release notes); conversions.json tensor entries (empty — the GGUF headers are the only
type authority). Estimates are marked; every number else traces to §1/§2 anchors.
