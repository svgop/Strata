# T51-R7 Dossier: Cold-Start & Warm-Up of Local LLM Inference

Researcher R7 | 2026-10-25 | Box: RTX 5090 32GB / 9950X3D / 184GB RAM; Qwen3.8-Flash-Next MoE IQ3_S (~43-50GB experts; tiers 20.6GiB VRAM / 36GiB pinned RAM / file tier), GDN hybrid, MTP spec-dec.
Measured baseline: boot-to-serving 60-90s w/ warm page cache; prewarm2.sh adopted; M1 219-232 t/s fully warm vs 205-215 fresh (best 234.1); TTFT p50 0.3s / p95 0.9s; 88k prefill 12.5s; slot admission copies 40-90ms; parking restore 9-15ms vs 275-326ms re-read; page cache served 100% of decode reads when warm; experts via mmap from GGUF.

## PART 1 - PER-SOURCE FINDINGS (A = primary/measured; B = solid/partial; C = context)

**[S1] ServerlessLLM (OSDI'24, arXiv:2401.14351) - A.** Loading-optimized checkpoint: per-GPU tensor groups, stripped metadata, index file -> large chunk-sequential reads at NVMe bandwidth; O_DIRECT (rejecting mmap), pinned staging, multithreaded I/O, pipelined DMA. RAID0 12GB/s: 3.6-4.7x vs safetensors, 6-8.2x vs PyTorch; per-technique bulk 1.2x / O_DIRECT 2.1x / multithread 2.3x / pinned 1.4x / pipelining 1.5x; saturates with 4 cores. safetensors cold load of 7B = ~112k page faults - mmap faulting is the cold enemy; startup 0.8s (6.7B) / 7.5s (30B). Our prewarm2 = the user-space equivalent.

**[S2] PPC/MAIO (FAST'26, Huawei) - A.** Programmable page cache (prefetch+eviction policy per-app, I/O templates): model-load latency -up to 79%; +36% throughput in elastic deployment. Validates page cache as the lever and EVICTION RESISTANCE as first-class - the half prewarm2 lacks.

**[S3] CRIUgpu (arXiv:2502.16631) + NVIDIA cuda-checkpoint - B.** lock->checkpoint->restore->unlock around CRIU; GPU state to driver-managed host memory; merged CRIU 4.0+ (driver 570+); deployed at MemVerge/Modal. LLaMA-3.1-8B/H100 (54GB): checkpoint 77.4s / restore 38.8s; restore is bytes/bandwidth-bound. Limits: no UVM, same GPU model/order, no NCCL multi-node. Our ~28-30GB GPU state: plausible ~15-25s restore preserving warm state exactly.

**[S4] mzCache (MobiCom'26, arXiv:2609.01338) - A (insight).** OS eviction inflates TTFT 0.8s->16s; full eviction = 18-20x paging cost. Generic compressors save 0.2-9.3% on KV/weights (don't bother); mlock protects weights not KV; PSI/vmpressure as the signal; evict LAST layers first (early layers needed for prefill); restore overlapped with prefill; chunk size to saturate device bandwidth.

**[S5] LMCache (arXiv:2510.09665) + docs - B.** GPU->CPU->disk->remote KV hierarchy; 400 Gbps CPU loads (4.5x vLLM); TTFT 1.9-8.1x at QPS=1; pin/unpin API. Cross-restart lookup NOT in the paper; issue #1175: needs a metadata layer hashing (prompt, model_config) - the hard part is the index, not the bytes. llm-d filesystem offload claims pod-restart persistence.

**[S6] LLM in a flash (arXiv:2312.11514, Apple) - B.** Windowing + row-column bundling (co-activated weights contiguous): 4-5x CPU / 20-25x GPU vs naive flash loading. Applies to our file tier + pinned-tier fill order.

**[S7] PowerInfer (arXiv:2312.12456) - C+.** Hot/cold power-law; tier assignment by measured activation frequency (we do VRAM tier; could drive pinned-vs-file tiering too).

**[P1] RunPod: vLLM cold starts to 90s (Sep 2026) - A.** Cold start = pull + weights + torch.compile + CUDA graph capture (2-4 MINUTES) + KV alloc/warmup; 8B vs 32B boot nearly equal (273s vs 324s) = compile/capture dominates, not bytes. Fixes: persistent VLLM_CACHE_ROOT (compile+graph cache on persistent volume), SAFETENSORS_LOAD_STRATEGY=prefetch (engine-integrated prewarm), limited cudagraph sizes, HF_HUB_OFFLINE. 324s -> ~91s (3.5x).

**[P2] Cedana: where LLM cold-start minutes go (Aug 2026) - B+.** "Readiness != warmth": engines mark ready before deferred compile/capture; first requests pay. Snapshot only AFTER warmup. Production restores 57-70s on 8xB200 from tmpfs vs 564-2051s native; restore storage-bound; restore slower than save. vLLM compile cache is portable between machines.

**[P3] llama.cpp #18758: mmap vs direct I/O for MoE (Jan 2026) - A (our family).** Qwen3-Next-80B IQ3_XXS (our model+quant family; 48x512 experts = 95.4% of bytes): (1) mmap+page cache BEATS O_DIRECT for warm repeat loads (O_DIRECT >=10x slower repeats); (2) explicit pread of non-resident expert slices with mincore() checks beats mmap faulting +13-14% e2e under pressure; (3) co-activation-aware expert layout: reads/token 1418 -> 775 -> ~370 (2.23x cold-decode I/O) on 1-2 GB/s drives; (4) bounded mmap caching cut peak RSS 47% in a 3.5GiB cgroup.

**[P4] SGLang HiCache docs + LMSYS blog (Sep 2025) - B.** L1 GPU -> L2 host -> L3 storage; write_through; host pool must exceed GPU pool or L3 hits are DROPPED (full recompute); prefetch policies; no documented cross-restart persistence.

**[P5] Ollama keep_alive - C.** Idle-timeout unloading is the common warm-state killer; our parked/pinned design is finer-grained.

**[P6] eunomia.dev OS-level challenges (Feb 2025) + NVIDIA CRIU blog - B.** mmap misses stall ~ms each; MAP_POPULATE, madvise(WILLNEED), mlock, pre-touch, larger readahead, THP (TLB wins; backfires fragmented), io_uring, cgroup v2 memory.min/low, NUMA-local placement 5-10%.

## PART 2 - SYNTHESIS
1. prewarm2.sh is literature-backed (ServerlessLLM sequential bulk + vLLM prefetch mode + PPC's prefetch half). The missing half = EVICTION RESISTANCE after warmup.
2. Modern boot is compile/capture-bound, not weight-I/O-bound (RunPod; Cedana "readiness != warmth"). Persist any JIT/graph/autotune artifacts; keep replay-before-ready.
3. mmap right for serving (never O_DIRECT - page cache IS our warm substrate, matching our 100%-warm measurement), wrong for cold: bulk-read-ahead fixes cold; mincore+pread fixes pressure-time faults; co-activation layout fixes cold I/O volume (slow-storage regime).
4. Pin selectively (mzCache): PSI-gated, not blanket; 184GB can hold the whole file tier.
5. Warm-process snapshot/restore is production-real (CRIU 4.0 + cuda-checkpoint; Modal/MemVerge; Cedana 57-70s); snapshot only after warmup.

## PART 3 - PROPOSALS (ranked)

**P1 - Pin + protect the pack/GGUF page cache (vmtouch -dl / mlock daemon / cgroup memory.min, PSI-gated) - RANK 1.** After prewarm2, lock the 43-50GB file tier into the ~120GB free RAM. Zero boot-time effect; makes the 219-234 band DURABLE under other workloads/idle. Hours (vmtouch) to a day (cgroup+PSI). Risk: over-pinning starves others; RLIMIT_MEMLOCK. Falsifier: eviction assault (40GB unrelated read + compile) after prewarm; M1 stays >=225 unpinned = skip; decays = pin. Site: prewarm2.sh tail + systemd.

**P2 - Persist compile/graph/autotune artifacts + readiness-after-warmup - RANK 2.** vLLM_CACHE_ROOT pattern: persist CUDA graph captures/autotune keyed (model, build, arch); ready only after shaped replay. If our 60-90s carries half the RunPod profile: -> 30-50s. 0.5-2 days. Risk: stale-cache invalidation (key hard). Falsifier: instrument boot phases once (weights vs graphs/compile vs replay); graphs+compile < 10s = dead. Site: engine startup before readiness.

**P3 - Cross-restart parking/prefix persistence (LMCache pattern + hashed metadata) - RANK 3.** Serialize parked-conversation KV + system prefixes to disk keyed (model_version, tokenizer_hash, prefix_hash); background-load index before ready. Turns post-restart TTFT for returning conversations from seconds toward 0.3-0.9s. 1-2 days. Risk: version-keying (wrong hash = corruption). Falsifier: count real resume-after-restart events; rare = system-prefix-only variant. Site: parking subsystem + boot loader.

**P4 - Co-activation pack layout + mincore/pread file-tier reads - RANK 4.** Log router co-activation in replay; rebuild pack adjacent (2.23x cold I/O on slow drives); file-tier misses via mincore-checked pread (+13-14% under pressure). Wins only when the file tier is hot (fresh boot/pressure). 1-3 days. Falsifier: log file-tier read share fresh-boot + induced pressure; <2-5% = drop. Site: pack builder + file-tier read path.

**P5 - Warm-process snapshot/restore (cuda-checkpoint + CRIU) - RANK 5.** After warm: lock/checkpoint (~28-30GB GPU state -> host) + criu dump; restore instead of reboot: ~20-40s preserving warm decode state bit-for-bit. 2-5 days + flakiness; no UVM; driver >=570. Falsifier: toy 30GB process + captured graphs through cuda-checkpoint --toggle; restore >45s or driver fail = drop. Site: systemd wrappers + signal handling.

**Stack effect (P1+P2+P3):** restart-to-first-quality-token ~halves (60-90s -> 30-45s + sub-second TTFT for returning conversations); the 219-234 warm band becomes durable under pressure instead of decaying to 205-215.

Sources: [ServerlessLLM](https://arxiv.org/html/2401.14351v2) | [PPC/MAIO FAST'26](https://www.usenix.org/system/files/fast26-liu-yubo.pdf) | [CRIUgpu](https://arxiv.org/html/2502.16631v1) | [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) | [NVIDIA CRIU blog](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu) | [mzCache](https://arxiv.org/html/2609.01338v1) | [LMCache](https://arxiv.org/html/2510.09665v2) + [issue #1175](https://github.com/LMCache/LMCache/issues/1175) | [LLM in a flash](https://arxiv.org/abs/2312.11514) | [PowerInfer](https://arxiv.org/html/2312.12456v1) | [RunPod cold starts](https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless) | [Cedana](https://cedana.com/blog/llm-cold-start) | [llama.cpp #18758](https://github.com/ggml-org/llama.cpp/discussions/18758) | [SGLang HiCache](https://docs.sglang.io/advanced_features/hicache_best_practices.html) | [Mooncake](https://arxiv.org/pdf/2407.00079) | [Ollama keep_alive](https://docs.ollama.com/faq) | [eunomia.dev](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations) | [vmtouch](https://github.com/hoytech/vmtouch)
