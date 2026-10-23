# T42-r1 — Expert-offload miss path map (strata-custom-2, IQ3_S native pack, RTX 5090)
All anchors relative to /home/svg/LLM/strata-custom-2. VERIFIED = read in code/log this pass. Runtime facts from custom-iq3_s.log (2026-10-22 boot).

## Summary table
| Q | Answer | Anchor |
|---|--------|--------|
| 1 | Misses live in a 42.29 GiB **page-locked "cache complement"** arena in host RAM (cudaHostAllocMapped, device-aliased; alt: 2 MiB hugepage mmap + mlock + cudaHostRegister). Pack files mmap'd only as fallback | expert_source.cpp:3545-3553, 3460, 3651, 3470; log:46-53; mmap 1457-1477 — VERIFIED |
| 1b | Read path: `blob()` = complement first, else page-cache fault on mapping; io threads pread whole blobs ahead | expert_source.cpp:4314-4330, 2744-2757, 906-930 — VERIFIED |
| 2 | **15 CPU workers + host thread drains** (16 phys cores, host on core 0). Queue = flat per-call batch, head_=epoch\|njobs\|idx CAS claims. Wake = `_mm_pause` spin 100 us (STRATA_POOL_SPIN_US=100 in launcher; default 20 ms) then condvar `sleep_cv_` | generate.cpp:6058; log:29-31; pool.hpp:200-207,243,301-311; pool.cpp:697,753-755 — VERIFIED |
| 3 | Miss is SPLIT: ~65% COMPUTED ON CPU (activations zero-copy D2H, outputs zero-copy H2D in pinned mapped bufs); ~35% (pcie_frac 0.35) WEIGHTS DMA'd to GPU staging, computed there. No blocking cudaMemcpy on the path — all mapped-memory flag polling | expert_source.cpp:4593-4805, 4852-4860, 5032; verify.cpp:2083-2094, 2245-2331 — VERIFIED |
| 4 | Prefill >=1024-token chunks: non-resident WEIGHTS stream through a pinned slot ring into VRAM, GPU computes (ncmoe style). <1024: measured CPU share (auto default ON, single CUDA GPU + pool). `--pcie-frac 0.35` = decode share of misses DMA'd to GPU = round(0.35*256)/256 | prefill.cpp:179,195-199,2864-2873; generate.cpp:9095,10109,2229-2240,2124 — VERIFIED |
| 5 | YES, two: (a) RouterLookahead thread recomputes next-layer router on CPU, warms predicted misses (madvise/pread + ForesightSwap GPU swap space); (b) io-prefetch scheduler: confidence admission, per-window byte budget from measured read rate | expert_source.cpp:2938-3010, 2811-2819, 2724-2763, 6041, 6079-6096, 812-860, 878-906; log:41 — VERIFIED |
| 6 | ExpertCache = one VRAM slot arena (cudaMalloc/VMM segments), per-layer admission quotas (R4.2g), **never evicts**; filled at boot from expert-profile rank (--resident-experts); adaptive swaps exchange with RAM complement | expert_cache.hpp:105-185; expert_cache.cpp:643-663, 471-560; log:45,55 — VERIFIED |
| 7 | Coordinated: PcieAuto balances link vs measured pool GB/s; PcieModel online regression picks per-layer share; io budget = worker rate x threads x window x frac with useful-fraction EWMA; decode gate on page-cache headroom | generate.cpp:2229-2240; expert_source.hpp:395-435; expert_source.cpp:812-898 — VERIFIED |

## Miss path, decode verify window, per layer (all VERIFIED)
1. GPU graph: router kernel -> `doorbell_publish`(_value/_res) writes ids/w/x to MAPPED host mem, ring `m_seq_++` — verify.cpp:2083-2094.
2. Host thread (pinned core 0) spins on mapped `h_seq_` — verify.cpp:4450, 4481 (poll; cudaEventQuery guard every 2 ms; 20 s timeout 4494). On ring: `pool(...)` = drive_pool_multi (verify.cpp:4508-4510 -> generate.cpp:1216-1221).
3. `expert_pool_dispatch_multi` plans: kind 0=VRAM hit, 1=PCIe (last `m=(nmiss*pcie_num)>>8` misses), 2=peer, -1=CPU — expert_source.cpp:4650-4782. Seq-cst fence, `P.publish` -> host raises **flagA** (4801-4803; verify.cpp:908, 3344).
4. `P.fetch` = `fetch_dma`: cudaMemcpyAsync/BatchAsync H2D of kind-1 blobs from pinned complement -> VRAM staging on copy stream; `cudaLaunchHostFunc` raises **flagB** on completion — expert_source.cpp:4805; verify.cpp:3316-3342. Async; host never blocks on it.
5. CPU share: activations quantized straight from mapped x (zero-copy D2H, expert_source.cpp:4852-4860); cold file reads overlapped (`prefetch_begin`/`wait_landed`, 4883/5011); **blocking** pool join `run_split_multi_native` (5032) writes rows into pinned `h_ymiss_`; host raises flagA/flagB/flag_ via volatile/atomic stores + sfence (verify.cpp:4516-4527, 3298-3311).
6. GPU: device spin kernels `wait_flag_ge`(flagA plan, verify.cpp:2232) -> grouped VRAM hits (2237) -> `wait_flag_ge`(flagB, DMA, 2245) -> PCIe groups (2255) -> `wait_flag_ge(_or)`(flag_, CPU rows, 2268/2284) -> `copy_rows_from_mapped` zero-copy H2D of y_miss (2293) -> `moe_hit_add`+combine (2302/2326) or `dm_wait_combine` single launch folding wait+rows+combine (2315; STRATA_DM=1 in launcher).

### Per-handshake cost inference (INFERRED unless noted)
- Doorbell ring -> host: device PCIe write + tight host spin; ~1-5 us, no CUDA call (polled).
- Device spin-kernel node: ~2.4 us per dependent kernel in a graph — verify.cpp:295 (VERIFIED comment; gfx12-measured).
- DMA submit: batched (STRATA_DMA_BATCH=1, launcher; "84.9% fewer submit cycles"); completion polled by device, not host.
- Pool join: BLOCKING (wait_parked/drain/repark split timed, pool.hpp:230-234); cost = compute, not handshake.
- Net: ~2-4 launches + 2 polled flags + 1 DMA submit ≈ **10-40 us/layer pure overhead** (INFERRED). Real wait measured: "decode waited 1477.8 ms in 7504 fetches" ≈ 0.2 ms/fetch — dominated by expert bytes, log:106 (VERIFIED).

## External doc claims
(a) "Binding constraint is DDR bytes of host-resident experts per verify pass" — LARGELY RIGHT as a mechanism, BUT the median is already neutralized: steady state has 0 file reads; all misses hit the 42.29 GiB page-locked complement (log:106-107); pool drains at ~34-44 GB/s measured (pool.hpp:135-136,166-167); ~35% of those bytes also traverse PCIe for the DMA share. Caveat: it is per-layer-per-window, and the CPU share is compute-bound (bytes/BW), not just "touched". Cross-check with the live hit-rate distribution (p50 97.2%, p10 88.7%): misses ≈ 8% of the median forward, ~31% at p10 — the TAIL term, not the median term.
(b) "~30-80 us per missed layer in handshakes" — ORDER-OF-MAGNITUDE RIGHT for overhead-only (launches ~2.4 us each, polled flags, batched DMA), but it is NOT the binding cost: real per-fetch wait is ~0.2 ms (log:106), i.e. bytes/compute dominate by ~10x.
(c) "Prefill missed experts compute on the CPU; shipping to GPU is an unpriced lever" — WRONG for this config. Default path for chunks >=1024 tokens streams weights to the GPU and computes there (prefill.cpp:148-150, 179, 2864; our --prefill auto:32768 always hits this). CPU share exists only for chunks <1024 (auto default, prefill.cpp:186-199, 2871-2873; armed at generate.cpp:8548, 16644), measured g/(c+g). The lever is already implemented and priced.

## Notes / non-mapped
- `STRATA_ARENA_MMAP` alternative tier (experts.bin mapped unlocked, expert_source.cpp:5576-5608) unused here. Hugepages gated by STRATA_HUGEPAGE_ARENAS; paced mlock path via STRATA_PIN_GUARD (3510-3532) — this boot took the one-shot cudaHostAlloc path ("page-locked and mapped", log:53).
- AUX cpus: all non-pool/host threads pinned to CPU 16 (log:32; aux_cpus.hpp:1-14).
- Verification status: all anchors above were read directly this pass (VERIFIED); latency figures labeled INFERRED are derived from code comments + log counters, not profiled here.
