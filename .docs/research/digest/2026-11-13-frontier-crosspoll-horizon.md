# Frontier cross-pollination + horizon audit (2026-11-13) — graphics/dynarec/audio/HFT/DB-planner lessons + 2027

Batch 3 of the 18-agent frontier wave.

## Ranked findings
1. **Cost-model feedback loop (Spark AQE lesson)** — the acceptance side of the controller is fully adaptive
   (EMA + damped IRLS); the COST side is a frozen offline fit (M15 8.4+1.57*rows) while the doorbell loop
   already collects the refit inputs every window (ms_wait/ms_pool, verify.cpp:3236-3238). A stale curve
   misprices every window under load/thermal/residency drift. LOW port (EMA refit, damp like the
   SurvivalEstimator). Also the enabler for 3 and 5: runtime-measured constants make every horizon item pay
   with zero code.
2. **Frame-graph capture-time liveness linter + scratch aliasing (EA FrameGraph lesson)** — carve() is a
   manual bump arena over ~60 buffers; no interference analysis within it (staging_/attn_scratch_/hit_scratch_
   could overlap = VRAM freed funds expert residency, the binding constraint of the zero-doorbell path); and a
   read/write linter over the flag choreography mechanically catches the #267/#871 stale-hand-off class.
   MED port, additive, low risk.
3. **CUDIMM/CAMM2 memory wave — the BINDING constraint gets +20-40% with zero code** (CUDIMMs 8000 MT/s on
   AM5 confirmed; CKD to 9200-9600; CAMM2 7500 2026; DDR6 2027). Higher consumer bandwidth makes the hybrid
   CPU-expert design MORE valuable. Depends on finding 1's feedback loop to exploit without code edits.
4. **Persistent prelaunch worker (HFT/audio rule: never spawn on the hot path)** — the overlapped split
   creates a fresh std::thread PER WINDOW (verify.cpp:3121) = 20-60 us + stack faults on every split-stage
   critical path. Pre-spawned stage worker on its own flag (pattern exists a dozen times in-tree).
   LOW-MED port; ~0.2-0.5% wall + tail-jitter cut.
5. **CUDA 13.1 green contexts** — SM partitioning to fence the busy-wait spin kernels away from expert
   kernels (SM contention with spin kernels is exactly our measured tail); device graph launch could remove
   the conditional wrapper's last host round trip. LOW-cost experiment battery.
6. **FP4 expert lane = VRAM capacity, not speed** (consumer GDDR7) — ~4.5 bpw resident experts roughly
   double the residency threshold → ar_on zero-doorbell windows more common (the hot path already built).
   Keep FP4 experts VRAM-only (the CPU miss lane never sees them). MED port; quality-gated.
7. **PLE NVMe reads sit on the doorbell thread** (audio rule: the RT thread never touches disk) —
   gather_batch runs NVMe inline on the poller (verify.cpp:3228-3234); a slow page holds the window's
   critical path hostage. Feeder-worker + flag; MED port.
8. MoE architecture trend (512 fine experts, 3-4% active; DSA going mainstream) — TAILWIND for our design;
   capacity check done (plan ring scales with rows*K not expert count). Mostly negative.
9. Spec-decode 2026 critique (MLSys "Performance or Illusion?") — our shape already answers it (per-depth
   EMAs, IRLS priors, mid-flight trim, SURVTOPK). Negative; only action = measure-and-gate SURVTOPK.
10. Threat scan: PCIe 6 not consumer before 2027+ (Zen 6 stays AM5/Gen5 — no trigger); new engines
    (LayerStoRm/NEO/expert-cache frameworks) none implement doorbell-paced interleave — moat intact.

## Negatives (verified embodied)
Disruptor/SPSC sequence-barrier discipline on the doorbell ring (monotonic cursor, single-writer flags,
separate 64B slots); no alloc/lock/driver-call in the steady loop (the 2ms-throttled query is the only
exception); HFT substrate complete (pinning, NUMA, THP arenas, _mm_pause); dynarec lessons embodied (8
per-width graphs = specialized blocks; refresh_ar = hot-path promotion); server.py control plane correctly
excluded from RT discipline.

## Record note
CONFLICT flagged: the llamacpp-delta digest says upstream bumped to 1.0.0 (#32190); this audit finds no
1.0.0 exists (semver v0.1.0/v0.2.0, Aug 2026). Unresolved — verify against the repo directly before citing
either way; immaterial to the findings.

Sources: EA FrameGraph GDC'17; D3D12 aliasing barriers; RPCS3 recompiler; LLVM JITLink; LMAX Disruptor
(paper + Fowler); AQORA; Spark AQE; Bencina RT-audio-101; wait-free SPSC; LWN HFT; C++ low-latency
patterns; CUDIMM-on-AM5 (MSI); Montage CKD; CAMM2 outlook; PCIe6 timeline; Nova Lake validation; CUDA 13.1
green contexts; CUDA 13.5 Rubin; NVFP4 announcement + vs-MXFP4; llama.cpp FP4 landing + semver; MLSys'26
spec critique; P-EAGLE; Qwen3-Next; DSA; LayerStoRm; NEO MLSys'25; CPU-GPU expert caching arXiv:2512.16473.
