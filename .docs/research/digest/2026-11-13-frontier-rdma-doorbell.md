# Frontier network/RDMA doorbell audit (2026-11-13) — io_uring/DPDK/RDMA/NVMe lessons vs our doorbell

Batch 3 of the 18-agent frontier wave. Framing: the quiet wait (0.98-1.55 ms) is GPU pre-work (compute-bound,
no protocol change collapses it); the busy tail (12.27 ms) is a VISIBILITY AND SCHEDULING artifact. The
network stack's law: in a well-built datapath the poll loop contains no syscalls and observation is decoupled
from busy cores. Our verify loop obeys; our SESSION loop does not.

## Ranked findings (order: 1 -> 3 -> 4 -> 2 -> 6 -> 5 -> 8 -> 7)
1. **Session loop's per-iteration cudaEventQuery is the io_uring anti-pattern** (session.cpp:701-722) — a
   syscall in the polling datapath AS A VISIBILITY MECHANISM; the three-experiment defense PREDATES the
   fence fix (the ~10ms/token recovery documented at elementwise.cu:199-204); the verify loop runs the
   throttled form on the same mapped memory and observes rings fine — the discrepancy is itself evidence
   the claim is stale. Trivial port (swap in the verify-loop throttle); tests 51/56 are the gate; if it
   still breaks, the experiment IS the finding (driver-injected visibility) and finding 2 becomes mandatory.
2. **Dedicated poller thread + futex/eventfd handoff** (the io_uring eventfd completion pattern; zero
   futex/eventfd in the tree) — one pinned watcher owns query/timeout and wakes the session thread;
   observation decoupled from busy cores STRUCTURALLY = the only durable fix for a preemption-caused tail.
   MED port; futex expected-value discipline for missed-wake-freedom.
3. **Core isolation for the doorbell spinner** (DPDK dedicated-core doctrine) — the machinery exists
   (pool_affinity_linux, aux_cpus, kStagePinFloor precedent); LOW port; opt-in.
4. **UMONITOR/UMWAIT on the doorbell line — ALREADY IN-TREE FOR THE POOL (pool.cpp:824-840) but used by NO
   doorbell spin** (verify.cpp:3150, session.cpp:721 bare _mm_pause) — the FIFTH built-but-unarmed instance
   of the wave. Arm umonitor on h_seq/h_flagA_ (each flag its own allocation = alias hazard designed out);
   copy the pool's gated pattern verbatim. LOW port; land finding 1 first for session.
5. **cuStreamWriteValue32/WaitValue32/BatchMemOp as graph nodes** — CUDA's literal doorbell registers:
   publish_plan/flagB raises become WriteValue32 nodes (no more cudaLaunchHostFunc driver callbacks,
   10-50us class + serialization); wait_flag_ge spin kernels become WaitValue32(GEQ) — frees a spinning SM
   per waiting layer. MED port; driver-gated (CAN_FLUSH_REMOTE_WRITES probe; keep kernel fallbacks);
   documented deadlock warning — keep event ordering.
6. **Completion-gated stage-buffer reuse** — the STAGE_PIN_DEFAULT=0 corruption (expert_source.cpp:1562-65)
   is the RDMA dereg-MR-in-flight bug VERBATIM; the fix is ownership-until-completion (stage_ev_ exists);
   re-enables pinned DMA staging by default = faster cache-miss loads in exactly the busy-window regime.
7. Adaptive query moderation (fixed 2000us gate -> last_ring_interval/4 clamped; the 20s timeout is
   over-coalesced failure latency). Trivial.
8. Exponential __nanosleep backoff on GPU spins (fixed 100ns -> ~10us cap): cuts idle PCIe read traffic
   10-100x during host-pool stretches. Trivial; arch guard stays.

## Negatives (best practice embodied)
The ring is a correctly-built NIC doorbell (monotonic producer counter + GEQ guard; BlueFlume fused
publish in one kernel); WC-flush discipline correct at every site (fence-before-ring; sfence drains WC
buffers in the right order); host->device flags = NVMe shadow doorbells (memory doorbell, no MMIO);
mapped-pin per field = registered-MR equivalent; one-writer-per-line (the pool already paid the
ping-pong tuition); poll-first-with-fallback correct on verify. Nit: uint32 compares not wrap-safe
(unreachable at ~5e7 tokens; two-line subtraction compare removes the class).

Sources: io_uring_setup(2), eventfd(2), futex(2)+Drepper; DPDK ring/power/PMD docs; kernel device-io +
mutex-design; virtio packed rings; CUDA memops docs + FLUSH semantics; Mellanox PRM (UAR/BlueFlame);
RDMA doorbell research; Kaminsky RDMA guidelines; NVMe shadow doorbell notes; adaptive moderation guide;
ibverbs MR/ODP; LWN FOLL_PIN.
