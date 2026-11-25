# F-451 — the PCIe Gen2 suspicion: investigated and REFUTED (platform exonerated; boot-fill gap is engine-side)

Operator directive 2026-11-23: "behaving like gen2 is unacceptable, please investigate."  Suspicion
arose in the F-450 rig audit: the boot log's cache pre-fill ran at 7555 MB/s - ~94% of a Gen2 x16
ceiling (8 GB/s) - while both link ends advertise Gen5 x16, and idle readings showed Gen1/Gen2.

## Method (measure-before-mutate; instruments preserved)

State capture (sysfs current_link_speed / lspci -vv both ends / nvidia-smi gen + replay counters),
then two CUDA probes (campaign/probes/pcie_probe.cu pinned, pcie_probe2.cu pageable) run with live
link-speed polling during transfers: 3x(40x256 MiB) H2D+D2H, then the engine's own fill pattern
(2 MiB x 9900 = 18.78 GiB equivalent).  nvcc 13.4, GPU otherwise idle (engine down).

## Results

| Measurement | Value |
|---|---|
| Idle link | 5.0 GT/s (Gen1/Gen2 view), P8 power state, zero replays |
| Under sustained DMA | **32.0 GT/s (Gen5) x16, uptrained within 250 ms** |
| Root port 00:01.1 | LnkCap 32GT/s x16, LnkSta 32GT/s (CPU-attached, not chipset) |
| Pinned H2D big-block (3 passes) | **52.2 GB/s** |
| Pinned D2H big-block | **43.6 GB/s** |
| Pinned 2 MiB x 9900 (engine fill pattern) | **39.0 GB/s (0.53 s)** |
| PAGEABLE big-block | 16.7 GB/s |
| PAGEABLE 2 MiB x 9900 | **20.5 GB/s (1.01 s)** |
| Engine boot fill (log, 2026-11-22) | 7.55 GB/s (2.7 s) |

## Verdict

1. PLATFORM EXONERATED.  The link trains Gen5 x16 in under 250 ms under load and sustains
   52.2/43.6 GB/s pinned with zero replays.  The idle Gen1/Gen2 readings are normal GPU power-state
   (P8) behavior, not a stuck link.  No BIOS/ASPM/reseat action.  Banked as a negative result.
   RE-OPEN CONDITION: if a decode-window nsys trace ever shows the link below Gen5 during active
   windows (bursty traffic with idle gaps can downtrain between bursts), revisit - a keep-alive or
   batching fix would then apply.
2. THE BOOT-FILL GAP IS ENGINE-SIDE.  The engine's 7555 MB/s is 2.7x slower than the identical
   byte pattern through pageable memory (20.5 GB/s) and 5x slower than pinned (39.0).  Candidate
   mechanisms (to pin inside the fill path when expert_source work resumes): scattered gather of
   profile-ranked experts across the 46.8 GiB mapping (my probe re-read one hot 2 MiB), per-slot
   bookkeeping/verification interleave, concurrent boot traffic (MTP draft load, KV pinning), or
   the custom DMA-batch path.  Recoverable: ~2.2 s of boot, pure transport, zero quality impact.
   Priority LOW (boot lane), complexity LOW once instrumented.
3. DECODE DMA CEILING IS NOT THE CONSTRAINT.  Even pageable 2-MiB transfers sustain 20.5 GB/s;
   decode windows move far less (telemetry: stage 0.10-0.15 ms, GPU-reach wait typically 0.40 ms).
   The decode bottlenecks remain the CPU pool (42% of window) and window tails - unchanged from
   the F-450 audit.  The earlier 8.77 ms GPU-reach phases stay on the nsys battery's list.

## Reproduce

    /usr/local/cuda-13.4/bin/nvcc -O2 campaign/probes/pcie_probe.cu  -o /tmp/pcie_probe  && /tmp/pcie_probe
    /usr/local/cuda-13.4/bin/nvcc -O2 campaign/probes/pcie_probe2.cu -o /tmp/pcie_probe2 && /tmp/pcie_probe2
    cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed   # poll during transfers
    lspci -vv -s 01:00.0 | grep LnkSta ; lspci -vv -s 00:01.1 | grep -E "LnkCap|LnkSta"
    nvidia-smi -q | grep -i replay
