# Frontier platform/hardware audit (2026-11-13) — AM5/DDR5/IOMMU/Blackwell state vs our box (live-probed)

Batch 2 of the 18-agent frontier wave. Probed live during active serving (read-only).

## Ranked findings
1. **DDR5: four DIMMs = 2DPC** (four spd5118 sensors on i2c) — the bandwidth-capping population on AM5; and
   the lane ceiling 49.5-50 GB/s matches DDR5-5600 JEDEC exactly (89.6 GB/s theoretical) — EXPO state
   unverifiable without root (named boundary: `sudo dmidecode -t memory | grep Configured Speed`). EXPO +
   tRFC/tREFI: **+2-5% lane for BIOS minutes, no code**; 2x96 GiB 1DPC swap: +5-10% ceiling (hardware). The
   four DIMM temp sensors make a thermal guard possible for tREFI. The lane sits at ~78% of one-CCD pull,
   ~56% of DIMM ceiling — real headroom on the BINDING constraint of the whole MoE design.
2. **IOMMU fully translated (DMA-FQ, lazy) with no iommu= arg** — every DMA-read byte of the 37.6 GiB pinned
   arenas + mapped doorbell pays page walks. `iommu.pt=on`: +1-3% on the DMA lanes, one grubby line +
   reboot, low risk.
3. **Clock pin searches for 3135/14001 and lands on the derate (2962/13801) each boot** — the frontier
   implicates boost/voltage transitions in Xid 79 on this exact GPU class; pin the derate directly
   (2950/13800), zero perf delta, stability gain.
4. L0 tripwire baseline is 1418 (historical), flat under serving — log the baseline per battery (hygiene).
5. **CCD split verified LIVE per-thread** (pool workers 8-15, lane service 0-7, GPU IRQs inside
   irqaffinity) — the T35/F-201 doctrine holds; the one remaining lever is a second worker bank via
   STRATA_POOL_CORES after widening the taskset (+0-15% if IMC core-starved; falsification history loud —
   one ABAB battery, env-only).
6-10. NEGATIVES (verified optimal, do not touch): THP madvise carrying the arenas (93.6 of 126.5 GiB anon
hugepage-backed; standing-pool doctrine holding); zero-copy mapped staging (CE-vs-SM question does not
arise); REBAR on and correctly unused; SMT one-worker-per-core; isolation+swap-guard+governor armed exactly
as doctrine. Micro-residual: min_free_kbytes/watermark_scale stock defaults (+0-1%, optional).

## The headline
Up to ~10% of headroom sits in the BINDING constraint (DRAM lane) from device-level changes with zero
engine code: EXPO/subtimings (+2-5%), iommu.pt (+1-3%), 1DPC DIMM swap (+5-10% ceiling). Verify EXPO state
first — one root command.

Sources: Level1Techs/Puget 2DPC; ocinside AM5 guide; overclock.net tREFI; DDR5 tuning blueprint; Chips and
Cheese 9950X; NVIDIA Xid 79 thread + forums; Stanford PGL; ReBAR paper; Phoronix EPYC SMT; Petronella
9950X3D; Lei Mao pinned page tables; kernel THP docs; AMD Instinct IOMMU docs; Blackwell whitepaper.
