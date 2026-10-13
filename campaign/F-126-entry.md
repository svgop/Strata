
### F-126 — THE DMA SHARE TESTED: `--pcie-frac 0.9` cuts the CPU term and costs more on the GPU side - REFUTED (2026-10-12 ~17:40, goal round 3)

**THE HYPOTHESIS (from F-125).** F-125 named the CPU expert path as the term that eats the draft yield
(13.56 of 29.12 ms at T 4.38, ~182 CPU experts/window at ~75 us). `--pcie-frac` decides what share of the
MISSES is served by DMA instead of CPU compute (`src/core/verify.cpp:1618`: "the PCIe share is pcie_frac
of the misses"). So: raise the share, cut the CPU term, convert it into copy-engine work that the GPU
does not have to think about. Config-only, one line.

**THE TEST.** Two boots, same binary, same launcher arms, same four request classes, same sweep
(`campaign/draft-yield-sweep.ps1`): boot 17:32:28 at `--pcie-frac 0.35` (the adopted value) vs boot
17:36:11 at `--pcie-frac 0.9` (config `strata-iq3_s-pcie90.json`, launcher
`launch-v2-diag-pcie90-8081.bat`). Sources: `campaign/results/sweep-margin05-batch4.txt`,
`sweep-diag-pcie90.txt`.

| class | T | CPU term 0.35 -> 0.9 | GPU wait 0.35 -> 0.9 | ms/window 0.35 -> 0.9 | t/s 0.35 -> 0.9 |
|---|---|---|---|---|---|
| repetitive list | 2.87 -> 2.25 | 4.70 -> **2.96** | 8.58 -> 9.61 | 17.05 -> 16.58 | 158.9 -> 132.1 |
| code | 4.01 -> 3.93 | 7.11 -> **6.15** | 9.54 -> **15.57** | 20.91 -> **27.06** | 184.1 -> **142.3** |
| prose | 2.69 -> 2.01 | 5.93 -> **2.77** | 8.37 -> 9.06 | 18.32 -> 15.02 | 140.8 -> 129.2 |
| literal repeat | 2.52 -> 2.53 | 3.69 -> **2.49** | 8.47 -> 9.31 | 15.39 -> 15.10 | 152.7 -> 156.3 |

**THE VERDICT: the mechanism works and the trade loses.** At matched depth (code, T 4.01 vs 3.93) the CPU
term falls 0.96 ms and the GPU wait rises **6.03 ms** - the DMA transfer is now on the critical path and
the GPU has nothing to do while it lands (F-125's "miss exposed = GPU idle"). Net: 20.91 -> 27.06 ms per
window, 184.1 -> 142.3 t/s. Three of four classes lost. The adopted 0.35 split stands; the engine's
default split was already near this machine's optimum, which is consistent with the playbook's own
reading of this box (PCIe Gen5 ~= RAM bandwidth, so neither path dominates).

**WHAT THIS CLOSES AND WHAT IT LEAVES.** Closed: "just DMA the misses" - the CPU term is not removable by
moving the bytes to the copy engine; the cost reappears as exposed GPU wait. Left standing as the only
lever that removes the term without spending VRAM, quality or PCIe: **overlap the CPU tier with the GPU
work** (compute the layer's CPU experts while that layer's GPU kernel runs, instead of the verify
blocking on them). That is engine work, and it is now the top item.

**RECORDED AS AN HONEST NEGATIVE.** No adoption; the standing lane was restored to the adopted recipe
(`--pcie-frac 0.35`, margin 0.5, batch 4) immediately after the sweep, and the diagnostic config/launcher
stay in the tree only as evidence.
