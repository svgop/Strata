# T33 — Community driver unlocks: BF16 tensor unlock (Zagitta/gpu-unlock) and consumer P2P (aikitoria/open-gpu-kernel-modules)

- **Date:** 2026-10-22 (all repo states checked today unless noted)
- **Trigger:** a tweet claims local-AI prefill on 3090s gains 10–15 % from two community driver upgrades: (1) P2P via aikitoria/open-gpu-kernel-modules, (2) BF16 GPU unlock via Zagitta/gpu-unlock.
- **Context:** our box is a single RTX 5090 (GB202, sm_120, 32 GB). F-252 already measured consumer-Blackwell half-rates fp32-accumulate tensor math (476 TFLOPS fp16-accum vs 238 fp32-accum; GeForce Blackwell whitepaper documents the 2:1). F-252 also tracked Zagitta/gpu-unlock as Ampere-only at that time — still true, and the repo has since grown measurably.
- **Status:** watch-only stance **unchanged**. Nothing here applies to our single-GPU sm_120 box today.

---

## Summary verdict

1. **The BF16 unlock is real, new, and Ampere-only.** Zagitta/gpu-unlock (first public release 2026-09-28, repo created 2026-10-18) lifts the BF16/TF32 input-type half-rate throttle on GA102/GA104 GeForce by writing "speed selector" registers (SS0 = 0x88888888 → "full BF16 rate") before the stock driver binds. Measured community numbers: 3080 Ti BF16 71.59 → 113.98 TFLOPS (+59 %), TF32 35.65 → 65.28 (+83 %) (issue #2). **No Ada or Blackwell support, no GB202/5090 issue or PR exists** (closest: an AD103/4080-SUPER port discussion opened today with offline evidence only, no code, no measured gain).
2. **The lock this tool lifts is not our lock.** Our measurable 5090 penalty is **fp32-accumulate 2:1** (F-252). Zagitta's unlock addresses the **BF16/TF32 input-rate** throttle; fp32-accum is never discussed or measured in the repo, even on Ampere. On GB202, bf16/fp16 with fp16-accum already runs at the full rate we measured (476 TFLOPS), i.e. the Ampere-style BF16-input lock is not what bites us. Even a perfect GB202 port of this tool would likely gain us ~nothing unless it also moved the fp32-accum rate — which is unproven on any architecture.
3. **P2P is multi-GPU only.** aikitoria/open-gpu-kernel-modules (fork of tinygrad's fork, rebased to NVIDIA 615.71.09, last push 2026-09-16) genuinely re-enables BAR1/NVLink P2P for RTX 30/40/50-series including 5090 and RTX PRO 6000 — but P2P needs ≥2 devices. **Zero benefit on a single-GPU box.** The only single-GPU-adjacent extra is an experimental hugepage `cudaHostRegister` fast path (marginal; not relevant to VRAM-resident inference).
4. **No Blackwell tensor unlock exists anywhere** (banked negative, checked 2026-10-22). The cmpunlocker ecosystem (its actual lineage) is GA100/CMP-170HX-only; the one Blackwell vGPU research repo notes Blackwell gates look **fuse-enforced** ("GSP firmware blocked by fused-off VF PRIV registers"), unlike GA100's firmware/OTP gates — reason for sustained skepticism that a CMP-style register unlock can move GB202 tensor rates at all.
5. **Tweet number check:** the "10–15 % prefill gain on 3090s" is **plausible but unverified end-to-end**. Mechanism supports it (bf16 GEMM-bound prefill on Ampere GeForce runs bf16 at half rate stock; ≤2× headroom on those GEMMs). No independent prefill benchmark found; the originating r/LocalLLaMA thread could not be fetched directly (blocked) — see methods note. The "P2P gains prefill 10–15 %" half of the claim can only be true for **multi-GPU** rigs (tensor-parallel prefill); on one card it is void.

---

## a) Zagitta/gpu-unlock — today

**Repo state (2026-10-22):** created 2026-10-18; commits: 0ebdb1c "Initial public release" (2026-09-28), bb1088b "Add all GA102 and GA104 device ids based on user reports" (2026-10-21), 1799023 "Add support me info" (2026-10-21). 226 stars, 37 forks, ~7 open issues. Not a GitHub fork (`fork: false`); no parent repo.

**Exact mechanism (from README):** a Linux kernel module shipped in the **initramfs** that runs **before the stock NVIDIA driver binds** the card. It is explicitly "neither a UEFI application nor a patched NVIDIA driver" — no driver file is patched. Opt-in via kernel cmdline `gpu_unlock.targets=0000:21:00.0` (or `all`). It "runs the unlock, writes the speed selectors (SS0 0x88888888, which selects the full BF16 rate, and the 4-bit SS1 0x8, whose effect is untested) and performs FLR before the unmodified NVIDIA driver binds the card." Requires a **cold card** (warm reboots refused); concurrent unlock ~1 s. Technique = "the CMP unlocker's V67/PLM technique" combined with "the pre-driver timing of the UEFI approach". GPL-2.0 module; payload derives from a pinned CMP reference.

**What the speed selectors are (via the cmpunlocker lineage):** SS0/SS1 are SM compute-throughput throttle registers behind NVIDIA's Platform Lock Manager (PLM). On the GA100 CMP 170HX (amoghmunikote/cmpunlocker, 782 stars, updated 2026-10-22: "unlocks full SM compute throughput (SS0/SS1)", memory geometry, PCIe Gen2, ECC, BAR1), the d3dx9 fork describes the exploit chain: the Falcon BootROM "loads the .fwsignature_ga100 ELF section content into DMEM **before** verifying the signature"; a ~63 KB ROP chain replaces it, one BAR0 write of 0xFFFFFFFF opens the PLM registers (plus WPR_CFG/FBPA/WPR/FEAT), then memory-unlock (CFG1, LMR) and compute-unlock (SS0, SS1) values are written via BAR0 — a daemon re-writes SS0/SS1 every second because the driver resets them. Zagitta's flow avoids the daemon by writing pre-driver + FLR.

**Supported architectures:** **GA102 and GA104 only** (GeForce + CMP device IDs; "professional RTX A-series and data-centre parts are not" accepted; "Other chips are rejected"). Only **one RTX 3090** verified by the author (CachyOS 7.2.8-1-cachyos kernel, NVIDIA 615.71.09, alongside an unmodified control GPU). No GA104 card tested; the GA102 firmware/V67 image is used for all.

**Blackwell / GB202 / RTX 5090 status:** **nothing.** All ~7 issues read (2026-10-22): #1 RTX 3090 Ti device-ID request; #2 "net result" Q; #3 older-kernel fix; #5 bus-mastering bug; #7 install help; **#4 (opened today, 2026-10-22) "AD103 / RTX 4080 SUPER support: offline Booter evidence and remaining validation gates"** — the only non-Ampere item. Issue #4 is port *discussion*, not a port: offline comparison shows AD103 booter assets byte-identical to AD102's, but the author states they have "not run an unlock or modified the GPU", have measured no gain ("the published Tensor microbenchmarks do not establish one"), and lists remaining gates: verifiable AD103 V67 payload, "effective AD103 PLM/SS writable masks and speed encoding", validated handoff to the stock driver. **No issue or PR mentions GB202, RTX 5090, or Blackwell at all.**

**Measured numbers:** README has none. Issue #2 (opened 2026-10-18): Ph0rk0z claims "Faster BF16/F16 operations. About 40 % gain"; solaasan measured on a **3080 Ti power-limited to 315 W** (kernel 7.2.11-1-cachyos-server, driver 615.78.08): **BF16 71.59 → 113.98 TFLOPS** (~+59 %), **TF32 35.65 → 65.28 TFLOPS** (~+83 %). These are consistent with lifting the Ampere GeForce input-type half-rate locks (BF16, TF32). **FP16/BF16-with-FP32-accumulate full rate: not claimed, not measured, not discussed anywhere in the repo.** No LLM/prefill/GEMM/llama.cpp/vLLM workload numbers anywhere in the repo.

**Upstream history:** credit goes to the "CMP unlocker" (amoghmunikote/cmpunlocker and its fork family — d3dx9, pearlfortune, Cyridd CMP-40HX, etc.) plus an unnamed "UEFI approach" for pre-driver timing. **No "Fateev" repository or author exists** in GitHub repo search or web search (multiple query forms, 2026-10-22) — the "forked from Fateev's original?" hypothesis in our journal is **not supported**; the closest famous namesake is DualCoder/vgpu_unlock, which is an unrelated vGPU (VGX/GRID) unlock.

**Upshot for us:** Ampere-only today; lifts a lock we don't suffer from; no Blackwell port path beyond one evidence-free Ada discussion. Watch item, not an action item.

**What would change our stance:** (i) an issue/PR landing GB202 "PLM/SS writable masks" evidence (the same gate issue #4 defines for AD103), or any repo demonstrating speed-selector writes changing **any** tensor rate on Blackwell GeForce; (ii) a measurement showing the SS lock governs the **fp32-accum** rate (on any arch) — then the tool class becomes directly relevant to our 2:1; (iii) an independent, reproducible end-to-end prefill benchmark on a 3090-class card in a serving engine (would validate the tweet's claim class generally).

---

## b) aikitoria/open-gpu-kernel-modules

**Identity and activity:** fork of **tinygrad/open-gpu-kernel-modules** (itself a fork of NVIDIA/open-gpu-kernel-modules), created 2024-10-14, 601 stars, 70 forks, **0 open issues**, branch `615.71.09-p2p` based on NVIDIA **615.71.09** (upstream version-bump commit 2026-09-09). The four custom commits are all dated 2026-09-11 and co-authored with the tinygrad crew (geohot, nimlgen, valdemardi, ducphuc, xycjscs) — this is effectively **tinygrad's P2P work rebased to the current driver, plus extras**. Last push 2026-09-16. (tinygrad's own fork mainline is older, 565.57.01-p2p, PCI table ending at Ada; aikitoria's is the current Blackwell-capable line.) The tinygrad fork's purpose is one line: "NVIDIA Linux open GPU with P2P support."

**What exactly is re-enabled (commit e698b05, 17 files, +331/−93):** GPU-to-GPU P2P over **PCIe BAR1** (and NVLink where present), including **mixed-generation pairs** (e.g. 3090 + 5090, which stock NVIDIA's proprietary PCIe P2P protocol refuses; those need an additional userspace `patch-libcuda-p2p.py`). Concretely:
- `kernel_bif.c`: P2P reads/writes forced enabled; `pcieP2PType` defaults to BAR1.
- Generated HAL tables (`g_kern_bus_nvoc.c`, `g_virt_mem_allocator_nvoc.c`): BAR1 P2P support extended to **Ampere (GA100–GA107) and Ada (AD102–AD107)**, not just Hopper/Blackwell.
- `p2p_caps.c`: new `areAllGpusP2PCompatible` — BAR1 becomes an architecture-independent transport; the proprietary protocol only gates its own fallback.
- `uvm_devmem.c`: skips PCI P2PDMA static-BAR1 registration on **non-coherent Blackwell (GB100+)** — i.e. consumer Blackwell is explicitly handled in-tree.
- `nv-reg.h`: `__NV_ENABLE_RESIZABLE_BAR` default flipped 0 → 1.
- Plus per-arch bus work in `kern_bus_gm107/gm200/gp100/tu102.c` and BAR1 aperture tracking in `io_vaspace.c`/`virt_mem_allocator_gh100.c`.

**Requirements:** Resizable BAR + Above-4G Decoding in firmware ("each GPU needs … a BAR1 aperture large enough to map its usable VRAM"); IOMMU passthrough `iommu=pt` (with an explicit warning it removes DMA isolation); **ACS disabled** if P2P is slow ("ACS on root ports forces all GPU-to-GPU traffic through the CPU root complex, killing P2P bandwidth"). Supported cards: RTX 30/40/50 series (including below the x90 models) and RTX PRO 6000 Blackwell; **Turing (RTX 20) not supported on the BAR1 path**.

**README benchmarks (author-reported):** 9-GPU rig (1× RTX PRO 6000 Blackwell + 8× RTX 5090, dual-socket EPYC 9575F): P2P writes ~55–57 GB/s per pair, bidirectional ~111 GB/s, latency ~14 µs (copy) → ~0.4 µs (P2P); 8× 5090 NCCL `all_reduce` busbw plateaus ~45 GB/s with `NCCL_P2P_LEVEL=SYS`. (Multi-GPU NCCL/tensor-parallel use is exactly the target workload; llama.c/vLLM TP is the implied consumer.)

**Single-GPU relevance: none for P2P** — a P2P transport definitionally requires a peer device. The fork carries exactly one other perf-relevant patch: commit 43cb9fb "Accelerate registration of hugepage-backed host memory" — an **experimental** `cudaHostRegister` fast path for 2 MiB/1 GiB hugepage-backed pinned host buffers. That is the only piece with any single-GPU exposure (host-pinning cost in servers that register host memory); it does nothing for weights/KV resident in VRAM, and there is no measured number for it.

**What would change our stance:** us adding a second GPU (tensor-parallel or pipeline serving) — then adopt this fork or its successor **at the driver version we run** and re-verify (the rebase cadence is ~driver releases; 615.71.09 is current). Until then: nothing. The hugepage-registration path would only matter if our serving path registers large pinned host buffers in a hot loop, and only if it survives out of "experimental".

---

## c) Broader scan (last ~12 months; positives and banked negatives)

**(i) Full-rate tensor paths on consumer Blackwell — no unlock exists.**
- GitHub repo search for 5090/GB202/BF16 unlock projects (2026-10-22) surfaced only: **bird/vgpu-unlock-blackwell** (29 stars, vGPU research on a 5090: "19 binary patches, full CPU-side pipeline working, **GSP firmware blocked by fused-off VF PRIV registers**" — direct evidence that Blackwell feature gates are fuse-enforced, unlike GA100's firmware/OTP gates; it targets virtualization, not tensor rates), **marduk191/ComfyUI-fp8_cublas** (software: routes Linear layers to FP8/MXFP8 on 4090/5090 — a mitigation, not an unlock; premised on FP8 running full-rate), a laptop power-limit ACPI mod, and noise.
- The **cmpunlocker ecosystem is GA100-CMP-only**: amoghmunikote/cmpunlocker (CMP 170HX), forks for CMP 40HX (Cyridd) / 90HX (pearlfortune); none target GeForce or Blackwell. LTT Labs covered the CMP 170HX unlock 2026-09-12 (article returned 403 to our fetcher; substance corroborated via Gigazine 2026-09-14: hidden 64 GB HBM2e on "8 GB" cards, PCIe Gen2, "most of the computing power of the A100").
- Zagitta/gpu-unlock: no GB202/5090/Blackwell issue or PR (all issues read); only the AD103 discussion (issue #4, 2026-10-22) with explicitly no working unlock.
- Official NVIDIA open-gpu-kernel-modules: nothing re-enables restricted GeForce tensor rates; the 2:1 fp32-accum on GeForce Blackwell stands as documented in the GeForce Blackwell whitepaper and as we measured in F-252 (476 vs 238 TFLOPS).
- Academic angle: arXiv:2507.10789 "Dissecting the NVIDIA Architecture with Microbenchmarks" (Jarmusch, Graddon, Chandrasekaran; 2025-07-14, 5080 vs H100) — abstract does not report an fp32-accum ratio; lead only, not yet evidence.

**(ii) P2P / BAR1 / zero-copy for SINGLE-GPU LLM inference — nothing credible.**
- P2P (any transport) requires ≥2 devices; all aikitoria/tinygrad benchmarks are multi-GPU.
- BAR1/ReBAR governs the host↔device aperture; VRAM-resident inference does not stream over BAR1. Host-mapped ("zero-copy") inference is PCIe-bound (Gen5 x16 ≈ 64 GB/s) vs ~1.8 TB/s device bandwidth — strictly worse whenever the model fits in VRAM. (Architectural reasoning, flagged as inference, not a sourced claim.)
- Software-side mitigation for our actual 2:1 penalty is **engine-level, not driver-level**: PyTorch ≥2.7 `torch.backends.cuda.matmul.allow_fp16_accumulation` (FP16 GEMMs accumulate in FP16; documented in PyTorch CUDA semantics notes; shipped by ComfyUI as an opt-in; a cuBLASLt crash with the flag fixed in PyTorch 2.8) is the same lever as choosing fp16-accum kernels in our own engine — already in F-252's scope. No driver/firmware lever for fp32-accum was found anywhere.

**(iii) Watch-list changes vs. F-252's stance.**
- F-252 said gpu-unlock was "Ampere-only (GA102/104) at that time" — **still true and now better characterized**: the repo is brand-new (Sept/Oct 2026), mechanism is speed-selector register writes (not a driver patch), measured gains exist (issue #2), and an Ada port discussion (not implementation) opened 2026-10-22.
- New watch item: aikitoria fork is the maintained, **Blackwell-capable** P2P line (tinygrad lineage) — becomes directly relevant only on a multi-GPU box.

---

## d) VERDICT table

| Item | Applies to our single-GPU sm_120 box NOW | APPLIES LATER (trigger) | DOESN'T APPLY (why) |
|---|---|---|---|
| **Zagitta/gpu-unlock** (BF16/TF32 speed-selector unlock) | No | Watch: GB202 "PLM/SS writable-mask" evidence landing (cf. issue #4's gates for AD103); any demo of SS writes changing a Blackwell tensor rate | GA102/GA104 only; only 3090 verified; and the lock it lifts (BF16/TF32 **input rate**) is not our measurable loss (**fp32-accum 2:1**, F-252) — bf16/fp16 @ fp16-accum already runs full-rate (476 TFLOPS) on our card |
| **aikitoria/open-gpu-kernel-modules** (P2P BAR1/NVLink) | No | **Yes, on a 2nd GPU**: tensor-parallel/pipeline serving; then adopt at our driver version (currently 615.71.09) and re-verify | P2P needs ≥2 peers — zero effect on a single GPU. Extra hugepage-`cudaHostRegister` path is experimental and irrelevant to VRAM-resident inference |
| **cmpunlocker ecosystem** (GA100 Falcon BootROM/PLM) | No | Watch only: someone demonstrating the Falcon/PLM approach on GB202 — currently contradicted by fuse-based Blackwell gating (bird/vgpu-unlock-blackwell) | CMP 170HX (GA100) device IDs only; all forks target CMP variants |
| **PyTorch-style fp16-accum GEMM choice** (engine lever, not driver) | **Yes — already in F-252 scope** | — | Not a driver upgrade; numerics-gated |

**Honesty on P2P for a single GPU:** there is no scenario on a one-GPU box where BAR1/NVLink P2P code paths execute meaningful work. The tweet's framing only holds for multi-3090 rigs.

---

## Unverified / could-not-verify (explicit)

- The tweet's **"10–15 % prefill gain on 3090s"**: no independent end-to-end prefill benchmark found (GitHub issues, web search). Mechanism-plausible; unproven. The originating r/LocalLLaMA thread ("Everyone go buy RTX 3090s NOW while you still can", id 1x91u12) describes the 3090 unlock as "a few more flops after applying the same hack as cmp170" on the prefill path for local LLM work — the thread itself is fetch-blocked for us (see methods).
- A search-engine snippet claiming a **"7.9× prefill gain"** tied to this unlock: **could not be substantiated by any direct source**; treat as noise. (A BF16 unlock cannot exceed 2× on bf16 GEMM-bound prefill.)
- LTT Labs CMP 170HX article (2026-09-12): returned HTTP 403; content taken secondhand from Gigazine's coverage — flagging rather than quoting primary text.
- Whether the fp32-accum 2:1 on GB202 is governed by any PLM/speed-selector register at all: **unknown**; no public evidence either way.
- Whether RTX PRO 6000 Blackwell (full GB202) runs FP16/BF16-with-FP32-accumulate at full rate (which would make it the natural "unlock target" state): spec pages found (TechPowerUp, Fixstars, StorageReview, NVIDIA RTX PRO Blackwell whitepaper) were not reconciled to a clean FP16-fp32acc figure; **left open**, not asserted.

## Sources (all accessed 2026-10-22)

Code / repos (GitHub):
1. https://github.com/Zagitta/gpu-unlock — README (mechanism, SS0=0x88888888, GA102/GA104-only, cold-card, initramfs).
2. https://api.github.com/repos/Zagitta/gpu-unlock (+ /commits) — created 2026-10-18, 3 commits, not a fork, last push 2026-10-21.
3. https://github.com/Zagitta/gpu-unlock/issues (incl. #1, #2, #4, #5) — measured numbers in #2; AD103 discussion #4 (2026-10-22); no Blackwell items.
4. https://api.github.com/repos/aikitoria/open-gpu-kernel-modules — fork of tinygrad's, branch 615.71.09-p2p, 0 open issues, pushed 2026-09-16.
5. https://github.com/aikitoria/open-gpu-kernel-modules — README (cards incl. 50-series/RTX PRO 6000, ReBAR/iommu=pt/ACS requirements, 5090 rig benchmarks).
6. https://api.github.com/repos/aikitoria/open-gpu-kernel-modules/commits/e698b05 — per-file patch breakdown (kernel_bif.c, p2p_caps.c, uvm_devmem.c GB100+, ReBAR default, GA10x/AD10x BAR1 P2P).
7. https://github.com/tinygrad/open-gpu-kernel-modules — upstream P2P fork, 565.57.01-p2p, table ends at Ada.
8. https://github.com/amoghmunikote/cmpunlocker — CMP 170HX unlock, "Full SM compute throughput (SS0/SS1)", nvidia-open 610.xx+.
9. https://github.com/d3dx9/cmpunlocker — Falcon BootROM signature-check bug, PLM registers, SS0/SS1 via BAR0.
10. https://api.github.com/search/repositories?q=cmpunlocker and q=5090+unlock… — ecosystem scan; bird/vgpu-unlock-blackwell ("fused-off VF PRIV registers"); marduk191/ComfyUI-fp8_cublas; DualCoder/vgpu_unlock (unrelated vGPU lineage).

Vendor docs / specifications:
11. NVIDIA RTX Blackwell GPU architecture whitepaper: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf (documents the GeForce fp32-accum 2:1; ratio already banked by our F-252 measurement 476/238 TFLOPS).
12. PyTorch CUDA semantics (allow_fp16_accumulation): https://docs.pytorch.org/docs/stable/notes/cuda.html

Literature:
13. arXiv:2507.10789 — Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks (2025-07-14; abstract checked; fp32-accum ratio not covered → lead only).

Practitioner / community:
14. r/LocalLLaMA thread 1x91u12 "Everyone go buy RTX 3090s NOW while you still can" — origin of the tweet-class claim; fetch-blocked, snippet-level only.
15. Gigazine, 2026-09-14: https://gigazine.net/gsc_news/en/20260914-cmpunlocker — cmpunlocker coverage (Amogh Munikote; 64 GB reveal; secondhand for LTT Labs).
16. LTT Labs, 2026-09-12: https://www.lttlabs.com/articles/2026/09/12/cmp-170hx-cmpunlocker (403 for us; referenced via #15).
17. Hacker News 44997957 ("5090's theoretical BF16 TFLOPs", Sept 2025) — community awareness of the 2:1; fetch timed out, snippet-level only.

## Methods note (searches that did not pay off, so they aren't repeated)

- Direct fetches of the r/LocalLLaMA thread blocked via www/old/.json and pullpush archive (empty) — snippet-level corroboration only; do not retry without a Reddit-authenticated path.
- lttlabs.com 403; news.ycombinator.com timed out twice; github.com issue pages for the aikitoria fork failed to render (fork has 0 open issues per API — use the API for that repo).
- "Fateev gpu-unlock" (multiple phrasings): zero hits — journal's lineage guess retired.
- WebSearch engine summaries twice volunteered unsourced figures ("7.9× prefill", "NVIDIA shipped 1.9× optimizations") that no primary source supported; both excluded from findings.
- Early generic searches ("BF16 unlock benchmark prefill") returned generic 3090 benchmark content, nothing unlock-specific — the specific claims live only in the Zagitta repo issues and the (blocked) Reddit thread.
