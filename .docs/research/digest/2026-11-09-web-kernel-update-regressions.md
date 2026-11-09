---
type: digest
id: w10-kernel-update-regressions
created: 2026-11-09
status: evidence
topic: Do kernel point updates silently regress CUDA/LLM inference throughput; detection and pinning practice. Applied to our case: fc45 7.2.12-300 -> 7.2.15-300 via PackageKit offline-update mid-experiment.
scope: read-only research; local evidence gathered with rpm/git patch census; no repo files modified besides this digest
---

## 1. Verdict for OUR CASE (7.2.12-300.fc45 -> 7.2.15-300.fc45)

**Prior probability that the kernel *content* caused a 5-20% throughput delta: LOW.** Evidence is local and primary:

- Upstream stable range v7.2.12..v7.2.15 = **334 commits**, censused by subject (`github.com/gregkh/linux/compare/v7.2.12...v7.2.15.patch`, fetched 2026-11-09): almost entirely correctness fixes (ksmbd, pinctrl-qcom, ASoC bits, hv, scsi, tracing, sched_ext UAFs). **Zero commits touching cpufreq/amd-pstate/intel_pstate, PCIe ASPM, THP/large folios/compaction, or IOMMU mode.**
- Only two scheduler hot-path entries exist: `sched/core: Call wakeup_preempt() for queued tasks` and `sched/fair: Avoid overflow in place_entity()` (EEVDF placement). Both are overflow/correctness-shaped; could shift wakeup-latency distributions, not 10-20% of GPU throughput.
- amdgpu entries are leak/teardown fixes (userptr GEM leak, sched entity leak, TTM mutex, runtime PM double-drop, eviction fence lifetime). The runtime-PM double-drop fix could change *power state residency* on amdgpu hardware; on our NVIDIA stack (akmod-nvidia 615.71.09) it is inert.
- Fedora-side package delta is nil: `rpm -q --changelog kernel-core-7.2.15-300.fc45` shows 7.2.13/14/15 packages carry "Linux v7.2.x" plus unrelated Fedora backports (media/cvs, drm/vmwgfx, net/skbuff zerocopy). No Fedora config flips in range.
- Bodhi FEDORA-2026-7a0f444c64 (kernel-7.2.15-300.fc45): medium-severity security update, pushed stable 2026-11-03, karma 10, kernel regression test case PASS, zero tester regressions. https://bodhi.fedoraproject.org/updates/FEDORA-2026-7a0f444c64

**Therefore: the risk to our A/A is not the kernel diff, it is the *uncontrolled transition*** — first-reboot state, akmod rebuild bookkeeping, offline-update artifacts, and anything PackageKit touched at 14:39. The corpus below (section 2) shows point releases CAN carry 20%-to-42x bombs (7.0.12 HMM case), so the A/A is still mandatory, not optional. Confidence: High (local primary evidence).

Confidence grades used below: **High** = two+ independent primary sources or primary doc read; **Medium** = single primary source or reputable secondary; **Low** = forum anecdote, fetch failed, search-summary only.

## 2. Documented kernel-caused inference/compute regressions (the corpus)

- **Upstream 7.0.12, AMDGPU HMM retry loop -> ROCm compute collapse, up to 42x.** SDXL via ComfyUI went **9 s -> 388 s** on AMD GPUs. Flagged 2026-06-15 on Fedora 44 moving 7.0.11 -> 7.0.12 (ROCm issue #6358); shipped to Ubuntu 26.04/24.04-HWE as 7.0.0-28.28; Ubuntu Kernel Team publicly advised ROCm users to skip it; fixed in stable 7.0.13 by commit dd03ceed9fc71fa8f6c3ae44d395ac53ea2a0dd0. Mechanism: "throughput degradation, not lockup" — a retry loop in heterogeneous memory management. Confidence: **High**. https://www.phoronix.com/news/Ubuntu-7.0-AMDGPU-Regress , https://discourse.ubuntu.com/t/amdgpu-performance-regression-in-kernel-7-0-0-28-28/85237
- **RHEL 10.2 newidle-balance NI_RANDOM scheduler regression -> up to -93% throughput.** CockroachDB 60% reads/c=128 on m6a.24xlarge: **-93%**; per-instance -68/-65/-31/-15%, geomean **-42%**. Introduced RHEL 10.2 (6.12.0-211.7.1) vs 10.1; upstream lineage = Chris Mason newidle-overhead patches + Zijlstra randomization (LKML Nov 2025), v6.12 backports. Symptom is **bimodal** (bistable flip between full speed and CPU starvation, 85-95% idle) — this is the exact signature that would look like a mysterious A/A shift. Runtime mitigation, no reboot: `echo NO_NI_RANDOM > /sys/kernel/debug/sched/features`. Confidence: **High**. https://developers.redhat.com/articles/2026/08/04/mitigate-newidle-linux-kernel-scheduler-regression
- **Fedora Workstation 44 distro-wide slowdown via Tuned-PPD + intel_pstate.** Phoronix six-distro round-robin found Fedora "strangely slow"; root cause: intel_pstate "incorrectly handling the case where the governor is set to performance" (Tuned-PPD sets performance; PPD does not). Fix: `intel_pstate: Adjust policy->cur in active mode to policy`, merged for 7.3 and backported to stable + Fedora 44. Magnitude numbers live in the linked reviews, not re-verified here. Confidence: **High** on mechanism, **Medium** on magnitude. https://www.phoronix.com/news/Fedora-Tuned-PPD-Perf-Fix
- **Linux 7.2 DRM scheduler "fair" default -> sustained-load stalling, reverted to FIFO inside 7.2.** RX 9070 XT under 100% GPU load: "roughly 10 fps or freezes outright"; Wayland compositor lockups. 20 revert patches merged before 7.2 final (release 2026-08-16). Relevance to us: the 7.2.y stable series is still absorbing the fallout of this policy churn, and our 7.2.12->7.2.15 window is inside it (census found no DRM-scheduler-policy commits in range, so the fallout fixes did not land here). Confidence: **High**. https://www.phoronix.com/news/Linux-7.2-Reverting-DRM-Fair , https://www.linuxjournal.com/content/linux-72-reverts-drm-scheduler-change-after-serious-gpu-regressions
- **ASPM policy change class: NVMe ASPM-disable patch -> -23% IOPS.** linux-nvme 2023-07 report: an ASPM-disable patch cost 23% IOPS until reverted. This is the mechanism class for "kernel update flips PCIe power-state policy and throughput moves": ASPM state per link is visible in `lspci -vv` (LnkCap/LnkCtl). Confidence: **Medium** (single mailing-list thread). http://lists.infradead.org/pipermail/linux-nvme/2023-July/041139.html , knob reference: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/power_management_guide/aspm (`pcie_aspm=off`)
- **ROCm stack kernel-version sensitivity, Strix Halo (kyuz0/amd-strix-halo-toolboxes #45).** "Kernel 6.17.9 breaks ROCm, but 6.17.7 works"; one kernel fix backported to 6.17.9 made older ROCm incompatible (VGPR count gfx1151). Measured pp512 swings: gpt-oss-120B Q6_K **673.78 -> 282.76 t/s (-58%)** from stack mismatch, restored to 631.80 by a later fix. Attribution discipline note: the maintainer proved the -58% was userspace (ROCm kernel selection), NOT the Linux kernel — the canonical example of misattributing a stack regression to the kernel. Confidence: **High**. https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/45
- **GSP (GPU System Processor) across kernel/driver updates.** Three threads: (a) RTX 5090 **Xid 79 GSP firmware crash under sustained CUDA load** with fix analysis — https://forums.developer.nvidia.com/t/bug-report-fix-rtx-5090-xid-79-gsp-firmware-crash-under-sustained-cuda-load/369440/5 ; (b) **GSP performance regression in context initialization** (latency on context creation, general slowness; disable-GSP resolves) — https://forums.developer.nvidia.com/t/performance-regression-with-gsp-firmware-and-context-initialization/306019 ; (c) NVIDIA vGPU known issue: **GSP cannot be disabled with open kernel modules** (`NVreg_EnableGpuFirmware=0` not honored) — https://docs.nvidia.com/vgpu/20.0/known-issues/bug-6353226-gsp-firmware-cannot-be-disabled-with-open-kernel-modules.html . Direct fetches of (a)/(b) timed out; graded from search corroboration. Key fact for our A/A: GSP firmware version comes from the driver package, not the kernel — it must be identical across both boots. Confidence: **Medium**.
- **IOMMU translation cost class.** IOTLB-miss overhead is the mechanism by which DMA translation slows devices (multi-100Gbps study: https://pmc.ncbi.nlm.nih.gov/articles/PMC10280580 ); `iommu=pt` is the standard trusted-host mitigation ( https://kernel-internals.org/iommu/war-stories ). Regressions appear when a kernel changes the default domain type (identity vs DMA). **No IOMMU-default commit in our 7.2.12->7.2.15 range** (census). Confidence: **High** on mechanism, irrelevant to our specific delta.
- **Negative result: no 2026 THP/large-folio regression corpus found.** Searches across LWN/Phoronix/LKML surfaced only 2024-and-earlier mTHP work ( https://lwn.net/Articles/974826 ) and THP-mode tuning lore ( `always` vs `madvise`, cryoSPARC stalls https://discuss.cryosparc.com/t/turning-off-transparent-hugepages/15819 ). Confidence: **Medium** (absence of evidence, searched 2026-11-09).

## 3. Mechanism map: what a kernel update can silently change under CUDA inference

1. **cpufreq/EPP/governor handling** — intel_pstate perf-governor bug (Fedora/Tuned-PPD case); amd-pstate-epp boost-after-resume regressions on 6.17/6.18 ( https://forum.manjaro.org/t/kernel-6-17-6-18-amd-pstate-epp-cpu-boosting-behavior-issues-after-resume-from-sleep/184465 , **Medium**). Host-side prefill/tokenization and pinned-memory paths track CPU clocks.
2. **Scheduler features** — NI_RANDOM newidle balance (-93% case, has a runtime knob); 7.2 added cache-aware scheduling ( https://www.ssdnodes.com/learn/whats-new-in-linux-kernel-7-2 , **Medium**); EEVDF placement overflow fix in our range.
3. **DRM scheduler policy** — fair vs FIFO (7.2 revert case); affects GPU submission fairness under load, mostly multi-client; single-process inference less exposed.
4. **PCIe power management** — ASPM defaults per kernel; also GPU runtime PM (the amdgpu runtime-PM double-drop fix is in our range, NVIDIA side unaffected).
5. **IOMMU/DMA** — default domain type changes -> IOTLB translation cost. Not in our range.
6. **MM** — THP mode, hugepage reservation state, compaction behavior. No commits in our range; our cmdline pins 2M hugetlb (54272 pages) + `hugetlb_free_vmemmap=on`, which is reboot-restore dependent.
7. **Security mitigations** — new mitigation defaults per kernel cost low single digits % on host paths. Not in our range, but verify identical `mitigations=` state across boots.
8. **Driver/ABI rebuild confounder** — akmod-nvidia rebuilds 615.71.09 against each kernel: driver *version* stays constant, but module instance and GSP handshake re-initialize per boot. Xid/GSP instability is a crash-mode risk more than a throughput-mode risk.

## 4. Detection / A-A protocol (what the corpus says to control)

Per-boot invariant checklist (log into the experiment header, both kernels):
```
uname -r; cat /proc/cmdline
cpupower frequency-info | grep -E 'governor|hardware limits'; cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
cat /sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference; cat /sys/devices/system/cpu/cpufreq/policy0/amd_pstate_mode 2>/dev/null
cat /sys/kernel/mm/transparent_hugepage/enabled; grep -E 'HugePages_(Total|Free)' /proc/meminfo
lspci -vv -s <GPU_BDF> | grep -E 'LnkCtl|ASPM|Speed|Width'
dmesg | grep -iE 'aspm|iommu|amd-vi|dmar|gsp|nvrm' | head
nvidia-smi -q | grep -iE 'GSP.Firmware|Driver Version|CUDA Version'
cat /sys/kernel/debug/sched/features   # confirm NI_RANDOM state identical
```
Protocol: alternate boots A/B/A/B (not A/A then B/B) to cancel drift; 1 warmup + >=3 measured reps per boot; lock GPU clocks (`nvidia-smi -lgc <f>`) and fix governor=performance before any rep; `llama-bench` with frozen args for the engine layer, phoronix-test-suite if you want comparable result files (`phoronix-test-suite merge-results` / OpenBenchmarking metadata logs governor+kernel automatically, https://github.com/phoronix-test-suite/phoronix-test-suite ); judge with geometric mean and spread, not single reps (kernel-comparison methodology caveats: https://www.reddit.com/r/linux/comments/zjtsf/a_critique_against_phoronix_benchmarks , **Low**).

Knob-bisect ladder if the A/A shows a delta (each step isolates one mechanism):
1. `NO_NI_RANDOM` into /sys/kernel/debug/sched/features (scheduler feature, no reboot)
2. `pcie_aspm=off` on cmdline (PCIe power states)
3. `iommu=pt` (DMA translation)
4. THP mode flip (`/sys/kernel/mm/transparent_hugepage/enabled`)
5. Governor/EPP/Tuned profile re-assert (`tuned-adm profile`, `cpupower frequency-set -g performance`)
6. `nvidia-smi -q` GSP/driver identity + dmesg Xid scan (driver-side, not kernel)

## 5. Pinning recipe (Fedora serving/experiment box)

```ini
# /etc/dnf/dnf.conf
[main]
excludepkgs=kernel*,kernel-core*,kernel-modules*,kernel-modules-extra*,akmod-nvidia*,kmod-nvidia*
installonly_limit=6
```
- Per-version pin does NOT work on dnf (glob excludes match the whole release line): exclude-all is the only reliable lever, keep N kernels with `installonly_limit` so the good one is never garbage-collected. Source: Fedora discussion #72318 (alciregi: "definitely unable to retain a specific kernel version"), https://discussion.fedoraproject.org/t/pin-or-retain-a-kernel-version-or-increase-the-number-of-kernels-retained/72318 , and Rocky forum `exclude=kernel*` practice, https://forums.rockylinux.org/t/question-regarding-os-pinning-and-kernel-update-prevention/13668 . Confidence: **High**.
- Kill silent apply paths: `dnf-automatic` -> `apply_updates = no` (download only) or `systemctl disable --now dnf-automatic-apply.timer`; PackageKit desktop path — `gsettings set org.gnome.software allow-updates false`, and verify no pending offline payload with `dnf5 offline status` / clean `/var/lib/PackageKit/prepared-updates`. This is exactly the path that burned us (PackageKit offline-update at 14:39). Confidence: **Medium-High** (commands standard, PackageKit specifics from search summary + local incident).
- Boot default: `grubby --set-default=/boot/vmlinuz-7.2.12-300.fc45.x86_64`; verify with `grubby --info=ALL | grep -E 'index|kernel'`.
- Vendor precedent for pinning: Intel llm-scaler FAQ tells vLLM users to "unify Linux kernel versions" via its BKC setup script ( https://github.com/intel/llm-scaler/blob/main/vllm/FAQ.md ); NVIDIA GPU Operator ships a documented workflow for deliberately outdated/pinned kernels ( https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/install-gpu-operator-outdated-kernels.html ). Confidence: **High**.
- Pre-experiment gate: run the section-4 invariant checklist and refuse to start an experiment if `uname -r` changed since the last baseline. (Our gap: no such gate existed; PackageKit rebooted into a new kernel mid-run.)

## 6. Source register

| # | Claim | Grade | URL |
|---|-------|-------|-----|
| 1 | 7.0.12 AMDGPU HMM 42x ROCm regression, fix dd03ceed9fc7 in 7.0.13 | High | https://www.phoronix.com/news/Ubuntu-7.0-AMDGPU-Regress |
| 2 | Ubuntu kernel-team advisory + community impact | High | https://discourse.ubuntu.com/t/amdgpu-performance-regression-in-kernel-7-0-0-28-28/85237 |
| 3 | NI_RANDOM -93% CockroachDB + NO_NI_RANDOM knob | High | https://developers.redhat.com/articles/2026/08/04/mitigate-newidle-linux-kernel-scheduler-regression |
| 4 | Fedora Tuned-PPD intel_pstate perf-governor bug, fix in 7.3/stable | High (mech) | https://www.phoronix.com/news/Fedora-Tuned-PPD-Perf-Fix |
| 5 | 7.2 fair-DRM-scheduler revert to FIFO | High | https://www.phoronix.com/news/Linux-7.2-Reverting-DRM-Fair |
| 6 | NVMe ASPM patch -23% IOPS | Medium | http://lists.infradead.org/pipermail/linux-nvme/2023-July/041139.html |
| 7 | ROCm kernel/firmware version sensitivity + -58% attribution case | High | https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/45 |
| 8 | Xid 79 GSP crash under sustained CUDA (RTX 5090) | Medium | https://forums.developer.nvidia.com/t/bug-report-fix-rtx-5090-xid-79-gsp-firmware-crash-under-sustained-cuda-load/369440/5 |
| 9 | GSP context-init latency perf regression | Medium | https://forums.developer.nvidia.com/t/performance-regression-with-gsp-firmware-and-context-initialization/306019 |
| 10 | GSP not disableable on open kernel modules | High | https://docs.nvidia.com/vgpu/20.0/known-issues/bug-6353226-gsp-firmware-cannot-be-disabled-with-open-kernel-modules.html |
| 11 | IOTLB translation cost mechanism | High | https://pmc.ncbi.nlm.nih.gov/articles/PMC10280580 |
| 12 | iommu=pt pass-through practice | High | https://kernel-internals.org/iommu/war-stories |
| 13 | vLLM ROCm kernel-version reports (6.18.56-lts gfx1030) | Medium | https://github.com/vllm-project/vllm/issues/38107 |
| 14 | Fedora pin/retain discussion (per-version pin impossible) | High | https://discussion.fedoraproject.org/t/pin-or-retain-a-kernel-version-or-increase-the-number-of-kernels-retained/72318 |
| 15 | exclude=kernel* practice | High | https://forums.rockylinux.org/t/question-regarding-os-pinning-and-kernel-update-prevention/13668 |
| 16 | Intel llm-scaler kernel-unification BKC | High | https://github.com/intel/llm-scaler/blob/main/vllm/FAQ.md |
| 17 | NVIDIA GPU Operator outdated-kernel workflow | High | https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/install-gpu-operator-outdated-kernels.html |
| 18 | Bodhi FEDORA-2026-7a0f444c64 (7.2.15-300.fc45) | High | https://bodhi.fedoraproject.org/updates/FEDORA-2026-7a0f444c64 |
| 19 | amd-pstate-epp resume/boost regression 6.17/6.18 | Medium | https://forum.manjaro.org/t/kernel-6-17-6-18-amd-pstate-epp-cpu-boosting-behavior-issues-after-resume-from-sleep/184465 |
| 20 | llama.cpp in-process slowdown confounder (#22360) | Low | https://github.com/ggml-org/llama.cpp/issues/22360 |
| 21 | 7.2 cache-aware scheduling feature note | Medium | https://www.ssdnodes.com/learn/whats-new-in-linux-kernel-7-2 |
| 22 | Local primary: 334-commit census v7.2.12..v7.2.15, rpm changelog, cmdline, akmod inventory | High | github.com/gregkh/linux compare + local rpm (this digest section 1) |
