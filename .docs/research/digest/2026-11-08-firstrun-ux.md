---
type: digest
id: firstrun-ux
created: 2026-11-08
status: evidence
topic: First-run environment detection and guidance UX for a local inference engine with a reserved-hugepage memory architecture — how Ollama/LM Studio/Jan/llama.cpp detect and phrase suboptimal states, how NVIDIA/CUDA tooling gates on driver state, how doctor-style tools (brew doctor, react-native doctor, systemd-analyze security, tuned-adm verify, lm-sensors) earn user trust, how the fix-needs-root boundary is crossed (pkexec, paste-blocks, fix-as-systemd-unit), and a recommended UX spec for our engine's first-run memory check
scope: read-only web research against primary sources (source files, manpages, official docs) + live local probes on this rig (systemd-analyze security, tuned-adm verify, nvidia-smi -q, all run 2026-11-08). Fresh fetches carry inline URLs dated 2026-11-08. Confidence grades: measured-public (primary source fetched) / measured-ours (executed on this rig) / design-inference (our spec). No standing-falsified entries touched.
---

# First-run detection and guidance UX — evidence digest (2026-11-08)

## 1. Question

Our engine (llama.cpp lineage) has a reserved-hugepage arena that must be armed per device: kernel cmdline/sysctl, a pool size that depends on installed models + RAM, plus THP state for the fallback class and an RLIMIT_MEMLOCK floor for mlock paths. Required UX: (a) DETECT suboptimal state at first run, (b) EXPLAIN it in one clear paragraph, (c) point at a tool/command that fixes it, (d) never nag after it is fixed. Boot must print honest page-class banners (we have: `ARENA PAGES: 2 MiB/4K DEGRADED CLASS`). This digest collects the strongest precedents for each of the four moves and lands a spec.

## 2. Findings

### 2.1 Local-AI tools — what they detect and how they phrase it — measured-public

**Ollama** (source: https://raw.githubusercontent.com/ollama/ollama/main/server/sched.go, fetched 2026-11-08) — the strongest precedent in the class. Its scheduler does a NUMERIC PREFLIGHT before spawn, narrates its own adaptive decisions, and fails with required-vs-available:

- Preflight log lines carry the numbers, not adjectives: `"llama-server model predicted to exceed available memory, evicting"` and `"llama-server model fits alongside existing models"`, each with structured fields `predicted`, `predicted_num_ctx`, `num_batch`, `available`, `gpu_free`, `system_free`, `system_limited`.
- Inventory lines: `"system memory"` (fields `total`, `free`, `free_swap`) and `"gpu memory"` (fields `id`, `library`, `available`, `free`, `minimum`, `overhead`); `"updated VRAM based on existing loaded models"` when accounting for co-resident models; `"predicted usage exceeds VRAM"` as the warning.
- Hard error when admission is impossible: `"model is too large for system memory"`.
- The decision rule is a fixed headroom law, not vibes: load is deferred if `predictedForLoad > freeMemory*80/100`; batch-fit checks use 80%/75%/60% headroom by batch size. A single tunable threshold makes the message checkable.
- Honest adaptive narration on OOM (the load itself proved the estimate wrong): `"llama-server load failed; reducing automatic context and retrying once"`, `"llama-server load failed; evicting all other models and retrying once"`, `"runtime OOM detected; expiring loaded models to clear memory before next request"`. Retry is capped at once (`oomRetryAttempted`), then the request fails fast. After eviction it waits for VRAM convergence (free memory recovering ~75% of the runner's VRAM, within 5s) before the next load.
- UX transfer: (1) every warning carries observed + expected numbers; (2) degradation is announced as an action with a reason ("evicting", "reducing X and retrying once"), never silent; (3) one capped retry, then honest failure.

**LM Studio** (https://lmstudio.ai/docs/cli, https://github.com/lmstudio-ai/lms, fetched 2026-11-08; hardware-page behavior corroborated by https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2176 and https://houtini.com/articles/how-to-set-up-lm-studio). Pattern: scan hardware once, expose it as a surface, and make every model-loading knob explicit — Settings > Hardware lists detected GPUs/VRAM with a per-model GPU-offload layer slider; CLI mirrors it (`lms status` reports daemon/server/hardware state; `lms load --gpu off|max` and context-length flags). Its load guardrail includes the context cache in the VRAM estimate, so the documented first fix is "lower context length" before touching offload (https://llmconfigurator.com/en/guides/troubleshooting/lm-studio-failed-to-load). Failure mode seen in the field: detection itself fails (1 of 2 GPUs found — issue #2176), which is why the hardware surface must print what it SAW, not just what it decided.

**Jan** (https://docs.jan.ai/desktop/troubleshooting, fetched 2026-11-08). Thin but instructive: the docs page for "Not enough RAM/VRAM for the model or KV cache" gives a short ordered fix list — update GPU drivers, try another GPU backend under Settings > Hardware, disable GPU offload. One paragraph, imperative verbs, no hedging. Context-length defaults drive VRAM blowups (issue https://github.com/janhq/jan/issues/3796).

**AnythingLLM** (https://github.com/Mintplex-Labs/anything-llm, fetched 2026-11-08 via search). Named boundary: no first-run hardware check at all — guidance is doc-level minimums (`docker/HOW_TO_USE_DOCKER.md`: ~2 GB RAM, 10 GB storage, Docker version floors) and OOM manifests as downstream crashes (#858, #2441). Counter-example that fixes (c): without a detection step, every failure is a crash report instead of a message.

**llama.cpp** (source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-alloc.c line 941 and https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-memory.cpp line 84, fetched 2026-11-08). The alloc-failure banner we inherit is honest but unactionable — number without fix:

- `GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size)` — backend + exact bytes.
- `throw std::runtime_error(std::string("failed to allocate buffer for ") + name)`.

Transfer: keep the exact-byte honesty, add the class name and the fix pointer. That is precisely the gap our banners close.

### 2.2 NVIDIA / driver-side precedents — measured-public + measured-ours

**nvidia-smi persistence mode** — the canonical "suboptimal state + one command" precedent. The query surface prints the state as a plain field (live on this rig, 2026-11-08): `Persistence Mode : Enabled`. Official fix text: `nvidia-smi -i <target gpu> -pm ENABLED` as root (https://docs.nvidia.com/deploy/driver-persistence/persistence-mode-legacy.html, fetched 2026-11-08); the modern recommendation upgrades to a boot-persistent unit: `sudo systemctl enable --now nvidia-persistenced` (https://askubuntu.com/questions/1400122/how-to-enable-nvidia-persistence-mode-on-boot-for-ubuntu-20-04-server). Pattern: state field readable everywhere; fix is one idempotent command; the better fix persists across reboot (our kernel-cmdline case is the same shape).

**NVIDIA installer preflight** — refuse-and-point. The runfile installer detects nouveau loaded and aborts with (verbatim, via https://forums.opensuse.org/t/opensuse-11.4-nvidia-install/63204, fetched 2026-11-08):

```
ERROR: The Nouveau kernel driver is currently in use by your system.
This driver is incompatible with the NVIDIA driver, and must be
disabled before proceeding. Please consult the NVIDIA driver README
for details on how to disable the Nouveau kernel driver.
```

Then the installer offers to write the blacklist itself (https://www.mxlinux.org forum reports; the installer flow offers the fix before dying). Pattern: state the conflict in one sentence, name the owning doc section, offer the write.

**libibverbs RLIMIT_MEMLOCK** — our memlock check's twin (https://www.open-mpi.org/faq/?category=openfabrics#ib-grey-listing, fetched 2026-11-08): `libibverbs: Warning: RLIMIT_MEMLOCK is 32768 bytes. This will severely limit memory registrations.` Fix set: `ulimit -l unlimited`, `/etc/security/limits.conf` (`* soft memlock unlimited` / `* hard memlock unlimited`), or systemd `LimitMEMLOCK`. Note the phrasing: observed number + named consequence ("will severely limit") — no fix command inline, fix lives in the FAQ. We can beat it by inlining the block (see 2.6).

**PyTorch driver gate** (https://raw.githubusercontent.com/pytorch/pytorch/main/torch/cuda/__init__.py, fetched 2026-11-08): availability inspection "never throws and returns 0 if the driver is missing or can't be" loaded (line 238 comment) — the engine DEGRADES to CPU with a warning instead of dying, and hard failures carry the probe that failed (`"cuda driver can't be loaded, is cuda enabled?"`, line 1530). Pattern: soft gate = warn + degrade + keep serving; hard gate = quote the probe path. Our arena: UNARMED degrades to 4K class and SAYS SO — same contract.

**dockerd kernel-capability warnings** (https://docs.docker.com/engine/containers/resource_constraints, https://github.com/moby/moby/issues/3886, fetched 2026-11-08): at startup dockerd prints `WARNING: No swap limit support` when the kernel booted without `cgroup_enable=memory swapaccount=1`. The fix is a kernel cmdline edit documented in the warning's doc link. This is the closest published precedent to "kernel cmdline gates our feature": warn once at startup, name the exact cmdline keys, link the how-to.

### 2.3 Doctor-style tools — what makes them beloved vs annoying — measured-public + measured-ours

**Homebrew `brew doctor`** (https://docs.brew.sh/Manpage + https://raw.githubusercontent.com/Homebrew/brew/master/Library/Homebrew/diagnostic.rb, fetched 2026-11-08):

- Contract: "Check your system for potential problems." Flags: `--list-checks` lists all audit methods, each runnable individually by name as arguments (`brew doctor check_for_stray_dylibs`); `-D/--audit-debug` debugs the audits themselves. "Will exit with a non-zero status if any potential problems are found."
- ~40+ named checks in source (`check_for_stray_dylibs`, `check_for_broken_symlinks`, `check_tmpdir_sticky_bit`, `check_access_directories`, `check_git_version`, ...). Messages are one-liners with the offending path or version: `"Your Python installation is unable to find `xattr`."`
- The anti-nag philosophy, verbatim from the manpage: diagnostic warnings "are just used to help the Homebrew maintainers with debugging if you file an issue" and — for anything cosmetic — "please don't worry or file an issue; just ignore this."
- Beloved because: every check is individually addressable, exit codes make it CI-usable, and severity matches consequence.

**`react-native doctor`** (https://reactnative.dev/blog/2019/11/18/react-native-doctor.html + https://github.com/react-native-community/cli/blob/main/docs/healthChecks.md + design history in https://github.com/react-native-community/cli/issues/694, fetched 2026-11-08). The richest interactive doctor:

- Checks carry version floors: Node >= 8.3, yarn >= 1.10, Watchman >= 4, Android SDK >= 26, NDK >= 19, Xcode >= 10, CocoaPods, ANDROID_HOME.
- On failure: menu to fix individually, fix all (`f`), skip; the fix-all prompt was renamed during design from "Fix all issues" to "Try to fix issues automatically" (issue #694) — honest about fix success rates.
- Each check implements `runAutomaticFix`; when automatic fix cannot work, the tool "will display a message and a helpful link explaining how to fix it manually" (blog, verbatim).
- Non-zero exit codes on failed checks make it CI-gateable.

**`flutter doctor`** — the format everyone copies: per-check `[✓]`/`[!]`/`[✗]` rows, `-v` for verbose detail, and fix strings that are imperative commands ("Run `flutter doctor --android-licenses` to accept the SDK licenses" — https://docs.flutter.dev). Source corroboration (https://raw.githubusercontent.com/flutter/flutter/master/packages/flutter_tools/lib/src/android/android_workflow.dart, fetched 2026-11-08): the license check prints task `'Validating Android SDK licenses'`, success `'All SDK package licenses accepted.'`, and detects stale state via regex `(\d+) of (\d+) SDK package licenses? not accepted.` — note the tool re-parses its own toolchain's output into counts.

**`systemd-analyze security`** (manpage https://www.freedesktop.org/software/systemd/man/latest/systemd-analyze.html fetched 2026-11-08; live sample on this rig, 2026-11-08):

- Scored scale: "calculates an overall exposure level for the whole unit, which is an estimation in the range 0.0…10.0 indicating how exposed a service is security-wise"; "High exposure levels indicate very little applied sandboxing."
- Per-setting table with weights and ✓/✗ marks, ending in a graded verdict line. Live: `→ Overall exposure level for systemd-journald.service: 4.9 OK :-)` (measured-ours). Manpage example: `→ Overall exposure level for systemd-logind.service: 4.1 OK 🙂`.
- Tuneable strictness: `--threshold=NUMBER` and `--security-policy=PATH` (JSON with per-test weights) — the score is a CONTRACT you can gate CI on, not a mood.
- Honest scope caveat, verbatim: "Note that this only analyzes the per-service security features systemd itself implements."

**`tuned-adm verify`** (https://docs.oracle.com/en/operating-systems/oracle-linux/9/tuned/tuned_command_reference.html; live on this rig 2026-11-08):

- Live success output, verbatim: `Verification succeeded, current system settings match the preset profile.` followed by a log pointer (`/var/log/tuned/tuned.log`).
- Failure output names the drift and carries ONE hint: "Verification failed, current system settings differ from the preset profile" + "You can mostly fix this by restarting the TuneD daemon, e.g.: `systemctl restart tuned`" (Oracle doc + https://bugzilla.redhat.com/show_bug.cgi?id=1956387). `-i/--ignore-missing` skips missing-plugin errors (https://docs.tuned.us.org / mankier manpage).
- Beloved because it VERIFIES APPLIED STATE, not declared state — exactly what our "armed" check must do (read back sysfs/meminfo, not the config file).

**docker `info` warnings + post-install paste-block** (https://docs.docker.com/engine/install/linux-postinstall/, fetched 2026-11-08): the docs page owns the root-needing fix as a copy-paste console block (`sudo groupadd docker`, `sudo usermod -aG docker $USER`) plus the re-login law ("Log out and log back in so that your group membership is re-evaluated") and `newgrp docker` as the no-relogin shortcut. Error-to-fix pairing: a quoted WARNING string maps 1:1 to a command block.

**lm-sensors `sensors-detect`** — the root-boundary masterclass (source: https://raw.githubusercontent.com/lm-sensors/lm-sensors/master/prog/detect/sensors-detect lines ~7436-7462, fetched 2026-11-08):

- Ends by printing a delimited paste-block: `To make the sensors modules behave correctly, add these lines to /etc/modprobe.conf:` then `#----cut here----` ... `#----cut here----`.
- Then OFFERS TO APPLY IT ITSELF when it can write: `Do you want to overwrite /etc/modprobe.d/lm_sensors.conf? (YES/no):` (line 7436-7446 shows it writing the file with a `# Generated by sensors-detect on <date>` header when permitted).
- Defaults are risk-scaled: safe probes default YES (`(YES/no):`), risky ones default NO (`(yes/NO):`, e.g. ISA I/O scanning at line 7655).

### 2.4 Crossing the "fix needs root" boundary — measured-public

Four proven moves, in ascending invasiveness:

1. **Print the exact block; user pastes.** lm-sensors cut-here block (2.3); MongoDB docs command chain (2.6); Docker post-install console block (2.3). Works everywhere, zero magic, survives SSH/CI. Cost: user does the work; re-login/reboot law must be stated.
2. **Ship the fix as an installable unit.** MongoDB does not ask users to hand-edit THP — it ships a systemd oneshot unit (`enable-transparent-huge-pages`) that runs the echo-chain before `mongod` starts, plus `systemctl enable` (https://www.mongodb.com/docs/manual/tutorial/transparent-huge-pages/, fetched 2026-11-08). For RHEL it hands over a tuned profile snippet (`/etc/tuned/virtual-guest-thp/tuned.conf` with `[vm] transparent_hugepages=always`, then `sudo tuned-adm profile virtual-guest-thp`). This is the deepest pattern for KERNEL-STATE fixes: one root action installs the persistence mechanism; everything after is unit management. NVIDIA's persistence daemon (`nvidia-persistenced`) is the same move in vendor form.
3. **Offer to apply from the tool itself, under the user's privilege.** lm-sensors writes `/etc/modprobe.d/lm_sensors.conf` when run as root (2.3); the NVIDIA installer writes its own blacklist then reboots. Legitimate when the tool is ALREADY running privileged (installers); never self-escalate at runtime.
4. **Delegate the escalation to the OS trust broker (polkit/pkexec).** (https://www.freedesktop.org/software/polkit/docs/latest/pkexec.1.html, fetched 2026-11-08): "pkexec allows an authorized user to execute PROGRAM as another user"; the authentication agent dialog "will display the full path to the program to be executed so the user is aware of what will happen" — transparency is built into the dialog contract. Failure semantics are precise: exit 126 "if the authorization could not be obtained because the user dismissed the authentication dialog", 127 on not-authorized/other error; the environment is stripped to a safe set (kills `LD_LIBRARY_PATH` injection) and `PKEXEC_UID` is exported. Custom `.policy` actions (`org.freedesktop.policykit.exec.path` annotation) let a tool expose ONE narrow root verb ("arm the memory pool") instead of blanket root. Caveat from the manpage: with retained/implicit authorization, unvalidated arguments "could be a security hole" — the privileged helper must validate its own inputs.

Anti-patterns named by the evidence: self-invoking `sudo` from a long-running process (no OS broker, dead-ends in TTY-less contexts); nagging every boot for a cosmetic issue (brew's explicit "just ignore this" tier exists precisely to prevent this); auto-fixing without saying what changed (react-native's "Try to fix issues automatically" wording exists to set expectations).

### 2.5 The transferable kernel-state warning set — measured-public (the money table)

Every precedent below warns about a kernel state of the same species as ours (hugepages/THP/sysctl/memlock/cmdline), and shows the phrasing that survived a decade of hostile users:

| Tool | Observed state in message | Fix artifact | Source |
|---|---|---|---|
| Redis | THP `enabled` with fork-COW consequence | `echo never > /sys/kernel/mm/transparent_hugepage/enabled` (inline, in the checklist) | https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/ |
| MongoDB (8.0+) | THP mode wrong for its allocator | systemd oneshot unit running `echo always > .../enabled` + `echo defer+madvise > .../defrag` + `echo 1 > /proc/sys/vm/overcommit_memory`; tuned profile on RHEL | https://www.mongodb.com/docs/manual/tutorial/transparent-huge-pages/ |
| dockerd | kernel booted w/o swap accounting | cmdline `cgroup_enable=memory swapaccount=1` + reboot | https://docs.docker.com/engine/containers/resource_constraints |
| libibverbs | `RLIMIT_MEMLOCK is 32768 bytes` + consequence | `ulimit -l unlimited` / limits.conf / LimitMEMLOCK | https://www.open-mpi.org/faq/?category=openfabrics#ib-grey-listing |
| DPDK | hugepages missing / not reserved at boot | cmdline `default_hugepagesz=1G hugepagesz=1G hugepages=4`; runtime `echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages`; tool `dpdk-hugepages.py`; fstab hugetlbfs mount for secondary processes | https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html |
| nvidia-smi | `Persistence Mode : Disabled` | `nvidia-smi -i <gpu> -pm ENABLED` / persistenced unit | https://docs.nvidia.com/deploy/driver-persistence/persistence-mode-legacy.html |

DPDK is our closest structural cousin: boot-time reservation "when physically contiguous memory is required" because runtime reservation can fail to find contiguous pages; 1 GiB pages preferred "if the platform supports them"; per-NUMA-node sysfs paths for placement; hugetlbfs mount noted as required only for secondary-process support. Note the DB split on THP direction (Redis wants `never`, MongoDB 8.0 wants `always`) — the kernel knob's meaning depends on the workload; state WHICH class you are and why in the message, or users will cross-apply internet advice.

### 2.6 Synthesis — the ten laws that separate beloved from annoying

1. Numbers over adjectives: observed value + expected value in every warning (Ollama fields, libibverbs bytes, systemd exposure 4.9).
2. One fix, exact text: a copy-pasteable command block beats a doc link (Redis, lm-sensors, Docker post-install); the doc link is a fallback, not the primary carrier.
3. Verdict line with a scale: a score + grade makes progress legible across runs (systemd 0.0-10.0, tuned's succeeded/failed string).
4. Applied-state verification, not config-file trust: read the kernel back (tuned-adm verify reads live settings; our doctor must read /proc + /sys, not the unit file).
5. Checks individually addressable: `--list-checks` + per-check invocation + non-zero exits (brew, react-native).
6. Honest auto-fix framing: "Try to fix issues automatically", announce what changed, never auto-fix silently (react-native #694).
7. Scope honesty on the report: state what the check does NOT cover (systemd's verbatim caveat).
8. Risk-scaled defaults: safe fixes default YES, risky ones default NO, with the default visible in the prompt (sensors-detect).
9. Root work goes through the OS broker or a paste-block, never self-sudo; failure exits are distinct and propagated (pkexec 126 vs 127).
10. Cosmetics carry an explicit ignore-me license (brew's "just ignore this"); nothing below "costs you performance" ever re-nags.

## 3. Recommended UX spec — our engine's first-run memory check

Design-inference below; every pattern cited to 2.x. Engine name placeholder: `strata`.

### 3.1 Banner contract (boot, always)

One line per class, always printed, even when OK — honesty includes good news (nvidia-smi prints `Persistence Mode : Enabled`; silence reads as "not checked"):

```
ARENA PAGES: 1 GiB CLASS OK (pool 32/32 pages, 32 GiB, memlock unlocked)
ARENA PAGES: 2 MiB/4K DEGRADED CLASS (no 1 GiB pages; run: strata-doctor)
ARENA PAGES: UNARMED (HugePages_Total=0; run: strata-doctor)
```

Rules: banner line NEVER repeats as prose; the one-paragraph explainer prints at most once per NEW bad verdict (see 3.5); max one explainer per boot; the banner carries the doctor pointer instead of re-explaining. This is the tuned/nvidia-smi contract (2.2, 2.3) plus Ollama's announce-degradation law (2.1).

### 3.2 Detection list

Every probe reads applied kernel state (law 4). Required page count = f(model manifest bytes, KV/scratch headroom, installed RAM) — same numeric preflight shape as Ollama's `predicted` vs `available` (2.1).

| ID | Check | Probe | Bad classes | Fix artifact |
|---|---|---|---|---|
| C1 | pool armed | `/proc/meminfo`: `HugePages_Total` vs required | UNARMED (0), DEGRADED (<required) | cmdline `default_hugepagesz=1G hugepagesz=1G hugepages=N` (DPDK form) or sysfs echo for runtime |
| C2 | page-size class | `Hugepagesize` + `/proc/cmdline` + cpuinfo `pdpe1gb` | DEGRADED (2 MiB/4K when 1 GiB supported) | same as C1 with 1G size; if `pdpe1gb` absent, class is a hard floor, not a warning — say so once |
| C3 | pool usable at start | `HugePages_Free` == `HugePages_Total` at engine start | DEGRADED ("reserved but exhausted/fragmented — reserve at boot") | move reservation to boot-time (DPDK 2.5) |
| C4 | THP fallback class | `/sys/kernel/mm/transparent_hugepage/enabled` active `[...]` mode | DEGRADED 2 MiB class (want `always` for our arena fallback; we are the MongoDB side of the Redis/MongoDB split — cite why in the message) | oneshot unit or tuned profile (2.5) |
| C5 | memlock floor | `getrlimit(RLIMIT_MEMLOCK)` vs pool bytes for mlock paths | DEGRADED (libibverbs phrasing: observed bytes + "will limit") | `limits.d/strata.conf` or unit `LimitMEMLOCK=infinity` |
| C6 | RAM sanity | `MemTotal`/`MemAvailable` vs pool + model bytes | DEGRADED ("pool of X + models Y > RAM Z — reduce pool") | resize command from --fix |

Sizing rule: `required_pages = ceil(model_bytes / hugepage_size) + headroom`; recomputed from the live model manifest, so installing a bigger model produces a NEW verdict (3.5), which is allowed to explain once.

### 3.2 Message templates (one paragraph + one block, no more)

UNARMED / DEGRADED — the paragraph answers what/how-bad/how-to-fix in 3 sentences, carries observed and expected numbers (laws 1-2):

```
ARENA PAGES: UNARMED (HugePages_Total=0; run: strata-doctor)
Large-page arena is UNARMED: /proc/meminfo reports HugePages_Total=0, and your
model set (32 GiB, manifest 9f3c) needs 32 pages of 1 GiB. Without the pool the
arena falls back to the 4K DEGRADED CLASS — expect slower tensor loads and
higher TLB pressure. Fix (root, survives reboot) — add to kernel cmdline and reboot:
#----cut here----
default_hugepagesz=1G hugepagesz=1G hugepages=32
#----cut here----
Or run: sudo strata-doctor --fix   (generates and installs strata-arm-memory.service)
This message will not repeat once the pool matches (strata-doctor remembers).
```

Undersized — name the delta and the cause (models grew):

```
ARENA PAGES: DEGRADED (pool 16/32 pages; model set grew to 32 GiB on 2026-11-07)
```

Memlock — libibverbs phrasing with the block inlined (2.5):

```
RLIMIT_MEMLOCK is 65536 KiB (soft) — the arena mlock of 32768 MiB will fail past
64 MiB. Raise it (root, persistent):
#----cut here----
# /etc/security/limits.d/strata.conf
*  soft  memlock  unlimited
*  hard  memlock  unlimited
#----cut here----
```

THP — state which side of the Redis/MongoDB split we are on (2.5), because users WILL paste `echo never` from Redis folklore:

```
transparent_hugepage/enabled is [never]; the 2 MiB arena fallback class wants
[always] (we are the MongoDB case, not the Redis case: our fallback relies on
kernel coalescing, we do not fork-and-COW the arena). Fix:
  echo always > /sys/kernel/mm/transparent_hugepage/enabled
Persist via: sudo strata-doctor --fix   (installs strata-arm-memory.service)
```

### 3.3 `strata-doctor` CLI contract

Modeled on brew (addressability, exits) + systemd-analyze (score) + react-native (auto-fix) + lm-sensors (apply-or-print):

- `strata-doctor` — run all checks; systemd-analyze-style table (✓/✗ per setting with observed vs expected values); verdict line: `→ MEMORY CLASS: 1 GiB OK (score 0.9 UNARMED-risk)` or `→ MEMORY CLASS: DEGRADED (score 6.2)`; exit per 3.4.
- `--list-checks` — print check IDs (C1..C6); `--check C1[,C3]` — run subset (brew law 5).
- `--json` — machine report: every check, probe path, observed, expected, verdict, fix command; ALWAYS reports current truth, never suppressed state (suppression governs prose, never data).
- `--verbose` — include raw probe dumps (cat of meminfo lines, cmdline, sysfs) for bug reports.
- `--fix` — attempt every fix that is legal WITHOUT root (none usually are — sysfs is root-owned), then PRINT all root blocks cut-here style + the unit file contents into `~/.strata/generated/`, with install commands. Never invokes sudo itself (2.4 anti-pattern). Wording follows react-native: "Attempting fixes automatically (non-root); root steps printed below."
- `--fix --pkexec` — the GUI path: ships `.policy` action `org.strata.arm-memory` annotated with `org.freedesktop.policykit.exec.path`; the helper validates its own inputs (pkexec manpage caveat); dialog shows the full helper path (polkit transparency, 2.4).
- Scope caveat printed in `--verbose` footer, systemd-style: "checks the memory-class state this engine implements; does not audit GPU driver state."

### 3.4 Exit codes

| Code | Meaning |
|---|---|
| 0 | all checks OK (verdict OK) |
| 1 | degraded (warnings; usable) |
| 2 | unarmed / cannot proceed |
| 3 | doctor internal error (probe crashed) |
| 126 | propagated: pkexec auth dialog dismissed — nothing changed |
| 127 | propagated: pkexec not authorized / helper failed |

CI usage: `strata-doctor --json && strata-doctor || exit $?` gates a pipeline the way react-native doctor does (2.3).

### 3.5 Nag-suppression ledger (requirement (d))

State file `/var/lib/strata/memory-ledger.json` (root-owned, written by the doctor/fix path) + `~/.cache/strata/boot-ledger.json` (per-user engine):

```
{ "type": "memory-ledger",
  "checks": { "C1": { "verdict": "OK", "hash": "a41f…" } },
  "model_manifest_hash": "9f3c…",
  "last_explained": { "C1": "2026-11-08T09:12:00Z", "verdict_hash": "e77b…" } }
```

Law: the explainer paragraph prints at boot only when the current verdict hash differs from `last_explained.verdict_hash`. A fix flips the ledger to OK and purges the entry — message never returns (requirement (d)). Installing a bigger model changes `model_manifest_hash` → C1 re-verdicts as DEGRADED with a NEW hash → one explanation is allowed. Banner lines (3.1) are exempt from suppression: they are one line, always honest, and are what the ledger keeps from ever becoming a paragraph again. This is brew's ignore-tier (2.3) made mechanical.

### 3.6 Boot vs standalone division

- Engine boot: banners (always) + at most one explainer (ledger-gated) + zero interactive prompts. Soft-gate degradation like PyTorch (2.2): UNARMED still serves on the 4K fallback and says so — never refuse-to-start for a performance class.
- `strata-doctor`: the verbose explainer, the table, the fix generator, the JSON API. The boot banner's ONLY call-to-action is this binary.
- Shipped persistence artifact (MongoDB pattern, 2.5): `strata-arm-memory.service` oneshot (sysfs echoes + THP + verified readback) + `/etc/security/limits.d/strata.conf` fragment + optional tuned profile include for RHEL-family hosts. `--fix` writes it; one root action installs persistence forever after.

### 3.7 Falsifiers for this spec

1. If `echo always` THP measurably hurts OUR fork/startup path (the Redis failure mode), C4's expected value flips — the message cites the workload reason precisely so this is checkable (2.5 DB-split).
2. If runtime sysfs reservation succeeds reliably on the host kernel, C3's boot-reservation advice is over-caught; measure reserve success on this rig before shipping the "reserve at boot" wording (DPDK's fragmentation rationale).
3. If `pdpe1gb` is absent on a target fleet device, C2 must NOT recommend 1 GiB pages — verify cpuinfo probe before message selection.

## 4. Sources

Primary (fetched 2026-11-08 unless noted):

- Ollama scheduler source: https://raw.githubusercontent.com/ollama/ollama/main/server/sched.go
- llama.cpp alloc source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-alloc.c ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-memory.cpp
- LM Studio CLI: https://lmstudio.ai/docs/cli ; https://github.com/lmstudio-ai/lms ; detection issue: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2176 ; guardrail note: https://llmconfigurator.com/en/guides/troubleshooting/lm-studio-failed-to-load
- Jan troubleshooting: https://docs.jan.ai/desktop/troubleshooting ; https://github.com/janhq/jan/issues/3796
- AnythingLLM minimums: https://github.com/Mintplex-Labs/anything-llm (docker/HOW_TO_USE_DOCKER.md); #858, #2441
- NVIDIA persistence: https://docs.nvidia.com/deploy/driver-persistence/persistence-mode-legacy.html ; https://askubuntu.com/questions/1400122/how-to-enable-nvidia-persistence-mode-on-boot-for-ubuntu-20-04-server
- NVIDIA installer nouveau error: https://forums.opensuse.org/t/opensuse-11.4-nvidia-install/63204
- libibverbs memlock: https://www.open-mpi.org/faq/?category=openfabrics#ib-grey-listing
- PyTorch CUDA gate: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/cuda/__init__.py
- dockerd swap-limit warning: https://docs.docker.com/engine/containers/resource_constraints ; https://github.com/moby/moby/issues/3886
- brew doctor: https://docs.brew.sh/Manpage ; https://raw.githubusercontent.com/Homebrew/brew/master/Library/Homebrew/diagnostic.rb
- react-native doctor: https://reactnative.dev/blog/2019/11/18/react-native-doctor.html ; https://github.com/react-native-community/cli/blob/main/docs/healthChecks.md ; https://github.com/react-native-community/cli/issues/694
- flutter doctor: https://docs.flutter.dev ; https://raw.githubusercontent.com/flutter/flutter/master/packages/flutter_tools/lib/src/android/android_workflow.dart
- systemd-analyze security: https://www.freedesktop.org/software/systemd/man/latest/systemd-analyze.html
- tuned-adm verify: https://docs.oracle.com/en/operating-systems/oracle-linux/9/tuned/tuned_command_reference.html ; https://bugzilla.redhat.com/show_bug.cgi?id=1956387
- docker post-install: https://docs.docker.com/engine/install/linux-postinstall/
- lm-sensors source: https://raw.githubusercontent.com/lm-sensors/lm-sensors/master/prog/detect/sensors-detect
- pkexec: https://www.freedesktop.org/software/polkit/docs/latest/pkexec.1.html
- Redis THP: https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/
- MongoDB THP: https://www.mongodb.com/docs/manual/tutorial/transparent-huge-pages/
- DPDK hugepages: https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html

Live probes (measured-ours, this rig, 2026-11-08): `systemd-analyze security systemd-journald.service` → `4.9 OK`; `tuned-adm verify` → `Verification succeeded, current system settings match the preset profile.`; `nvidia-smi -q` → `Persistence Mode : Enabled`.
