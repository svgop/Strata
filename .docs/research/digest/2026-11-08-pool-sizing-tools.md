---
type: research-digest
id: 2026-11-08-pool-sizing-tools
date: 2026-11-08
scope: how established systems size memory reservations and hugepage pools per device; stealable design for a per-device pool-sizing tool (detect → size from catalog → arm → verify → release/re-arm)
subject: hugeadm heuristics, Oracle ASMM/AMM advisor loop, TuneD plugin architecture, DPDK dpdk-hugepages.py, PostgreSQL huge_pages semantics and pgtune fractions, catalog-driven sizing precedents
method: web research against primary sources (man pages, project sources, vendor docs, kernel docs) + live box evidence (/proc/meminfo, /proc/cmdline, /etc/sysctl.d/91-strata-hugepages.conf, tuned, grubby, /proc/cpuinfo pdpe1gb)
---

# Pool-sizing tools — the digest

## Verdict (read this first)

**The tool is a `tuned`-style profile generator wrapped around a k8s-style catalog sizing rule, verified with DPDK's read-back-or-fail discipline.** Every mature system we surveyed separates the same four concerns: (1) a *declarative* input (catalog, boot param, or config value), (2) a deterministic sizing formula over that input, (3) one arming path with a single owner, (4) a read-back verification that treats "kernel granted less than asked" as a failure. **Our box currently fails step 4**: `/etc/sysctl.d/91-strata-hugepages.conf` declares `vm.nr_hugepages=54272` (106 GiB) but `/proc/meminfo` reports `HugePages_Total: 45076` (88.04 GiB = 94.51 GB) on 2026-11-08 — **9,196 pages (17.96 GiB) short**, and 94.51 GB < the DS arena's all-or-none line at 110.85 GB, so `huge_arena` defaults OFF and the DS CPU-lane copy is a live regression right now (the exact failure the 11-07 digest priced: "a smaller pool both disarms it AND shrinks the copy budget"). `/proc/cmdline` carries no hugepages args — the arm came through systemd-sysctl, which the kernel does not guarantee whole. The formula in §7 reproduces the settled 106 GiB target from first principles and adds the missing verify/release paths.

---

## Q1. libhugetlbfs `hugeadm` — what sizing heuristics exist, what they get right/wrong

**Surface.** `hugeadm --pool-list` prints Minimum / Current / Maximum per supported pagesize. Minimum = the guaranteed static pool (`vm.nr_hugepages`); pages between min and max allocate dynamically on demand with **no availability guarantee**; Maximum is the overcommit ceiling ([hugeadm(8)](https://linux.die.net/man/8/hugeadm)).

**Heuristics it ships.**
- `--pool-pages-min <size>:[+|-]<count|memsize>` — the core primitive; accepts memory units (e.g. `2MB:+2G`), so the *caller* owns the sizing arithmetic; hugeadm only converts and applies (hugeadm(8)).
- `--add-temp-swap[=N]` / `--add-ramdisk-swap` — grows the pool by staging a temporary swap file (default 5 huge pages) so reclaim can defragment; removed after the resize. This is the best-known remedy for "runtime grow fails under fragmentation" (hugeadm(8)).
- `--hard` — retry a failed min-pool resize up to 5 times while progress is made (hugeadm(8)).
- `--set-recommended-min_free_kbytes` — raises `vm.min_free_kbytes` to reduce pageblock mobility mixing so the pool can resize dynamically (hugeadm(8)).
- `--set-recommended-shmmax` / `--set-shm-group GID` — compute and persist `kernel.shmmax` and `vm.hugetlb_shm_group` ([huge_page_setup_helper.py](https://github.com/libhugetlbfs/libhugetlbfs/blob/master/huge_page_setup_helper.py)).
- `--obey-numa-mempol`, `--enable-zone-movable` — NUMA-mempolicy-obeying allocation and MOVABLE-zone allocation for growability (hugeadm(8)).
- `huge_page_setup_helper.py` — the shipped interactive tuner: reads total memory from `hugeadm --explain`, asks the operator for a memory amount (refuses to leave < 128 MB for the system), computes `pages = reqKB / (pagesize/1024)`, then writes `vm.nr_hugepages`, `vm.hugetlb_shm_group`, `kernel.shmmax` into `/etc/sysctl.conf` (backing up, stripping duplicates) and per-user `memlock` limits into `/etc/security/limits.d/hugepages.conf` (huge_page_setup_helper.py source).

**Gets right:** the min/max pool split mirrors the real kernel semantics; read-back expectations ("verify pool sizes after resizing"); temp-swap for runtime growth; the guard against starving the OS of base memory.

**Gets wrong / misses:** it ships **no formula** — everything from "how much" onward is the operator's problem (the helper asks for an MB amount). It defaults to sysctl-level persistence, which is the boot-fragile path (see Q5/kernel evidence below). It has no model/workload concept, no all-or-none awareness, and no per-consumer grouping. The [HOWTO](https://github.com/libhugetlbfs/libhugetlbfs/blob/master/HOWTO) adds the two durable rules we adopt verbatim: *boot-time is the safer route when memory is constrained* (pre-2.6.24 it was mandatory), and *always verify* with `hugeadm --pool-list` or `HugePages_Total` because "there is no guarantee that the system can allocate the hugepages requested".

## Q2. Oracle ASMM/AMM — the feedback loop, and what's stealable

**Architecture.** `SGA_TARGET > 0` enables ASMM: the database auto-distributes one declared total across buffer cache / shared pool / large pool / Java pool / streams pool; `SGA_MAX_SIZE` is the static ceiling (settable only at startup). Resizing happens in **granules** — 4 MB for SGA ≤ 1 GB, 16 MB above (512 MB at the very top end) — one granule size for all components, fixed for the instance lifetime ([Oracle 19c tgdba, Tuning the SGA](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgdba/tuning-system-global-area.html); [23ai edition](https://docs.oracle.com/en/database/oracle/oracle-database/23/tgdba/tuning-system-global-area.html)). The **MMAN** background process is "the instance memory manager... performs the resizing of memory components" ([Oracle Database Concepts, Background Processes](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/processes.html)); **MMON** feeds workload stats into AWR, which powers the advisors (requires `STATISTICS_LEVEL=TYPICAL+`).

**The advisory loop.** `V$SGA_TARGET_ADVICE` / `V$MEMORY_TARGET_ADVICE` rows are *simulated counterfactuals*: `(MEMORY_SIZE, MEMORY_SIZE_FACTOR, ESTD_DB_TIME, ESTD_DB_TIME_FACTOR)` — e.g. factor 0.5 → DB time ×1.344, factor 1.25 → ×0.9817. The kernel simulates performance at sizes it never ran, exposes the knee, and the operator (or an automation) moves the target to the knee ([managing-memory 19c](https://docs.oracle.com/en/database/oracle/oracle-database/19/admin/managing-memory.html); [worked example](https://orasg.wordpress.com/2015/10/31/tuning-sga_target_using-vsga_target_advice)). Resize *history* is first-class: `V$SGA_RESIZE_OPS` keeps the last **800 completed** GROW/SHRINK operations with timestamps and status; `V$SGA_DYNAMIC_FREE_MEMORY` shows what remains distributable (tgdba Table 12-1).

**Growth/shrink rules worth copying.** Manually-pinned components act as **minimum levels** ASMM never shrinks below (tgdba); with an SPFILE, "the database remembers the sizes of the automatically tuned SGA components across instance shutdowns" — learned state persists (managing-memory §6.4.2); growth prefers taking free granules, shrink returns them; AMM (`MEMORY_TARGET`) exchanges memory between SGA and PGA but is **incompatible with standard huge pages** (POSIX `/dev/shm` mappings, not SysV SHM_HUGETLB) — the documented reason Linux DBAs pin `SGA_MAX_SIZE`/ASMM when using hugepages (managing-memory; tgdba LOCK_SGA notes).

**Stealable for the inference pool:** (1) *counterfactual advisory rows* — our tool can log per-boot `(pool_size, witness lines)` (the engine already prints `huge_arena` ON/OFF and copy timings) into a resize-ops history to justify future targets; (2) *granule discipline* — resize only in whole-GiB steps, one unit for all consumers; (3) *minimum levels* — a consumer's catalog entry can pin a floor inside the pool; (4) *persist learned state* — the tool's chosen target should survive reboots in one owned file, not re-derive silently; (5) the AMM lesson in reverse: **pick one memory management mode when hugepages are involved**; ours is "all organ mass on 2 MiB pages, all-or-none consumers must fit whole".

## Q3. TuneD (RHEL) — per-device tuning structure and the plugin architecture

**Profile format.** INI `tuned.conf` per profile dir; `/usr/lib/tuned/<profile>/` holds vendor profiles, `/etc/tuned/<profile>/` overrides; `[main]` carries `include=<profile>` with later values winning. Every other section is a **plugin instance**: `type=` selects the plugin (upstream: audio, bootloader, cpu, disk, modules, mounts, net, scheduler, script, scsi_host, selinux, sysctl, sysfs, systemd, usb, video, vm); `devices=` scopes a plugin instance to specific devices; `replace=1` overrides a conflicting instance of the same type, default is merge ([tuned.conf(5)](https://man.archlinux.org/man/extra/tuned/tuned.conf.5.en); [RHEL 8: customizing TuneD profiles](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/customizing-tuned-profiles_monitoring-and-managing-system-status-and-performance)).

**Boot integration (the part we need).** The bootloader plugin owns kernel cmdline: `cmdline=` (replace), `+cmdline_x=` (append), `-cmdline_x=` (remove), per-profile keys merged in `_get_effective_options`. It detects rpm-ostree / bootc / BLS / plain GRUB2, patches grub config + `grub2-editenv` (`tuned_params`, `tuned_initrd`), records what it applied in `/etc/tuned/bootcmdline`, and **verifies against `/proc/cmdline`**, reporting per-key absence vs value mismatch. Options persist in grub across reboots; a reboot is required for kargs ([plugin_bootloader.py source](https://raw.githubusercontent.com/redhat-performance/tuned/master/tuned/plugins/plugin_bootloader.py); [RHEL bootloader plug-in](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/customizing-tuned-profiles_monitoring-and-managing-system-status-and-performance)).

**Detection/activation.** `tuned-adm recommend` picks a profile via `/usr/lib/tuned/recommend.d/*.conf` matching rules (overridable in `/etc/tuned/recommend.d`); active profile persists in `/etc/tuned/active_profile`; `tuned-adm verify` re-checks applied state ([tuned README](https://raw.githubusercontent.com/redhat-performance/tuned/master/README.md)).

**Custom plugins.** Plugins are Python modules loaded from the tuned plugins directory; third-party instances can also drive `type=script` sections (`script=${i:PROFILE_DIR}/profile.sh`) and `type=sysctl` maps for pure sysctl sets (tuned.conf(5)). This box runs `tuned-adm` with profile `throughput-performance` active (live check 2026-11-08).

**Stealable shape:** profile = named, versioned *intent*; plugin = mechanism; state file + verify = drift detection. Our tool should emit two profiles — `strata-hugepool` (arm 54,272) and `strata-lean` (release for non-LLM periods) — with a verify mode that reads `/proc/meminfo` and reports declared-vs-actual per key, exactly the bootloader plugin's per-key discipline.

## Q4. DPDK `usertools/dpdk-hugepages.py` — the best-known scripting for this problem

**Sizing/appply path ([source](https://github.com/DPDK/dpdk/blob/main/usertools/dpdk-hugepages.py); [docs](https://doc.dpdk.org/guides/tools/hugepages.html)):**
- Discovers valid page sizes from `/sys/kernel/mm/hugepages/hugepages-*kB/` directory names; default size parsed from `/proc/meminfo` `Hugepagesize`.
- Writes reservations **directly to sysfs** (`nr_hugepages` per size, or per node at `/sys/devices/system/node/nodeN/hugepages/`), then **reads back and raises `OSError("got X pages instead")` if the kernel granted fewer than requested** — the single most stealable line in the surveyed tooling.
- Size-string → pages: parse `(\d+)[KMG]` binary-scaled to kB, then `req = reserve_kb // pagesize_kb`, **rejecting non-multiples** with ValueError (no silent rounding).
- NUMA detection = existence of `/sys/devices/system/node`; `--node N` filters; requesting a node on non-NUMA fails loudly.
- `--setup SIZE` = clear → unmount → reserve → mount, in that order; `-c/--clear` zeroes every size on every selected node; mounting runs `mount -t hugetlbfs -o pagesize=...,uid=...,gid=...` on `/dev/hugepages`; permissions are **mount-level** (`uid=`/`gid=`), not chmod.
- Root required to modify; any user may `-s/--show`.

**`hugetlb_shm_group`:** legacy SysV-shm consumers need their group in `vm.hugetlb_shm_group`; modern DPDK flows avoid it (in-memory mode or hugetlbfs mount perms); the [kernel admin doc](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html) states shmget/SHM_HUGETLB users need the group and that no mount is needed for `mmap(MAP_HUGETLB)` — same class as our engine's `huge_alloc` (mmap-based, no group needed; our box runs `vm.hugetlb_shm_group=0`, live check).

**Sizing guidance ([Linux GSG, sys_reqs](https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html)):** prefer 1 GB pages when `pdpe1gb` (ours: present); example 1024×2 MB or 4×1 GB; **boot-time preferred when you need lots of physically contiguous memory**; "some kernels block runtime reservation of 1 GB pages, so boot-time may be the only option"; persistence via `/etc/fstab` (`nodev /mnt/huge hugetlbfs pagesize=1GB 0 0`) plus kernel cmdline (`default_hugepagesz=1G hugepagesz=1G hugepages=4`). The older interactive `dpdk-setup.sh` was deprecated in 2020 and removed in **DPDK 20.11** ([deprecation notice](http://mails.dpdk.org/archives/dev/2020-August/177629.html)) — one tool, one owner.

**Per-socket precedent:** OVS-DPDK sizes per NUMA socket from a formula, not a guess: shared mempool = 256K mbufs/socket × mbuf size (MTU-dependent), per-port mempool = f(rxqs, descriptors); `dpdk-socket-mem` reserves per-socket hugepages so each port's mempool is NUMA-local ([OVS DPDK memory models](https://docs.openvswitch.org/en/latest/topics/dpdk/memory); [How Much Hugepage Memory](https://developers.redhat.com/blog/2018/03/16/ovs-dpdk-hugepage-memory)) — the max-over-groups-of-sums pattern of §7 in the networking domain.

## Q5. PostgreSQL — `huge_pages` semantics and the fraction rules

**Semantics ([runtime-config-resource](https://www.postgresql.org/docs/current/runtime-config-resource.html)):** `huge_pages = try | on | off`, server-start only. `try` (default): request huge pages, **fall back to normal pages on failure**; `on`: failure **prevents startup**; `off`: never request. State is observable (`huge_pages_status`). Only affects the main shared memory area; THP is explicitly discouraged. The sizing oracle is **`postgres -C shared_memory_size_in_huge_pages`** — the server *itself* computes the exact page count it needs (e.g. 3170) before starting ([kernel-resources §18.4.5](https://www.postgresql.org/docs/current/kernel-resources.html)). Allocation guidance: "a larger setting would be appropriate if other programs on the machine also need huge pages"; allocate at runtime via `sysctl -w` or at boot via `hugepagesz=2M hugepages=3170` because "sometimes the kernel is not able to allocate the desired number... due to fragmentation, so it might be necessary to repeat the command or to reboot" — **immediately after a reboot most memory can be converted**. Permissions: `vm.hugetlb_shm_group` and/or `ulimit -l` memlock. `huge_pages=on` is the enforcement twin of our all-or-none consumers; `try` is the documented *partial-fallback* mode with its cost made visible via `huge_pages_status`.

**Fraction rules ([docs](https://www.postgresql.org/docs/current/runtime-config-resource.html)):** "a reasonable starting value for shared_buffers is 25% of the memory" on a dedicated server ≥ 1 GB RAM; **"it is unlikely that an allocation of more than 40% of RAM... will work better than a smaller amount"** — Postgres deliberately leaves the rest to the OS page cache; larger shared_buffers want larger `max_wal_size`. [pgtune](https://pgtune.leopard.in.ua/) mechanizes these fractions from inputs (DB version, RAM, connections, storage, workload class): `shared_buffers = RAM/4`, `effective_cache_size = RAM×3/4`, `maintenance_work_mem = RAM/16` (cap 2 GB), `work_mem = (RAM − shared_buffers)/(max_connections+1)` bounded 64 KB–2 GB ([le0pard/pgtune](https://github.com/le0pard/pgtune) formulas; [formula table](https://github.com/litviakk/tf-postgres-config/blob/main/SPEC.md)).

**Why it matters here:** PG's 25%-with-a-40%-ceiling is the *shape* of a safety rail for our general-RAM floor (§7), and `shared_memory_size_in_huge_pages` proves the best sizing oracle is the **consumer itself** — for us, the engine's own `all` witness line (110.85 GB) is exactly that oracle, already measured across five boots.

## Q6. Catalog-driven sizing precedents (size for max(X, Y, Z) + slack)

1. **Kubernetes huge pages — the purest declarative precedent.** The node pre-allocates at boot (`hugepagesz=1G hugepages=2 hugepagesz=2M hugepages=512` in GRUB); kubelet exposes `hugepages-2Mi`/`hugepages-1Gi` as Capacity/Allocatable; pods request whole allocations (`resources.limits: hugepages-2Mi: 100Mi`) which **must equal limits, do not overcommit, and schedule only where allocatable minus consumed covers the request** — a node-exact max-over-pods accounting ([Kubernetes: scheduling huge pages](https://kubernetes.io/docs/tasks/manage-hugepages/scheduling-hugepages/)). Dynamic post-boot allocations require a kubelet restart to be visible — reservation and accounting must re-sync.
2. **Oracle `SGA_MAX_SIZE` / `MEMORY_MAX_TARGET`** — provision the ceiling from the planned workload (largest SGA component sum), then let the target float under it ([tgdba](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgdba/tuning-system-global-area.html)): "you will run X" → ceiling sized once, distribution tuned live.
3. **PostgreSQL installers/pgtune** — declared inputs (RAM, workload class, connection count) → deterministic config; the huge-page count then derives from the *declared* `shared_buffers` via the server's own oracle (Q5). Sizing is catalog-driven at one remove.
4. **OVS-DPDK / kolla-ansible per-socket percentages** — deployments declare `dpdk-socket-mem` per NUMA socket (fixed MiB or a % of node memory, commonly ~30%/socket in OVS-DPDK deployments) sized from the formula in Q4 ([OVS memory models](https://docs.openvswitch.org/en/latest/topics/dpdk/memory); [Red Hat walkthrough](https://developers.redhat.com/blog/2018/03/16/ovs-dpdk-hugepage-memory)).
5. **Kernel boot params themselves** — `hugepages=0:1,1:2` per-node boot format is a declarative statement about planned consumption, and the [kernel doc](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html) calls boot-time allocation "the most reliable method since memory is not yet fragmented".

**No surveyed system ships a model-catalog-driven hugepage sizer.** The nearest neighbors are k8s (declarative *consumption* accounting, human-sized pools) and pgtune (deterministic *fractions*, no catalog). Our tool is genuinely new ground: catalog of packs → per-group sums → max → slack → whole-GiB pool → arm + verify. The 11-07 digest is the first entry in its own genre; this tool generalizes it.

---

## Recommended design — `strata-poolsize` (per-device sizer/verifier)

### Inputs (detected, never assumed)

| Input | Source | This box (2026-11-08) |
|---|---|---|
| RAM | `/proc/meminfo MemTotal` | 193,105,804 kB = 184.21 GiB |
| Page size + supported sizes | `/proc/meminfo Hugepagesize`, `ls /sys/kernel/mm/hugepages` | 2048 kB; 2 MiB + 1 GiB (`pdpe1gb` in /proc/cpuinfo) |
| NUMA nodes | `/sys/devices/system/node/node*` | 1 node |
| Consumer count from engine | engine witness line / manifest (DS `all` = 110.85 GB measured across 5 boots) | catalog below |
| Arming tools | `command -v grubby hugeadm sysctl` | grubby ✓, hugeadm ✗, sysctl ✓; tuned ✓ (`throughput-performance`) |
| Current pool + cmdline | `/proc/meminfo HugePages_*`, `/proc/cmdline` | **Total 45076; no hugepages kargs — drift, see Verdict** |

### Catalog schema (declarative, per device)

```yaml
# pool-catalog.yaml (one per device, in campaign/)
consumers:
  - id: dsv41-8081            # DeepSeek pack; ALL-OR-NONE (huge_arena rule, ds4_experts_rule.hpp:172)
    bytes: 110.85GB           # measured worst-case 'all' (11-07 digest §2)
    group: solo               # port law: alternates with qwen group, never concurrent
  - id: qwen-standing
    bytes: 80.4GiB            # store 38 + largest arena 39.4 + KV ~3 (concurrent sum)
    group: qwen-pack
  - id: ud-iq4-xs
    bytes: 94GiB              # arena 56 + store 38 (whole-fit requirement)
    group: qwen-pack
slack_gib: 2                   # cliff margin (11-07 §3.1)
floor_general: max(16GiB, 10%MemTotal)
```

### Sizing formula (and the worked example)

```
demand_pages = max over groups g of Σ ceil(bytes_i / 2,097,152)     # all-or-none consumers never split
pool_pages   = round_up_512(demand_pages + slack_gib × 512)          # whole-GiB steps, Oracle granule style
cap_pages    = floor(MemTotal − floor_general) / 2 MiB               # hard general-RAM rail
pool_pages   = min(pool_pages, cap_pages)                            # refuse + name the gap if cap binds
```

```
DS group:      ceil(110.85e9 / 2,097,152) = 52,860 pages
Qwen group:    80.4 GiB × 512 = 41,165 pages
UD-IQ4_XS:     94 GiB × 512   = 48,128 pages
demand         = max(52,860; 41,165; 48,128)          = 52,860
+ slack 2 GiB  = +1,024 → 53,884
round up 512:  53,884 → 54,272 pages = 106 GiB = 113.82 GB   ← reproduces the deployed target exactly
cap            = (184.21 − 18.42) GiB ≈ 84,860 pages   → not binding (54,272 ≤ cap, ~60 GiB margin)
general RAM    = 184.21 − 106 = 78.21 GiB remaining    → within the 11-07 coexistence analysis
```

### Outputs (commands + config)

1. **Primary arming — kernel cmdline (pristine-buddy law):** `grubby --update-kernel=ALL --args="default_hugepagesz=2M hugepagesz=2M hugepages=54272"`. Kernel docs: boot-time is "the most reliable method since memory is not yet fragmented" (kernel admin doc); PG docs: post-reboot conversion is the reliable path (§18.4.5).
2. **Secondary (survives bootloader regeneration) — owned sysctl file:** `/etc/sysctl.d/91-strata-hugepages.conf` with `vm.nr_hugepages=54272` + the header comment from `campaign/91-strata-hugepages.conf`. Idempotent when cmdline already armed (kernel clamps to satisfied pool).
3. **Verify (always, after arm or boot):** read `/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages` + `/proc/meminfo HugePages_Total`; declared ≠ actual ⇒ **fail with the DPDK message shape**: `got 45076 pages instead of 54272`; log `(declared, applied, method, timestamp)` — the missing record that hides today's drift.
4. **Partial-fallback remedy:** engines down → `echo 1 > /proc/sys/vm/compact_memory` → re-raise in GiB steps (the campaign file's documented steps; hugeadm's `--add-temp-swap` is the same trick with tooling, not installed here). Never partial-arm silently while an all-or-none consumer is configured.
5. **Release/re-arm (non-LLM periods):** release = write a lower `vm.nr_hugepages` (only unused pages return; shrinking below in-use converts the rest to *surplus* — kernel admin doc — so release only after engines stop); re-arm = re-run apply+verify; engine restart is a perfect re-arm because pages return whole (11-07 law, campaign header).
6. **No `hugetlb_shm_group` change** — consumers are mmap-based (`MAP_HUGETLB`-class), group 0 stays (kernel doc: no group needed for mmap; live box confirms 0).

### Safety rails

- **Never reserve below the general-RAM floor:** pool ≤ MemTotal − max(16 GiB, 10% MemTotal); refuse at plan time, print the gap (PG's "leave room for the OS" rule, made hard).
- **All-or-none awareness:** a consumer flagged all-or-none (DS `huge_arena`/`copy_pin` semantics) must be sized *whole*; a pool under its line is a **regression, not degradation** (45056 was rejected for exactly this, 11-07 §3.2) — the tool refuses to emit a pool below any all-or-none consumer's minimum and names which consumer breaks.
- **Read-back or fail:** every apply verifies; the kernel may silently grant fewer pages or skip NUMA nodes (kernel admin doc; hugeadm "no guarantee"; DPDK raises). Exit nonzero on shortfall.
- **Single arming owner:** cmdline primary, sysctl.d fallback, state file records which one armed — prevents the mixed-path drift found on this box.
- **Release only when quiescent:** shrink requires engines stopped (surplus-conversion hazard); verify release reached 0 before handing the box to non-LLM work.

## Boundary

`hugeadm` specifics were taken from the die.net man page (old but the only complete HTML copy) cross-checked against the shipped helper script; RHEL 9 TuneD docs returned 403 and the bootloader-plugin claims rest on the plugin source + RHEL 8 doc summaries; OVS memory-formula details cite the OVS docs topic page and the Red Hat walkthrough (full per-port formula not re-derived here). Live-box discrepancy (54272 declared vs 45076 resident) is observed, not yet root-caused — the tool's verify log is the instrument that closes it.

## Sources

- [hugeadm(8) man page](https://linux.die.net/man/8/hugeadm) · [huge_page_setup_helper.py](https://github.com/libhugetlbfs/libhugetlbfs/blob/master/huge_page_setup_helper.py) · [libhugetlbfs HOWTO](https://github.com/libhugetlbfs/libhugetlbfs/blob/master/HOWTO)
- [Oracle 19c tgdba: Tuning the SGA](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgdba/tuning-system-global-area.html) · [19c Admin's Guide: Managing Memory](https://docs.oracle.com/en/database/oracle/oracle-database/19/admin/managing-memory.html) · [23ai tgdba](https://docs.oracle.com/en/database/oracle/oracle-database/23/tgdba/tuning-system-global-area.html) · [SGA_TARGET advice example](https://orasg.wordpress.com/2015/10/31/tuning-sga_target_using-vsga_target_advice)
- [tuned.conf(5)](https://man.archlinux.org/man/extra/tuned/tuned.conf.5.en) · [plugin_bootloader.py](https://raw.githubusercontent.com/redhat-performance/tuned/master/tuned/plugins/plugin_bootloader.py) · [tuned README](https://raw.githubusercontent.com/redhat-performance/tuned/master/README.md) · [RHEL 8: customizing TuneD profiles](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/customizing-tuned-profiles_monitoring-and-managing-system-status-and-performance)
- [dpdk-hugepages tool docs](https://doc.dpdk.org/guides/tools/hugepages.html) · [dpdk-hugepages.py source](https://github.com/DPDK/dpdk/blob/main/usertools/dpdk-hugepages.py) · [DPDK Linux GSG: system requirements](https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html) · [dpdk-setup.sh deprecation/removal](http://mails.dpdk.org/archives/dev/2020-August/177629.html) · [OVS-DPDK memory models](https://docs.openvswitch.org/en/latest/topics/dpdk/memory) · [OVS-DPDK hugepage walkthrough](https://developers.redhat.com/blog/2018/03/16/ovs-dpdk-hugepage-memory)
- [PostgreSQL: runtime-config-resource](https://www.postgresql.org/docs/current/runtime-config-resource.html) · [PostgreSQL: kernel-resources §18.4.5](https://www.postgresql.org/docs/current/kernel-resources.html) · [pgtune](https://pgtune.leopard.in.ua/) · [le0pard/pgtune formulas](https://github.com/le0pard/pgtune) · [formula table](https://github.com/litviakk/tf-postgres-config/blob/main/SPEC.md)
- [Kubernetes: scheduling huge pages](https://kubernetes.io/docs/tasks/manage-hugepages/scheduling-hugepages/) · [Kernel admin guide: HugeTLB Pages](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)
- Local: `/home/svg/LLM/strata-custom-2/.docs/research/digest/2026-11-07-preboot-ds-pool-sizing.md` · `/home/svg/LLM/strata-custom-2/campaign/91-strata-hugepages.conf` · `/etc/sysctl.d/91-strata-hugepages.conf` · live `/proc/meminfo`, `/proc/cmdline` (2026-11-08)
