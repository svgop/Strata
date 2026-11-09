---
type: research-digest
id: 2026-11-09-bootstorm-fixes
date: 2026-11-09
scope: the engine boot-time memory storm (kswapd 87.7 GiB/min, 16k allocstalls, 91% compaction failure, zram 94%, store-to-tmpfs fallback) — four patch-ready fixes, priced, each with its law-17 test
subject: src/core/expert_source.cpp paced complement populate; src/core/shadow_experts.cpp memfd store populate + reserve-first; include/strata/platform/hugepage.hpp Rsvd-correct pool accounting; serve/server.py pre-boot sweep + pool settle + drop-cache wiring; campaign/battery-lib.sh battery default
method: live kernel probe on this box (memfd MFD_HUGETLB reservation accounting + MADV_POPULATE_WRITE timing, 4 GiB over the real reserved pool), source reads of all five files, scratch-copy diffs verified with g++ -fsyntax-only (CUDA 13.4 headers), python3 -m py_compile, functional smoke of the new helpers against live /proc/meminfo, git apply --check on the combined patch. No src/ file was edited; diffs delivered from /tmp/bootstorm-scratch.
---

# Bootstorm fixes — four patch-ready diffs, priced, with law-17 tests

## 0. Verdict table (read this first)

| fix | file(s) | mechanism | price (boot / storm) | law-17 witness line |
|---|---|---|---|---|
| F1 | expert_source.cpp | one `MADV_POPULATE_WRITE` over the pool mapping replaces the 4 KiB-stride walk; the per-GiB gate stays off **on the pool arm only** | **−6-12 s** boot; ~0 GiB storm directly, the load-vs-reclaim overlap window shrinks by the same 6-12 s | `"faulted in one MADV_POPULATE_WRITE pass"` |
| F2 | shadow_experts.cpp | store faults first in one kernel pass, then copies fault-free; chunked memcpy stays as the EINVAL fallback | **−0-2 s** clean boot, **−3-10 s** squeezed boot; populate variance 4.5-16.8 s → ~2.5-3 s floor | `"memfd, reserved and populated in X s"` with X ≤ 4.0 |
| F3 | server.py + battery-lib.sh | files sweep runs BEFORE the heavy load (fresh-process + reload hooks); `STRATA_DROP_CACHE_ON_EXIT` default-on for batteries + the dead exit-path wiring fixed | **storm −70-100 %**: boot-minute pgsteal 87.7 → ≤ ~20 GiB target, allocstall 16,236 → < 1,000, compact_fail 1,732/min → under the ≤1,000/cycle gate; **−10-25 s** boot-to-READY in cycles | `"pre-boot sweep: N files, X GiB"` before any engine line |
| F4 | shadow_experts.cpp + hugepage.hpp + server.py | store's memfd mmap is the atomic pool reservation (attempt-first); pool readers subtract `HugePages_Rsvd`; launcher waits (bounded) for the dead engine's pool pages | M5 eliminated: tmpfs-tier boots 1-in-N → 0; per squeezed boot **pswpout −14,386 pages, zram 94 % → < 50 %, engine Swap 840 MiB → 0** | `"hugetlb pool settled at X GiB drawable"`; zero boots log `"the pool could not reserve the store"` |

Combined patch: `.docs/research/digest/2026-11-09-bootstorm-fixes.patch` (382 lines, 14 hunks). Verified: `git apply --check` passes on this repo today; both .cpp files compile clean (`g++ -fsyntax-only -std=c++20 -I include -I /usr/local/cuda/include`); server.py byte-compiles; the new helpers smoke-tested against live `/proc/meminfo`.

**Verdicts on the two open questions (task 3 and 4):**

- **Can the Prewarm files sweep run BEFORE the engine's heavy load? Yes, and it should** — the old docstring's objection ("anything warmed before the pin is discarded") is pool-era-void: on the pool route (`huge_pool_covers` passes) `drop_file_cache` never fires (`expert_source.cpp:3529` — the drop lives in the `pin_depends_on_reclaim` branch, before the pool route is even considered). Two hooks cover every boot: `main()` before the `StrataEngine` constructor (fresh process = every battery leg) and `Service.ensure_loaded` before `engine.restart()` (reload boots). The post-READY sweep stays (it is then a cache-warm re-read, seconds).
- **Should `STRATA_DROP_CACHE_ON_EXIT` be default-on for batteries? Yes — but it is also currently dead code on the normal path.** Its only call site (`server.py:8902`) sits inside the KeyboardInterrupt kill branch of the closers loop; a graceful close (every battery `cleanup_engine` TERM) never drops. The patch wires the drop on both the reload path and the normal-exit path, and `battery-lib.sh` defaults it on. Desktop default stays off (a same-model restart wants the cache warm — the feature's own docstring tradeoff).
- **Does ftruncate reserve memfd MFD_HUGETLB pages? No — probed.** The MAP_SHARED **mmap** is the atomic reservation. Live evidence, §1. The engine already reserves correctly by construction; what was missing is the honest check (`Free − Rsvd`), the attempt-first ordering, and the launcher's settle wait for cross-engine overlap.

## 1. Live probe: the reservation primitive and populate rates (kernel 7.2.12-300.fc45, pool armed)

Probe: `/tmp/bootstorm-probe/probe.c`, 4 GiB memfd `MFD_HUGETLB|MFD_HUGE_2MB` over the real pool.

```
before                             Total=45076 Free=9491 Rsvd=0 (Free-Rsvd=9491)
after ftruncate 4 GiB (no mmap)    Total=45076 Free=9491 Rsvd=0 (Free-Rsvd=9491)   <- ftruncate does NOT reserve
after mmap (still unfaulted)       Total=45076 Free=9491 Rsvd=2048 (Free-Rsvd=7443) <- the MAP_SHARED mmap reserves
madvise(MADV_POPULATE_WRITE) rc=0  4 GiB in 0.163 s = 24.5 GiB/s                    <- faults move Rsvd->allocated
after MADV_POPULATE_WRITE          Free=7443 Rsvd=0
memset 4 GiB after populate:       0.150 s = 26.7 GiB/s   (fault-free copy)
memset 4 GiB WITHOUT populate:     0.244 s = 16.4 GiB/s   (fused fault-in-copy, the old shape)
after munmap+close(mfd)            Free=9491 Rsvd=0 (full return, instant)
```

Three facts this nails:

1. **ftruncate is size-only; the mmap reserves.** HugePages_Rsvd moved by exactly the store's page count at mmap, all-or-nothing (ENOMEM when the pool cannot cover), released at munmap+close. A memfd-arm failure is therefore atomic and cheap, and the check the old code needed was never `huge_pool_covers` — it was the mmap itself.
2. **`HugePages_Free` includes `HugePages_Rsvd`.** The store's standing reservation is invisible to a Free-only check: `huge_pool_covers()` and `huge_pool_free_bytes()` overstated the drawable pool the moment anything reserved first. F4 corrects both readers.
3. **MADV_POPULATE_WRITE works here at 24.5 GiB/s** (kernel ≥ 5.14 required; this box 7.2.12). Fused fault-in-copy is 16.4 GiB/s idle — so F2's decoupled populate is NOT an idle-box time win on the copy itself; its value is storm-window fault batching and the reservation-then-copy guarantee (§3 price honesty).

## 2. F1 — the complement: one MADV_POPULATE_WRITE over the pool mapping

`src/core/expert_source.cpp` (paced complement path, the `while (got < bytes)` walk). The stride walk wrote 1 byte per 4 KiB — ~9.9M user cache-line stores faulting the same ~19k pool pages one user-store at a time, plus a `/proc/meminfo` availability read per 1 GiB step. On the pool arm those gates guard a failure class the pool cannot produce (pool pages are reserved, unswappable, unmovable at boot). One in-kernel pass faults the whole mapping; the expert-boundary cut and the single `cudaHostRegister` are untouched. The THP fallback arm keeps gate + walk (the qwen-next digest's split): the condition binds `huge_map.hugetlb`, not `huge_ok`.

```diff
--- a/src/core/expert_source.cpp
+++ b/src/core/expert_source.cpp
@@ -59,6 +59,11 @@
 #include <cerrno>
 #include <linux/aio_abi.h>   // the kernel's own asynchronous reads (io_submit): no library, allowed by Docker's seccomp
 #include <sys/syscall.h>
+// 2026-11-09 bootstorm (F1): in-kernel prefault of a whole mapping (kernel 5.14+; this box 7.2.12 probed,
+// 24.5 GiB/s over reserved-pool 2 MiB pages).  Older glibc does not spell the constant - keep the number.
+#if !defined(MADV_POPULATE_WRITE)
+#define MADV_POPULATE_WRITE 20
+#endif
 #endif
 
 // a 64-bit seek (as in pinned.cu): the 32-bit `fseek` wraps past 4 GiB, and the spelling differs per platform
@@ -3644,6 +3649,23 @@
                     constexpr uint64_t kStep = 1ull << 30, kPage = 4096;
                     uint64_t got = 0;
                     const char* stop = nullptr;
+                    // 2026-11-09 bootstorm (F1): on the POOL-backed mapping the pages are faulted ONCE,
+                    // in-kernel, at the kernel's own huge-page granularity - MADV_POPULATE_WRITE replaces
+                    // the 4 KiB-stride walk below, which wrote ~9.9M cache lines to fault the same ~19k
+                    // pool pages one user-store at a time (probed on this box: 4 GiB of pool pages in
+                    // 0.163 s, 24.5 GiB/s).  The per-GiB availability gate stays off on THIS arm only
+                    // (huge_map.hugetlb, not the THP fallback): the pages come from the reserved pool -
+                    // never reclaimed, never compacted, the failure class #1250's pacing guards does not
+                    // exist there.  EINVAL (kernel < 5.14) and the THP arm take the paced stride walk.
+                    const bool populate_pass = [&] {
+                        if (huge_ok && huge_map.hugetlb && madvise(arena, (size_t) bytes, MADV_POPULATE_WRITE) == 0) {
+                            std::fprintf(stderr, "FileExpertSource: the complement's 2 MiB-backed pages were faulted in one "
+                                                 "MADV_POPULATE_WRITE pass\n");
+                            return true;
+                        }
+                        return false;
+                    }();
+                    if (populate_pass) got = bytes;
                     while (got < bytes) {
                         const uint64_t step = std::min<uint64_t>(kStep, bytes - got);
                         uint64_t avail_now = 0;
@@ -3686,8 +3708,9 @@
                                       "resident but pageable, so its copies to the GPU go through the CPU (#1250)",
                                       (double) partial_pin / 1073741824.0, (double) bytes / 1073741824.0, stop);
                     else
-                        std::snprintf(msg, sizeof msg, "%.2f GiB page-locked after taking the pages in steps",
-                                      (double) partial_pin / 1073741824.0);
+                        std::snprintf(msg, sizeof msg, "%.2f GiB page-locked (the pages were taken in %s)",
+                                      (double) partial_pin / 1073741824.0,
+                                      populate_pass ? "one MADV_POPULATE_WRITE pass" : "paced steps");
                     note += std::string(note.empty() ? "" : "; ") + msg;
                     if (stop != nullptr) std::fprintf(stderr, "FileExpertSource: %s\n", msg);
                 } else if (budget_bytes > 0 && partial_on) {
```

**Price.** Populate at the probed 24.5 GiB/s: 37.86 GiB ≈ **1.5 s** in-kernel. The qwen-next digest priced walk+gate at 6-12 s buyable (the fault cost itself, ~3-6 s of huge-page zeroing, is irreducible either way). **Boot −6-12 s** (upper end under storm conditions, where every serialized fault context-switch competes with kswapd). Storm: ~0 GiB directly — pool faults draw from the reservation, never reclaim; the indirect effect is the 6-12 s smaller window in which the load overlaps the previous engine's cache reclaim.

**Law-17 test.** Same-window boot pair (same pack, interleave order A/B/A/B, pool asserted `HugePages_Total == 45076` before each boot, engine-down between boots): witness lines `faulted in one MADV_POPULATE_WRITE pass` and `X.XX GiB page-locked (the pages were taken in one MADV_POPULATE_WRITE pass)`; wall-clock the `engine started` → complement-complete window. Acceptance: ≥ 6 s saved, smaps `Private_Hugetlb` for the complement identical (37.86 GiB, `ht` flag), lanes unchanged (law 13). `STRATA_PIN_GUARD=2` forces the paced path for a forced-A boot.

## 3. F2 + F4 (store) — populate treatment and attempt-first reservation

`src/core/shadow_experts.cpp`. Two changes in the memfd arm: (F2) the store faults whole in one kernel pass, then copies fault-free — the chunked memcpy loop stays as the EINVAL fallback (kernel < 5.14); (F4) the `huge_pool_covers` pre-check is dropped, the memfd arm attempts first, and the mapping's own reservation is the pool check — with a shortfall note that names the numbers instead of the old silent `"memfd populate refused"`.

```diff
--- a/src/core/shadow_experts.cpp
+++ b/src/core/shadow_experts.cpp
@@ -28,6 +28,11 @@
 #if !defined(MFD_HUGE_2MB)
 #define MFD_HUGE_2MB (21 << 26)
 #endif
+// 2026-11-09 bootstorm (F2): in-kernel prefault of the whole store mapping (kernel 5.14+; this box 7.2.12
+// probed, 24.5 GiB/s over reserved-pool 2 MiB pages).  Older glibc does not spell the constant.
+#if !defined(MADV_POPULATE_WRITE)
+#define MADV_POPULATE_WRITE 20
+#endif
 #include "strata/platform/hugepage.hpp"
 #endif
 
@@ -155,7 +160,15 @@
     const char* mfd_note = "";
 #if !defined(_WIN32) && defined(MFD_HUGETLB)
     const char* mfd_off = std::getenv("STRATA_SHADOW_MEMFD");
-    if ((mfd_off == nullptr || mfd_off[0] != '0') && strata::platform::huge_pool_covers((size_t) want)) {
+    // F4 (2026-11-09 bootstorm): ATTEMPT the memfd arm first and let the mapping's own reservation be
+    // the pool check.  Probed on this box (kernel 7.2.12): ftruncate on a memfd MFD_HUGETLB file does
+    // NOT reserve pages (Rsvd stayed 0); the MAP_SHARED mmap RESERVES the whole store atomically
+    // (HugePages_Rsvd moved by exactly the store's page count, all-or-nothing, -ENOMEM when the pool
+    // cannot cover).  The earlier huge_pool_covers() pre-check raced the pool's other drawers and its
+    // all-or-nothing refusal is what landed the store on the swappable 4 KiB tmpfs tier in the 11:05
+    // squeeze (log:667335) - the store now reserves its pages the moment they exist, ahead of the
+    // complement's draw and every later organ.
+    if (mfd_off == nullptr || mfd_off[0] != '0') {
         const auto t0 = std::chrono::steady_clock::now();
         int mfd = ::memfd_create("strata-shadow-q2_0", MFD_HUGETLB | MFD_HUGE_2MB);
         if (mfd >= 0) {
@@ -169,22 +182,36 @@
                 if (src != MAP_FAILED && dst != MAP_FAILED) {
                     uint8_t* d = (uint8_t*) dst;
                     const uint8_t* s = (const uint8_t*) src;
-                    constexpr size_t kChunk = 256ull << 20;
-                    for (size_t o = 0; o < (size_t) want; o += kChunk) {
-                        const size_t n = std::min(kChunk, (size_t) want - o);
-                        std::memcpy(d + o, s + o, n);
+                    // F2 (2026-11-09 bootstorm): fault the whole store in ONE kernel pass first
+                    // (MADV_POPULATE_WRITE, kernel 5.14+; probed on this box: 4 GiB of pool pages in
+                    // 0.163 s, 24.5 GiB/s), then copy fault-free; on EINVAL (kernel < 5.14) the chunked
+                    // loop below stays the populate and faults in the copy as before.
+                    if (::madvise(dst, rounded, MADV_POPULATE_WRITE) == 0) {
+                        std::memcpy(d, s, (size_t) want);
+                    } else {
+                        constexpr size_t kChunk = 256ull << 20;
+                        for (size_t o = 0; o < (size_t) want; o += kChunk) {
+                            const size_t n = std::min(kChunk, (size_t) want - o);
+                            std::memcpy(d + o, s + o, n);
+                        }
                     }
                     ::munmap(src, (size_t) want);
                     ::mprotect(dst, rounded, PROT_READ);   // the store is read-only from here on
                     const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                     char buf[128];
-                    std::snprintf(buf, sizeof buf, "; 2 MiB pool pages (memfd, populated in %.1f s)", secs);
+                    std::snprintf(buf, sizeof buf, "; 2 MiB pool pages (memfd, reserved and populated in %.1f s)", secs);
                     mfd_note = buf;
                     view = dst;
                 } else {
                     if (src != MAP_FAILED) ::munmap(src, (size_t) want);
                     if (dst != MAP_FAILED) ::munmap(dst, rounded);
-                    mfd_note = "; memfd populate refused - file mapping";
+                    char buf[192];
+                    std::snprintf(buf, sizeof buf,
+                                  "; the pool could not reserve the store (%.1f GiB free of %.1f GiB needed) - "
+                                  "the swappable tmpfs tier serves every miss",
+                                  (double) strata::platform::huge_pool_free_bytes() / 1073741824.0,
+                                  (double) rounded / 1073741824.0);
+                    mfd_note = buf;
                 }
             } else {
                 mfd_note = "; memfd ftruncate refused - file mapping";
```

**Price (honest, from the probe).** Probed floors per 31.64 GiB: fused fault-in-copy 16.4 GiB/s ≈ 1.9 s idle; populate+copy ≈ 1.3 + 1.2 ≈ 2.5 s idle — **F2 costs ~+0.5 s on an idle box** (one extra 31.64 GiB zero-write pass) and buys: the store's ~15,820 dst faults leave the interleaved copy loop and ride one kernel pass through the storm window; the whole store is proven resident before the first src byte is read. The field spread (4.5-16.8 s measured) is src-side — cache misses and storm interference — which F3's pre-boot sweep addresses, not F2. **Net price: −0-2 s clean boots, −3-10 s squeezed boots** (populate variance tightens to the ~2.5-3 s floor). F4's part of the price sits in §5.

**Law-17 test.** Boot pair, same pack, pool asserted: witness line `"memfd, reserved and populated in X s"`. Acceptance: X ≤ 4.0 s on a clean boot and the clean-vs-squeezed X spread collapses (max − min ≤ 2 s across 3 boots); smaps `Private_Hugetlb` store = 31.64 GiB unchanged; store-miss µs lane unchanged (law 13). `STRATA_SHADOW_MEMFD=0` A/B leg isolates the store populate if the pair disagrees.

## 4. F3 — the boot-order storm: sweep before load, drop-cache default for batteries

`serve/server.py` + `campaign/battery-lib.sh`. Four hunks. (1) `Prewarm.warm_before_load()` — the existing `_file_targets` + `_warm_files` machinery (fadvise WILLNEED, then 4 readers), files only, capped at 600 s, run while the old process is reaped and the new one not spawned. (2) The `ensure_loaded` pre-boot sequence: drop the dead pack cache (opt-in honored) → bounded hugetlb pool settle (F4) → pre-boot sweep → `engine.restart()`. (3) The fresh-process hook in `main()` before the engine constructor — a battery leg is a fresh server, so its FIRST boot otherwise has no Prewarm object when the constructor blocks on the heavy load. (4) The exit-path drop wiring: the only pre-existing call site fired solely on the KeyboardInterrupt kill branch; the graceful close (every battery stop) never dropped. Plus `battery-lib.sh`: `STRATA_DROP_CACHE_ON_EXIT` defaults on for battery legs (explicit value always wins; desktops unchanged).

```diff
--- a/serve/server.py
+++ b/serve/server.py
@@ -3701,6 +3701,7 @@
 
 # ------------------------------------------------------------------------------------------------ prewarm
 PREWARM_ASK_S = 600.0        # one warm-in request's patience: a cold mid prefill is slow, and warming it is the point
+PREWARM_PREBOOT_MAX_S = 600.0   # the pre-boot files sweep's own cap (F3): it holds the load, never starve it
 # The warm-in texts: deliberately unlike any probe or battery prompt - a warm-in that shares a prefix with a probe
 # serves the probe from the conversation cache, and the probe stops measuring (the campaign's law).
 PREWARM_SHORT = ("Sketch the lifecycle of a container image from build to retirement, naming the layers, registries "
@@ -3770,6 +3771,27 @@
         threading.Thread(target=self._run, daemon=True, name="strata-prewarm").start()
         return self
 
+    def warm_before_load(self) -> bool:
+        """The files sweep BEFORE the engine's heavy load (the 2026-11-09 bootstorm fix, F3).  The boot
+        minute otherwise overlaps two storms: kswapd reclaiming the previous engine's page cache (measured
+        +87.7 GiB pgsteal, +16k allocstalls in one load minute) and the new engine's own faulting.  Run
+        here - the old process is reaped, the new one not spawned - so the dead cache is recycled by our
+        own reads and the heavy load faults over a warm cache.  Files only (the replay needs the engine
+        up, and _cycle still runs it there); -> True when every byte was read."""
+        if not self.files_on:
+            return False
+        targets = self._file_targets()
+        if not targets:
+            return False
+        gib = sum(sz for _, sz in targets) / 2 ** 30
+        print(f"[strata] prewarm: pre-boot sweep: {len(targets)} files, {gib:.1f} GiB into the page cache "
+              "(before the engine's load, so the reclaim storm plays out here)", flush=True)
+        t0 = time.monotonic()
+        full = self._warm_files(targets, t0 + min(self.timeout_s, PREWARM_PREBOOT_MAX_S))
+        print(f"[strata] prewarm: pre-boot sweep {'warm' if full else 'partial'} in "
+              f"{time.monotonic() - t0:.0f} s", flush=True)
+        return full
+
     def snapshot(self) -> dict:
         with self.lock:
             return dict(self.state)
@@ -3996,6 +4018,82 @@
         return total > self._base + self._own
 
 
+def prewarm_file_sweep_preboot(cfg: dict, env: dict, arch: str) -> None:
+    """The fresh-process boot's pre-boot files sweep (F3, 2026-11-09 bootstorm): a battery leg launches a
+    fresh server, so its FIRST boot has no Prewarm object yet when the engine constructor blocks on the
+    heavy load - this runs the same sweep before that constructor.  Reload boots go through
+    Service.ensure_loaded -> Prewarm.warm_before_load instead.  Fails open, always."""
+    try:
+        pw = Prewarm(cfg, env or {}, arch or "", "", 0)
+        if pw.files_on:
+            pw.warm_before_load()
+    except Exception as e:
+        print(f"[strata] prewarm: the pre-boot sweep did not finish ({type(e).__name__}: {e}); "
+              "the boot continues", flush=True)
+
+
+HUGEPAGE_POOL_WAIT_S = 20.0      # bounded wait for the dead engine's hugetlb pages to return (F4)
+
+
+def huge_pool_free_gib() -> float:
+    """Drawable pool (HugePages_Free - HugePages_Rsvd) x Hugepagesize, in GiB; 0.0 when unreadable.
+    Free includes Rsvd (probed 2026-11-09, bootstorm F4) - a standing reservation counts against it."""
+    free_pages = rsvd_pages = size_kib = None
+    try:
+        with open("/proc/meminfo") as f:
+            for line in f:
+                if line.startswith("HugePages_Free:") and free_pages is None:
+                    free_pages = int(line.split()[1])
+                elif line.startswith("HugePages_Rsvd:") and rsvd_pages is None:
+                    rsvd_pages = int(line.split()[1])
+                elif line.startswith("Hugepagesize:") and size_kib is None:
+                    size_kib = int(line.split()[1])
+    except (OSError, ValueError, IndexError):
+        return 0.0
+    if free_pages is None or size_kib is None:
+        return 0.0
+    drawable = max(0, free_pages - (rsvd_pages or 0))
+    return drawable * size_kib / (1024 * 1024)
+
+
+def wait_huge_pool_settle(need_gib: float = 0.0, wait_s: float = HUGEPAGE_POOL_WAIT_S) -> float:
+    """F4: the previous engine's reserved-pool pages return to HugePages_Free only at its exit; booting
+    before they land left HugePages_Free short at the store's arm and sent the expert store to the
+    swappable tmpfs tier (the 11:05 squeeze, log:667335).  Wait (bounded) until the pool stops growing
+    for 2 s, then one line with what the boot will draw against.  -> the settled free GiB."""
+    settled = huge_pool_free_gib()
+    if settled <= 0.0:
+        return settled
+    stable, deadline = 0, time.monotonic() + max(0.0, wait_s)
+    while time.monotonic() < deadline:
+        time.sleep(0.5)
+        now = huge_pool_free_gib()
+        if now > settled + 0.05:
+            settled, stable = now, 0
+        else:
+            stable += 1
+            if stable >= 4:                              # 2 s without growth: the pool has settled
+                break
+    note = f", the store needs {need_gib:.1f} GiB" if need_gib > 0 else ""
+    warn = "  WARNING: short - the expert store may land on the swappable tmpfs tier" \
+        if 0 < need_gib and settled < need_gib else ""
+    print(f"[strata] boot: hugetlb pool settled at {settled:.1f} GiB drawable{note}{warn}", flush=True)
+    return settled
+
+
+def shadow_store_gib(env: dict | None) -> float:
+    """STRATA_SHADOW_EXPERTS' experts.bin size, rounded up to whole 2 MiB pages; 0.0 when unset/unreadable."""
+    shadow = (env or {}).get("STRATA_SHADOW_EXPERTS") or ""
+    if not shadow:
+        return 0.0
+    try:
+        size = (Path(shadow) / "experts.bin").stat().st_size
+    except OSError:
+        return 0.0
+    page = 2 << 20
+    return (max(0, size) + page - 1) // page * page / (1 << 30)
+
+
 class Service:
     def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                  vision: Vision | None = None, sampling_defaults: dict | None = None,
@@ -4518,6 +4616,31 @@
             code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
             print(f"[strata] the engine had stopped (exit code {code}); starting it again "
                   "(a minute or two) ...", flush=True)
+        # 2026-11-09 bootstorm: three cheap steps BEFORE the heavy load, in this order.
+        # (1) The old process is reaped - give its dead pack cache back now (STRATA_DROP_CACHE_ON_EXIT),
+        #     so the boot minute's kswapd cannot reclaim it mid-load (measured: +87.7 GiB pgsteal,
+        #     +16k allocstalls in one load minute).  A no-op unless opted in.
+        try:
+            from serve import cache_drop
+            cache_drop.drop_after_exit(self.engine)
+        except Exception:
+            pass
+        # (2) The old engine's reserved-pool pages return only at its exit; booting before they land
+        #     sent the expert store to the swappable tmpfs tier (the 11:05 squeeze).  Wait, bounded.
+        engine_env = getattr(self.engine, "spawn", (None,) * 7)[4] or {}
+        try:
+            wait_huge_pool_settle(shadow_store_gib(engine_env))
+        except Exception:
+            pass
+        # (3) Warm the new pack's files into the cache the heavy load will read (Prewarm's machinery,
+        #     files only); the post-READY _cycle still runs its own sweep for what the pin may drop.
+        pw = getattr(self, "prewarm", None)
+        if pw is not None:
+            try:
+                pw.warm_before_load()
+            except Exception as e:
+                print(f"[strata] prewarm: the pre-boot sweep did not finish ({type(e).__name__}: {e}); "
+                      "the boot continues", flush=True)
         try:
             self.engine.restart()
         except Exception:
@@ -8575,6 +8698,12 @@
             effort_end = effort_end_args(cfg, exe, tok)  # #458
         except ValueError as e:
             raise SystemExit(f"[strata] config {e}")
+        if a.engine == "strata" and not lazy and cfg.get("prewarm") is not None:
+            # F3 (2026-11-09 bootstorm): the FIRST boot of a fresh server has no Prewarm object yet when
+            # the engine constructor below blocks on the heavy load - sweep the pack's files into the
+            # page cache first, so the previous engine's dead cache is recycled here and the load faults
+            # over a warm cache.  Reload boots run the same sweep in Service.ensure_loaded.
+            prewarm_file_sweep_preboot(cfg, env, cfg.get("arch") or "")
         from serve import replicas as rp
         try:
             groups = rp.gpu_groups(cfg)
@@ -8908,6 +9037,15 @@
             if ready_path is not None:
                 ready_file.remove_if_owned(ready_path, svc.instance_id)
             finished.set()
+        # 2026-11-09 bootstorm: the graceful close's engine process is gone - give its pack cache back
+        # here too (the pre-2026-11-09 wiring only fired on the kill path, so the opt-in was dead on
+        # every normal stop).  A no-op unless STRATA_DROP_CACHE_ON_EXIT is set (battery-lib.sh defaults
+        # it on for battery legs).
+        try:
+            from serve import cache_drop
+            cache_drop.drop_after_exit(engine)
+        except Exception as e2:  # noqa: BLE001 - never block the exit
+            print(f"[strata] could not release the model files' cache: {e2}", flush=True)
         if interrupted:
             print("[strata] stopped", flush=True)
     exit_code = (8 if not drained else
```

```diff
--- a/campaign/battery-lib.sh
+++ b/campaign/battery-lib.sh
@@ -20,6 +20,13 @@
 REPO=${REPO:-/home/svg/LLM/strata-custom-2}
 PORT=${PORT:-8081}
 
+# 2026-11-09 bootstorm: a battery's next leg boots while THIS leg's pack still sits in the page cache -
+# the measured storm (kswapd +87.7 GiB pgsteal, +16k allocstalls in one load minute) is exactly that
+# dead cache being reclaimed mid-load.  Drop it at every server stop instead (the drop runs after the
+# engine process ends, where nothing latency-critical competes).  Off by default outside batteries: a
+# same-model restart wants the cache warm.  An explicit value always wins.
+export STRATA_DROP_CACHE_ON_EXIT=${STRATA_DROP_CACHE_ON_EXIT:-1}
+
 port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
 
 cleanup_engine() {
```

**Price.** The 77.9 GiB dead cache is recycled by the sweep's own reads (NVMe-paced, ~8-16 s at 5-10 GiB/s) instead of by kswapd mid-load; the drop-cache default makes every battery stop pay it at exit, where nothing latency-critical competes. **Boot-to-READY −10-25 s in battery/reload cycles** (the sweep moves pre-spawn and the heavy load's src reads stop racing kswapd); **storm: boot-minute pgsteal_kswapd 87.7 GiB → ≤ ~20 GiB target** (residual = the new engine's own draw), **allocstall_movable 16,236/min → < 1,000**, **compact_fail 1,732/min → under the ≤1,000/cycle gate** (2026-11-07-preboot-aging-proof H3). The tmpfs-store squeeze case additionally loses its 31.64 GiB shmem-vs-pool double-store once F4 lands (§5).

**Law-17 test.** Same-window battery-leg pair on the same pack: leg A with the patch, leg B with `STRATA_DROP_CACHE_ON_EXIT=0` and the sweep knob off (`prewarm.files: false`), interleaved A/B/A/B. Witnesses: vmstat delta over each boot minute (`pgsteal_kswapd`, `allocstall_movable`, `compact_stall/compact_fail`), the `pre-boot sweep` line timestamped before the first engine line, boot-to-READY wall clock. Acceptance: boot-minute pgsteal ≤ 20 GiB and compact_fail ≤ 1,000 on A; B reproduces the storm numbers (the pair prices the fix, not the box); tok/s lanes unchanged (law 13). Boundary: the fresh-process hook needs the `prewarm` section present in the run config — batteries carry it.

## 5. F4 (accounting + launcher) — the honest pool check and the settle wait

`include/strata/platform/hugepage.hpp` + the server.py hunks above (already shown in §4). Two facts drive this: the store's reservation is invisible to a Free-only check (§1 probe), and the 11:05 tmpfs-tier miss was cross-engine overlap — the launcher booted the next engine while the previous one still held 69.5 GiB of the pool (its pages return to `HugePages_Free` only at exit, instantly). The engine-side attempt-first (§3) fixes the within-boot race; `huge_pool_covers`/`huge_pool_free_bytes` subtracting `Rsvd` makes every subsequent cover check honest; the launcher's bounded settle wait closes the overlap gap. The alternative the task floated — freeing the tmpfs copy before boot — is **rejected with evidence**: when `STRATA_SHADOW_EXPERTS` points at `/dev/shm`, that tmpfs file is the memfd arm's memcpy SOURCE (`shadow_experts.cpp:137-165` opens `st.dir + "/experts.bin"`); unlinking it before boot breaks the arm's source and the size check (`sb.st_size != want` refuses the whole shadow). It stays as F-367's standing RAM copy.

```diff
--- a/include/strata/platform/hugepage.hpp
+++ b/include/strata/platform/hugepage.hpp
@@ -38,38 +38,47 @@
 }
 
 #if defined(__linux__)
-/// Bytes free in the kernel's reserved 2 MiB pool (HugePages_Free x Hugepagesize); 0 when unreadable.
-/// MemAvailable does NOT count the reserved pool: any memory budget whose allocation can draw from
-/// the pool must add this (the same correction dsv41's copy budget carries, ds4_os.hpp).
+/// Bytes DRAWABLE from the kernel's reserved 2 MiB pool ((HugePages_Free - HugePages_Rsvd) x
+/// Hugepagesize); 0 when unreadable.  MemAvailable does NOT count the reserved pool: any memory budget
+/// whose allocation can draw from the pool must add this (the same correction dsv41's copy budget
+/// carries, ds4_os.hpp).  F4 (2026-11-09 bootstorm, probed on this box): HugePages_Free INCLUDES the
+/// reserved subset - a store's standing MAP_SHARED reservation leaves Free untouched while Rsvd counts
+/// against it, so Free alone overstates what the next mmap can draw; the honest free subtracts Rsvd.
 inline uint64_t huge_pool_free_bytes() {
     std::ifstream f("/proc/meminfo");
     if (!f) return 0;
-    uint64_t pages = 0, size_kb = 0;
+    uint64_t pages = 0, rsvd = 0, size_kb = 0;
     std::string line;
     while (std::getline(f, line)) {
         unsigned long long v = 0;
         if (std::sscanf(line.c_str(), "HugePages_Free: %llu", &v) == 1) pages = v;
+        else if (std::sscanf(line.c_str(), "HugePages_Rsvd: %llu", &v) == 1) rsvd = v;
         else if (std::sscanf(line.c_str(), "Hugepagesize: %llu", &v) == 1) size_kb = v;
     }
-    return pages * size_kb * 1024ull;
+    return (pages > rsvd ? pages - rsvd : 0) * size_kb * 1024ull;
 }
 
 /// True when the reserved 2 MiB pool can cover `bytes` whole (MAP_HUGETLB is all-or-nothing per
-/// mapping).  Read from /proc/meminfo at call time - the pool shrinks as other organs draw it.
+/// mapping).  Read from /proc/meminfo at call time - the pool shrinks as other organs draw it, and a
+/// standing reservation (the shadow store's MAP_SHARED reserve) counts against the cover (Free
+/// includes Rsvd; the honest cover test is Free - Rsvd, F4 2026-11-09 bootstorm).
 inline bool huge_pool_covers(size_t bytes) {
     const size_t page = 2ull << 20;
     const size_t rounded = (bytes + page - 1) / page * page;
-    uint64_t free_kib = 0;
+    uint64_t free_kib = 0, rsvd_kib = 0;
     if (std::ifstream f("/proc/meminfo"); f) {
         std::string line;
-        while (std::getline(f, line))
+        while (std::getline(f, line)) {
             if (line.rfind("HugePages_Free:", 0) == 0) {
                 unsigned long n = 0;
                 if (std::sscanf(line.c_str(), "HugePages_Free: %lu", &n) == 1) free_kib = n;
-                break;
+            } else if (line.rfind("HugePages_Rsvd:", 0) == 0) {
+                unsigned long n = 0;
+                if (std::sscanf(line.c_str(), "HugePages_Rsvd: %lu", &n) == 1) rsvd_kib = n;
             }
+        }
     }
-    return free_kib * (page / 1024) >= rounded / 1024;
+    return (free_kib > rsvd_kib ? free_kib - rsvd_kib : 0) * (page / 1024) >= rounded / 1024;
 }
 #else
 inline bool huge_pool_covers(size_t) { return false; }
```

(The F4 server.py hunks — `huge_pool_free_gib`, `wait_huge_pool_settle`, `shadow_store_gib`, and step (2) of the `ensure_loaded` sequence — are in §4's diff. Helper smoke on this box today: `free_gib: 88.04`, `shadow_gib(/dev/shm): 31.64`, settle returned in 2.0 s with the numbers line printed.)

**Price.** M5 goes conditional → eliminated whenever the pool is armed to spec: per squeezed boot the store's 31.64 GiB stays unevictable pool pages instead of swappable 4 KiB shmem — **pswpout −14,386 pages, zram peak 94 % → < 50 %, engine Swap 840 MiB → 0, MemAvailable squeeze floor 14.4 → ~40 GiB class**. Cost: the settle wait adds ≤ 2 s (probed: pool settles 2.0 s after a clean exit); the KV-shadow starvation ordering flips only when `STRATA_POOL_SMALL_ARENAS=1` re-arms small arenas (then the complement's cover check — now Rsvd-honest — refuses and falls to the one-shot path, the same organ the ≥8 GiB gate already protects).

**Law-17 test.** Three battery cycles with deliberate overlap (fire the next boot before `cleanup_engine` finishes — reproduce the 11:05 shape on purpose), same pack. Witnesses per boot: every boot logs `2 MiB pool pages (memfd, reserved and populated ...)`; zero boots log `the pool could not reserve the store`; `hugetlb pool settled at X GiB drawable` shows the post-exit jump; `HugePages_Rsvd` ≈ 15,820 pages at store-arm then absorbed at populate; zram < 50 % through the load minute; engine Swap 0 at READY. Law 13: GSQ/UD banks unchanged. Red on any witness voids the boot window (law 17 hygiene) and books a re-run.

## 6. Apply and verify

```
cd /home/svg/LLM/strata-custom-2
git apply --check .docs/research/digest/2026-11-09-bootstorm-fixes.patch   # verified clean today
git apply        .docs/research/digest/2026-11-09-bootstorm-fixes.patch
g++ -fsyntax-only -std=c++20 -I include -I /usr/local/cuda/include src/core/shadow_experts.cpp   # verified rc=0
g++ -fsyntax-only -std=c++20 -I include -I /usr/local/cuda/include src/core/expert_source.cpp    # verified rc=0
python3 -m py_compile serve/server.py                                                            # verified
```

No `src/` file was edited for this digest; the diffs were written in `/tmp/bootstorm-scratch` and validated there, then checked against the repo with `git apply --check`.

## 7. Boundaries

- **Partial-pool middle case**: pool drawable 20 GiB < store 31.64 GiB still lands on tmpfs — atomically, and now named with numbers in the boot line. A chunked multi-memfd partial store is future machinery; MAP_HUGETLB is all-or-nothing per mapping.
- **Pre-boot sweep holding the load**: capped at `min(prewarm.timeout_s, 600 s)`; on this box's NVMe the 77.9 GiB pack is ~8-16 s. A config without a `prewarm` section gets the pool-settle and drop-cache steps but no sweep.
- **First-boot sweep is Linux-shaped** (`posix_fadvise` guard already inside `_warm_files`); Windows keeps the cold path as before.
- **`huge_pool_free_bytes()` semantics changed** (now Free − Rsvd): its one caller (`expert_source.cpp:3397`, the physical-memory budget) gets the honest drawable value; dsv41's `ds4_pool.hpp` reads `/proc/meminfo` itself and is untouched.
- **MADV_POPULATE_WRITE needs kernel ≥ 5.14** (both .cpp files carry the `#if !defined` numeric guard, so pre-5.14 glibc headers compile; runtime EINVAL falls back to the old loops — verified shapes retained).
- **Cross-boundary**: F1's `−6-12 s` and F3's `−10-25 s` are the qwen-next digest's priced bands plus this probe's rates; the discriminating A/B boots are the law-17 tests above, not yet run — engines were up under the operator all session. The GPU clock-v2 confound ledger (2026-11-09-regression-memory-state §Attribution) still applies to any lane number read alongside these fixes.
