# SCOUT DIGEST — Cold/warm durability (RTX 5090, 184GB RAM, ~45GB MoE GGUF via mmap)

Scope: 6 comparable implementations for pinning page cache, protecting it from eviction, and persisting warm artifacts / parked KV across restarts. Date 2026-10-25.

Grades: A = directly applicable mechanics + real failure data; B = strong pattern, gaps to verify; C = indirect only.

---

## 1. vmtouch page-cache pinning — GRADE A-
**Mechanics.** vmtouch mmaps target files and walks pages; flags (from vmtouch.pod): `-t` touch (fault in, NOT protected), `-l` mlock + stay resident, `-L` mlockall (locks unrelated pages — avoid), `-d` daemonize after crawl (requires -l/-L), `-P pidfile`, `-m SIZE` max file size (**default 500M — our 45GB GGUFs are silently skipped without `-m 64G`**), `-p A-B` partial-range pinning (pin only hot expert slices), `-b list` batch file list, `-0` NUL-delimited, `-e` evict (inverse — perfect for fault injection), `-v/-q`.
**Limits.** mlock needs RLIMIT_MEMLOCK (distro default 64KB–8MB); root or CAP_IPC_LOCK bypasses; systemd services ignore limits.conf — use `LimitMEMLOCK=infinity` in the unit (Elasticsearch precedent, github.com/elastic/elasticsearch#22733/22734). Since Linux 2.6.9 there is NO system-wide total-locked-memory cap, so 45GB is fine once the rlimit is raised. `vm.max_map_count` bounds files-per-process (one VMA per file) — irrelevant at our file count. NFS ok; do not vmtouch S3FS/FUSE mounts.
**Real-world.** r/LocalLLaMA "Speed Up Your Model Loading Times with vmtouch" (1ghtnb7) is the canonical LLM-GGUF precache post (Reddit body unfetchable from here; pattern confirmed by multiple secondary citations).
**Failure modes.** mlock EPERM/ENOMEM at rlimit; locked pages are unswappable and unreclaimable, so the OOM-killer will hunt *other* processes under pressure — keep ~2× model headroom on 184GB; if the vmtouch process dies, pages unlock and become evictable again (this is a feature: run the pin as an independent systemd service so server restarts never unlock).
**Key subtlety for us.** Pin from a separate tiny daemon: mlock on file pages protects the *page cache* pages, which our server's mmap mapping shares — zero copy, inference RSS stays low.

## 2. cgroup v2 memory.min / memory.low — GRADE B
**Mechanics.** Page cache is charged per-page, on first fault, to the cgroup that touched it, and stays charged until eviction (closing the fd does not release the charge). `memory.min` = hard floor (reclaim skips these pages), `memory.low` = best-effort weight; both bounded by *effective* parent protection (kernel docs: effective boundary propagates down, capped by unclaimed parent headroom; root cgroup gives no protection). systemd: `MemoryMin=` / `MemoryLow=` on the unit.
**Telemetry.** memory.current (includes file cache), memory.stat `file`, `file_mapped`, `inactive_file`, `active_file`, memory.events counters (`low/high/max/oom_kill`), memory.pressure PSI, and crucially `workingset_refault_file` — the direct eviction-thrash counter for our falsifier. Debugging: /proc/pid/pagemap + /proc/kpagecgroup (biriukov's `cgtouch`).
**Gotchas (reported).** (a) Protection only binds the *owning* cgroup — pinning from a service outside the server's cgroup protects nothing for memcg purposes; (b) `memory.high` set aggressively causes direct-reclaim of page cache → I/O thrash (serverfault 1166165); (c) memory.min hoards under global pressure: kernel docs warn large min can push the system toward global OOM — the OOM-killer can still kill your process even though its cache is protected; (d) writeback I/O charged to the originating memcg; vm.dirty_* applied whichever threshold first; (e) author guidance: prefer `low` set near working-set size, avoid hard min/max.
**Verdict.** Use as belt-and-suspenders + telemetry around the mlock pin, not as the primary mechanism.

## 3. Ollama keep-alive + reload path — GRADE B (behavior contract, not mechanism)
**Mechanics.** Default keep_alive 5m; per-request `keep_alive` on /api/generate & /api/chat; `OLLAMA_KEEP_ALIVE` env; `-1` forever, `0` unload immediately, `ollama stop` to evict; empty-prompt request = preload (their prewarm analog). Concurrency knobs: OLLAMA_MAX_LOADED_MODELS (default 3×GPU count), OLLAMA_NUM_PARALLEL (KV RAM scales ×N), OLLAMA_MAX_QUEUE=512 → 503.
**What stays warm.** While "loaded": weights resident + KV/prefix cache. On unload: KV destroyed, weight mappings dropped — only the OS page cache of the GGUF may survive, so reload skips disk I/O *only if* pages weren't evicted. There is no cross-restart KV persistence at all; warm speed depends 100% on unprotected page cache — precisely the hole our row closes.
**Measured.** Users report 15–25s per-request latency when the model unloads between calls vs sub-second first token warm (stackoverflow 79526074). keep_alive=0 also used as workaround for KV/memory accumulation bugs (issue #17875).
**Takeaway.** Copy their keep_alive/preload API surface; beat their reload path (our 60–90s baseline).

## 4. LMCache local-disk backend — GRADE B+ (best design pattern for parked-KV)
**Mechanics (docs.lmcache.ai local_storage).** One file per KV chunk under the `local_disk: file://...` dir; in-memory metadata dict `CacheEngineKey → DiskCacheMetadata(path, size, shape, dtype, pinned, positions)`; eviction LRU (policy-configurable); a priority-queue worker runs PUT (pri 2) / DELETE (1) / PREFETCH (0); `chunk_size` (e.g. 256 tok); `local_disk_path_sharding`; `use_odirect` bypasses page cache (recommended when CPU RAM is scarce — keeps our pin budget clean); prefetch lifts disk→LocalCPUBackend while keeping the disk copy.
**Pin.** `pinned` flag exists in DiskCacheMetadata (excluded from eviction) but docs expose no user pin API — internal only.
**Cross-restart lookup.** Docs describe lookup via the in-memory dict only; no on-disk index documented. Whether init rescans the directory to rebuild the metadata index is unverified (my raw-source fetches 404'd — repo paths moved; resolve via `gh api repos/LMCache/LMCache` tree before copying the pattern). If it doesn't rescan, restart = all parked KV invisible despite files on disk — the exact bug class we must avoid.
**Keying.** Key→file is hash-of-key-string (xxhash per repo references); real failure modes in issues: #1767 token-database prefix-hash bug breaking cache hits; #2878 vLLM `cache_salt` missing from the key (collision/wrong-reuse class).
**Copy.** Chunked, prefix-hash-keyed files + metadata carrying dtype/shape for validate-on-load + explicit pinned flag surviving eviction.

## 5. vLLM / SGLang persistence — GRADE B-
**vLLM.** Prefix/APC cache lives only in GPU/CPU memory; no restart persistence (docs.vllm.ai prefix_caching; discuss.vllm.ai thread 1663; RFC #44223 for reuse interface; llm-d built prefix-aware routing precisely because naive restarts lose the cache). What DOES persist: torch.compile artifacts at `~/.cache/vllm/torch_compile_cache` — explicitly copyable across hosts/machines (v0.8.3 docs), keyed by model/config hash, brittle on config drift (#16501), disable via VLLM_DISABLE_COMPILE_CACHE. → This config-hash keying is the pattern for our warm-artifact manifest.
**SGLang HiCache.** Tiers L1 GPU / L2 host DRAM (per-instance; `--hicache-ratio`, `--hicache-size`) / L3 storage backends `--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic}` (file backend defaults `/tmp/hicache`). Write policies `write_through` / `write_through_selective` (frequency threshold) / `write_back`; prefetch policy `best_effort|wait_complete|timeout` (prod rec: timeout, base 2s + 0.1s/Ktok, max 30s); mem layout `layer_first|page_first|page_first_direct`. L3 lives outside the process → de-facto cross-restart KV persistence, but the docs make no restart-survival guarantee. Failure modes: node-local /tmp mistaken for shared cache; non-linear hit-rate saturation; prefetch-policy latency traps; rank sync via all_reduce(min) on prefetch decisions.

## 6. llama.cpp mmap/mlock in practice — GRADE A- (our exact stack: MoE GGUF + RTX + expert offload)
**Flags.** 2026 loader refactor (PR #20834) replaced `--no-mmap`/`--mlock` with `--load-mode {none,mmap,mlock,mmap+mlock}`; `mlock` *implies* mmap — "no mmap + pin" became inexpressible (issue #26110; maintainer: "--mlock without --mmap is basically a noop"). Closed as completed 2026-07-27 via PR #26135 (direction: pin only CPU-offloaded buffers via llama_mlock, decoupled from the mmap region).
**Measured MoE failure data (#26110)** — gpt-oss-120b F16 ~65GB, RTX 5070 Ti 16GB, 60GB RAM, `--n-cpu-moe`:
- `--load-mode none` (read at load, no lazy paging): OS swaps out CPU-resident expert pages under sustained load; 25–27 t/s → **7.55 t/s**. This is literally our row's failure mode (unpinned warm band decaying under pressure).
- `--load-mode mlock`: mmap region + separate CPU buffer for overridden tensors = double buffering; required RAM ≈ full file + offloaded portion → OOM-kill at `--n-cpu-moe 31` (file-rss ~57GB). Reproduced on GLM-5.2-UD-IQ2_M by a second user.
** mmap default: lazy paging, fast start, low RSS, but page-fault stalls whenever pages were evicted; `none` = slower load, no lazy faults. Consensus: mmap+mlock is the sweet spot when RAM is ample — ours is (184GB vs ~50GB file). On small-RAM boxes mlock fails at RLIMIT_MEMLOCK / OOM (Raspberry Pi reports).
**Implication.** Our server should run plain mmap for experts and hold the pin *externally* (vmtouch daemon) — avoiding llama.cpp's in-process double-buffer trap entirely.

---

## PIN PLAN (43–50GB GGUF tier, 184GB RAM)

```bash
# 0) Sanity: expect default tiny memlock
ulimit -l; free -g   # pin ~46G of 184G → keep >100G headroom for anon/KV

# 1) Independent pinning daemon (survives inference-server restarts)
sudo tee /etc/systemd/system/gguf-pin.service <<'EOF'
[Unit]
Description=Pin MoE GGUF weights in page cache
After=local-fs.target
[Service]
Type=simple
ExecStart=/usr/local/bin/vmtouch -dl -m 64G -q /models/moe/model.gguf
LimitMEMLOCK=infinity
# Alternative to LimitMEMLOCK: AmbientCapabilities=CAP_IPC_LOCK CapabilityBoundingSet=CAP_IPC_LOCK
OOMPolicy=continue
Restart=on-failure
Nice=5
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now gguf-pin
grep VmLck /proc/$(pidof vmtouch)/status        # expect ~47000000 kB
vmtouch /models/moe/model.gguf                   # expect 100% resident, no 'e'
# Multi-file tier: pin via list, not glob-in-shell
# ExecStart=/usr/local/bin/vmtouch -dl -m 64G -q -b /etc/gguf-pin.list

# 2) Belt-and-suspenders: protect the SERVER's own cgroup touch-charge
sudo systemctl edit llama-server   # add: [Service] MemoryLow=96G
# (avoid MemoryMin=48G alone under global pressure — OOM risk; low is the safer floor)

# 3) Fault injection handles for the falsifier
vmtouch -e /models/moe/model.gguf                 # deliberate evict
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches  # nuclear evict (test only)
```
Rules: `-m 64G` mandatory (default 500M cap skips our files). Never `-L` (mlockall). One pin process per tier; if it dies, pins drop — alert on unit failure + `workingset_refault_file` delta.

## OUR NEXT STEP — eviction-assault falsifier (`evict_assault.sh` sketch)

```
Setup: pin daemon on; 20-prompt warm probe -> record baseline t/s (expect 219–234)
Gates on: t/s, /proc/<server>/stat minflt, cgroup memory.events + workingset_refault_file,
          MemFree, PSI memory.some, VmLck of pin daemon

A) CLEAN EVICT:  vmtouch -e <gguf>; drop_caches=3; probe 20 prompts
   FAIL if t/s within 10% of baseline without re-pin (proves pin needed)
   then wait for daemon re-touch (it holds the lock; re-run -t if -l not yet active)

B) PRESSURE:     hammer proc mallocs anon until MemFree<5% for 60s; sample every 5s
   PASS if t/s flat AND VmLck steady AND refault_file delta ~0

C) RESTART DURABILITY: kill server -> run assault B hard 10min -> cold-start server
   PASS if first-quality-token < 15s, no prewarm call needed, t/s>=200 in first 60s

D) RLIMIT REGRESSION: run pin under default ulimit -l; capture EPERM/ENOMEM signature
   (document for the runbook)

Output: CSV (assault, t, tps, minflt, refaults, MemFree, psi_some, VmLck) + gate verdicts
```
This falsifies the claim "our warm band survives pressure/idle" directly against kernel behavior instead of trusting vmtouch.

---

## Sources
- vmtouch: github.com/hoytech/vmtouch (+TUNING.md, vmtouch.pod); manpages.debian.org/testing/vmtouch/vmtouch.8; reddit.com/r/LocalLLaMA/comments/1ghtnb7; elastic/elasticsearch#22734; access.redhat.com/solutions/1257953
- cgroup v2: docs.kernel.org/admin-guide/cgroup-v2.html; biriukov.dev/docs/page-cache/6-cgroup-v2-and-page-cache; facebookmicrosites.github.io/cgroup2/docs/memory-controller.html; serverfault.com/questions/1166165
- Ollama: docs.ollama.com/faq; github.com/ollama/ollama/issues/17875; stackoverflow.com/questions/79526074
- LMCache: docs.lmcache.ai/kv_cache/storage_backends/local_storage.html (+/dev/ variant); blog.lmcache.ai raw-block post; LMCache#1767, #2878, #2113, #3262; arxiv.org/abs/2510.09665
- vLLM/SGLang: docs.vllm.ai/en/stable/design/prefix_caching; docs.vllm.ai/en/v0.8.3/design/v1/torch_compile.html; vllm-project/vllm#44223, #16501; discuss.vllm.ai/t/1663; docs.sglang.io/advanced_features/hicache_design.html; lmsys.org/blog/2025-09-10-sglang-hicache
- llama.cpp: ggml-org/llama.cpp#26110 (+PR #26135, PR #20834); discussion #19883; news.ycombinator.com/item?id=35426679
