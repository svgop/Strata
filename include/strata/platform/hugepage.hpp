// T35 C2 (STRATA_HUGEPAGE_ARENAS=1, default off): back the large pinned host arenas with 2 MiB pages.
//
// The CPU-expert complement (tens of GiB) and the KV host shadow (1.55 GiB flat per process) are read by the CPU pool
// and DMA'd by the GPU.  With 4 KiB pages the pool's DMAs walk ~9M PTEs and pay the IOTLB/dTLB tax F-201-zen5 #6
// priced at +5-15% on the staging path; 2 MiB pages cut that to ~18k.
//
// HOSTMEM-NO-REBOOT (2026-11-07): the reserved 2 MiB pool comes FIRST (MAP_HUGETLB) - pool pages cannot
// fragment, swap, or age, and the map either fits the pool or fails instantly (no THP policy, no
// compaction stall).  The original THP/madvise path stays as the fallback for an empty or short pool.
// The registered mapping must be faulted 2 MiB BEFORE cudaHostRegister pins it, which both routes do.
// The boot line reports the outcome by reading /proc/self/smaps ("accepted" is measured, never
// assumed; hugetlb VMAs carry the smaps VmFlags "ht" bit and count fully).  Linux only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#endif

namespace strata::platform {

/// The gate: STRATA_HUGEPAGE_ARENAS=1.
inline bool hugepage_arenas_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HUGEPAGE_ARENAS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

#if defined(__linux__)
/// True when the reserved 2 MiB pool can cover `bytes` whole (MAP_HUGETLB is all-or-nothing per
/// mapping).  Read from /proc/meminfo at call time - the pool shrinks as other organs draw it.
inline bool huge_pool_covers(size_t bytes) {
    const size_t page = 2ull << 20;
    const size_t rounded = (bytes + page - 1) / page * page;
    uint64_t free_kib = 0;
    if (std::ifstream f("/proc/meminfo"); f) {
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("HugePages_Free:", 0) == 0) {
                unsigned long n = 0;
                if (std::sscanf(line.c_str(), "HugePages_Free: %lu", &n) == 1) free_kib = n;
                break;
            }
    }
    return free_kib * (page / 1024) >= rounded / 1024;
}
#else
inline bool huge_pool_covers(size_t) { return false; }
#endif

struct HugeMapping {
    void* addr = nullptr;   ///< the 2 MiB-aligned mapping, null when the allocation failed
    size_t bytes = 0;       ///< its rounded size (a multiple of 2 MiB)
    bool hugetlb = false;   ///< backed by the reserved 2 MiB pool (MAP_HUGETLB), not by THP
};

#if defined(__linux__)

/// mmap a region for `bytes` on 2 MiB pages: the reserved pool first, the THP madvise fallback second.
/// STRATA_NO_LARGEPAGES skips the pool attempt (the same-run A/B switch pinned.cu honors).
inline HugeMapping huge_alloc(size_t bytes) {
    HugeMapping out;
    const size_t page = 2ull << 20;
    if (bytes == 0) return out;
    const size_t rounded = (bytes + page - 1) / page * page;
    if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr) {
        void* p = mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {   // 2 MiB-aligned and -sized by construction; no madvise needed
            out.addr = p;
            out.bytes = rounded;
            out.hugetlb = true;
            return out;
        }
        // empty or short pool (MAP_HUGETLB is all-or-nothing per mapping) - the THP path below
    }
    const size_t total = rounded + page;   // one extra page of slack, for aligning the start
    void* raw = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return out;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + page - 1) & ~(uintptr_t) (page - 1);
    const size_t head = aligned - start;
    const size_t tail = total - head - rounded;
    if (head > 0) (void) munmap((void*) start, head);
    if (tail > 0) (void) munmap((void*) (aligned + rounded), tail);
    if (madvise((void*) aligned, rounded, MADV_HUGEPAGE) != 0) {
        (void) munmap((void*) aligned, rounded);
        return out;
    }
    out.addr = (void*) aligned;
    out.bytes = rounded;
    return out;
}

inline void huge_free(HugeMapping& m) {
    if (m.addr != nullptr) (void) munmap(m.addr, m.bytes);
    m.addr = nullptr;
    m.bytes = 0;
    m.hugetlb = false;
}

/// Sum 2 MiB-backed bytes over the VMAs overlapping [addr, addr+bytes): AnonHugePages (THP) plus
/// hugetlb VMAs whole (smaps VmFlags "ht" - every page of one is a reserved-pool 2 MiB page).
/// Returns the huge bytes, the covered bytes, and the pool-backed subset; measured, never assumed.
inline void hugepage_measure(const void* addr, size_t bytes, uint64_t& huge_bytes, uint64_t& covered_bytes,
                             uint64_t& ht_bytes) {
    huge_bytes = covered_bytes = ht_bytes = 0;
    std::ifstream smaps("/proc/self/smaps");
    if (!smaps) return;
    const uintptr_t lo = (uintptr_t) addr, hi = lo + bytes;
    uintptr_t start = 0, end = 0;
    bool in = false, vma_ht = false;
    uint64_t vma_huge_kib = 0;
    std::string line;
    auto flush = [&] {
        if (!in || end <= lo || start >= hi) return;
        const uintptr_t a = start > lo ? start : lo, b = end < hi ? end : hi;
        covered_bytes += (uint64_t) (b - a);
        if (vma_ht) {
            huge_bytes += (uint64_t) (b - a);
            ht_bytes += (uint64_t) (b - a);
        } else {
            huge_bytes += vma_huge_kib * 1024ull;
        }
        in = false;
    };
    while (std::getline(smaps, line)) {
        unsigned long a = 0, b = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx", &a, &b) == 2) {
            flush();
            start = a;
            end = b;
            in = true;
            vma_ht = false;
            vma_huge_kib = 0;
        } else if (in && line.rfind("AnonHugePages:", 0) == 0) {
            unsigned long kib = 0;
            if (std::sscanf(line.c_str(), "AnonHugePages: %lu", &kib) == 1) vma_huge_kib = kib;
        } else if (in && line.rfind("VmFlags:", 0) == 0) {
            if (line.find(" ht") != std::string::npos) vma_ht = true;
        }
    }
    flush();
}

/// One boot line: what the pages really are (smaps), and the kernel's THP mode.
inline void hugepage_report(const char* what, const void* addr, size_t bytes) {
    uint64_t huge = 0, covered = 0, ht = 0;
    hugepage_measure(addr, bytes, huge, covered, ht);
    std::string thp = "unknown";
    if (std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled"); f) {
        std::string mode;
        std::getline(f, mode);
        thp = mode;
    }
    const double mib = 1048576.0;
    std::fprintf(stderr, "strata: hugepages: %s: %.2f GiB of %.2f GiB in 2 MiB pages "
                         "(%s; THP=%s)\n", what, (double) huge / (1024.0 * mib), (double) covered / (1024.0 * mib),
                 ht > 0 ? "HUGETLB POOL" : (huge > 0 ? "THP accepted" : "rejected"), thp.c_str());
}

#else  // not Linux

inline HugeMapping huge_alloc(size_t) { return {}; }
inline void huge_free(HugeMapping& m) { m = {}; }
inline void hugepage_measure(const void*, size_t, uint64_t& huge, uint64_t& covered, uint64_t& ht) {
    huge = covered = ht = 0;
}
inline void hugepage_report(const char*, const void*, size_t) {}

#endif

}  // namespace strata::platform
