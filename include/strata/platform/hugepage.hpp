// T35 C2 (STRATA_HUGEPAGE_ARENAS=1, default off): back the large pinned host arenas with 2 MiB transparent huge pages.
//
// The CPU-expert complement (tens of GiB) and the KV host shadow (1.55 GiB flat per process) are read by the CPU pool
// and DMA'd by the GPU.  With 4 KiB pages the pool's DMAs walk ~9M PTEs and pay the IOTLB/dTLB tax F-201-zen5 #6
// priced at +5-15% on the staging path; 2 MiB pages cut that to ~18k.  This box has THP=madvise, so the hint is
// honoured only when asked, and it must be asked BEFORE cudaHostRegister pins the pages (the register faults them).
//
// A mapping is anonymous, 2 MiB-aligned and 2 MiB-sized, MADV_HUGEPAGE'd, then registered by the caller.  The boot
// line reports the outcome by reading /proc/self/smaps, so "accepted" is measured, never assumed.  Linux only; the
// other systems get the empty/false forms and every caller falls back to today's allocation.

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

struct HugeMapping {
    void* addr = nullptr;   ///< the 2 MiB-aligned mapping, null when the allocation failed
    size_t bytes = 0;       ///< its rounded size (a multiple of 2 MiB)
};

#if defined(__linux__)

/// mmap an anonymous region for `bytes`, align it to and round it up to 2 MiB, MADV_HUGEPAGE it.  Empty on failure.
inline HugeMapping huge_alloc(size_t bytes) {
    HugeMapping out;
    const size_t page = 2ull << 20;
    if (bytes == 0) return out;
    const size_t rounded = (bytes + page - 1) / page * page;
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
}

/// Sum AnonHugePages over the VMAs overlapping [addr, addr+bytes); returns the huge bytes and the covered bytes.
inline void hugepage_measure(const void* addr, size_t bytes, uint64_t& huge_bytes, uint64_t& covered_bytes) {
    huge_bytes = covered_bytes = 0;
    std::ifstream smaps("/proc/self/smaps");
    if (!smaps) return;
    const uintptr_t lo = (uintptr_t) addr, hi = lo + bytes;
    uintptr_t start = 0, end = 0;
    bool in = false;
    uint64_t vma_huge_kib = 0;
    std::string line;
    auto flush = [&] {
        if (!in || end <= lo || start >= hi) return;
        const uintptr_t a = start > lo ? start : lo, b = end < hi ? end : hi;
        covered_bytes += (uint64_t) (b - a);
        huge_bytes += vma_huge_kib * 1024ull;
        in = false;
    };
    while (std::getline(smaps, line)) {
        unsigned long a = 0, b = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx", &a, &b) == 2) {
            flush();
            start = a;
            end = b;
            in = true;
            vma_huge_kib = 0;
        } else if (in && line.rfind("AnonHugePages:", 0) == 0) {
            unsigned long kib = 0;
            if (std::sscanf(line.c_str(), "AnonHugePages: %lu", &kib) == 1) vma_huge_kib = kib;
        }
    }
    flush();
}

/// One boot line: what the pages really are (smaps), and the kernel's THP mode.
inline void hugepage_report(const char* what, const void* addr, size_t bytes) {
    uint64_t huge = 0, covered = 0;
    hugepage_measure(addr, bytes, huge, covered);
    std::string thp = "unknown";
    if (std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled"); f) {
        std::string mode;
        std::getline(f, mode);
        thp = mode;
    }
    const double mib = 1048576.0;
    std::fprintf(stderr, "strata: hugepages: %s: %.2f GiB of %.2f GiB accepted in 2 MiB pages "
                         "(%s; THP=%s)\n", what, (double) huge / (1024.0 * mib), (double) covered / (1024.0 * mib),
                 huge > 0 ? "accepted" : "rejected", thp.c_str());
}

#else  // not Linux

inline HugeMapping huge_alloc(size_t) { return {}; }
inline void huge_free(HugeMapping& m) { m = {}; }
inline void hugepage_measure(const void*, size_t, uint64_t& huge, uint64_t& covered) { huge = covered = 0; }
inline void hugepage_report(const char*, const void*, size_t) {}

#endif

}  // namespace strata::platform
