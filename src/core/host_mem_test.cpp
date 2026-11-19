#include "strata/core/host_mem.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = available_host_bytes();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
    {   // evict_for_ram (upstream #5697 + #2253): the admission-gate eviction loop's three stops.
        struct Sim {
            std::optional<uint64_t> avail;      // what the reading says (nullopt: unknown)
            std::vector<uint64_t> entries;      // bytes of each evictable parked conversation, oldest first
            size_t evictions = 0;
            std::vector<uint64_t> gains;        // what the reading gains per eviction (default: the entry's bytes)
            bool admit(uint64_t allocation, uint64_t floor) const {
                return avail && *avail >= floor && allocation <= *avail - floor;
            }
        };
        auto run = [](Sim& s, uint64_t allocation, uint64_t floor, size_t max_evictions,
                      uint64_t required, uint64_t evictable) {
            return evict_for_ram(
                [&] { return s.admit(allocation, floor); },
                [&]() -> std::optional<uint64_t> { return s.avail; },
                [&]() -> size_t {
                    if (s.evictions >= s.entries.size()) return 0;
                    const size_t freed = (size_t) s.entries[s.evictions];
                    const uint64_t gain = s.evictions < s.gains.size() ? s.gains[s.evictions] : freed;
                    ++s.evictions;
                    if (s.avail) *s.avail += gain;
                    return freed;
                },
                max_evictions, required, evictable);
        };
        {   // fits after two evictions: the loop stops the moment admit() passes
            Sim s{{1000}, {300, 300, 300}, 0, {}};
            const auto r = run(s, 1150, 200, 8, 1150 + 200, 900);
            check(r.fits && r.evicted == 2 && !r.no_progress && !r.cannot_help, "two evictions admit the park");
        }
        {   // #5697 no-progress: the first eviction gives nothing back, the rest stay parked
            Sim s{{1000}, {300, 300, 300}, 0, {0}};
            const auto r = run(s, 900, 200, 8, 0, UINT64_MAX);
            check(!r.fits && r.evicted == 1 && r.no_progress && !r.cannot_help,
                  "an eviction that gave nothing back stops the loop after one entry");
        }
        {   // #2253 cannot-help: short by more than everything evictable, nothing is evicted at all
            Sim s{{100}, {300, 300}, 0, {}};
            const auto r = run(s, 900, 200, 8, 900 + 200, 600);
            check(!r.fits && r.evicted == 0 && r.cannot_help && !r.no_progress && r.short_by == 1000,
                  "short by more than all that may leave: nothing evicted, short_by reported");
        }
        {   // short, but within what the entries can give: evicted as before
            Sim s{{700}, {300, 300}, 0, {}};
            const auto r = run(s, 900, 200, 8, 1100, 600);
            check(r.fits && r.evicted == 2, "a deficit the entries can cover is evicted normally");
        }
        {   // nothing left that may go: freed 0 ends the loop
            Sim s{{100}, {}, 0, {}};
            const auto r = run(s, 900, 200, 8, 0, UINT64_MAX);
            check(!r.fits && r.evicted == 0 && !r.no_progress, "an empty cache ends without the no-progress stop");
        }
        {   // max_evictions bounds the pass
            Sim s{{100}, {300, 300, 300}, 0, {}};
            const auto r = run(s, 900, 200, 2, 0, UINT64_MAX);
            check(!r.fits && r.evicted == 2, "the eviction count is bounded by max_evictions");
        }
    }
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total physical memory, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX physical{};
    physical.dwLength = sizeof physical;
    check(GlobalMemoryStatusEx(&physical) != 0, "Windows physical-memory API is available");
    check(available.has_value() && *available <= physical.ullTotalPhys,
          "Windows provider returns a known, physically bounded sample");
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
