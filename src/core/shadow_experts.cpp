// src/core/shadow_experts.cpp - F-359: STRATA_SHADOW_EXPERTS, the shadow Q2_0 expert store.  See the header.
#include "strata/core/shadow_experts.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/hadamard.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#if !defined(MFD_HUGETLB)
#include <linux/memfd.h>
#endif
#if !defined(MFD_HUGE_2MB)
#define MFD_HUGE_2MB (21 << 26)
#endif
// 2026-11-09 bootstorm (F2): in-kernel prefault of the whole store mapping (kernel 5.14+; this box 7.2.12
// probed, 24.5 GiB/s over reserved-pool 2 MiB pages).  Older glibc does not spell the constant.
#if !defined(MADV_POPULATE_WRITE)
#define MADV_POPULATE_WRITE 20
#endif
#include "strata/platform/hugepage.hpp"
#endif

namespace strata::core::shadow {

namespace {
struct ShadowState {
    bool tried = false;                    // arm() ran once; the env is read a single time
    bool armed = false;
    const uint8_t* base = nullptr;
    int64_t n_layers = 0, n_expert = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> layer_off;       // per layer: where its 512 blobs start (the canonical form)
    std::string dir;
};
ShadowState g_shadow;

// The shadow layout's owner is <dir>/experts.json (tools/pack_layer.py's machine-written file).  Its shape
// is fixed, so the reader extracts the four numbers that describe the store and refuses everything else:
// "blob_bytes", "n_layers", and each layer entry's "layer" / "offset" pair.  Quoted-key matching is exact,
// so the top-level "offsets" object and a layer's "role_offsets" never match "offset".
bool json_u64(const std::string& s, size_t from, const char* key, uint64_t& out) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t at = s.find(pat, from);
    if (at == std::string::npos) return false;
    size_t i = at + pat.size();
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
    out = std::strtoull(s.c_str() + i, nullptr, 10);
    return true;
}

bool refuse(const std::string& why) {
    std::fprintf(stderr, "strata: STRATA_SHADOW_EXPERTS: %s - the native store serves every miss\n", why.c_str());
    return false;
}
}  // namespace

bool arm(int64_t n_layers, int64_t n_expert) {
    ShadowState& st = g_shadow;
    if (st.tried) return st.armed;
    st.tried = true;
    const char* env = std::getenv("STRATA_SHADOW_EXPERTS");
    if (env == nullptr || env[0] == '\0') return false;   // the gate: unset means no mapping, no banner, no branch
    st.dir = env;
    // A Hadamard-folded pack's experts are stored rotated; the canonical Q2_0 form cannot carry the signs
    // (the same refusal expert_layout_load gives a pack that folds a Q2_0 expert), so the shadow would
    // compute UNROTATED experts.  Never arm over one.
    if (strata::kernels::cpu::hadamard_spec().any())
        return refuse("this pack folds its experts (hadamard.txt), which the shadow's canonical Q2_0 form cannot carry");
    if (n_layers <= 0 || n_expert <= 0) return refuse("the model's expert geometry is empty");
    using strata::kernels::cpu::BLOB;

    // The shadow layout: read <dir>/experts.json and require it to BE the canonical layout - one blob size
    // (BLOB) for every layer and `layer offset == layer * n_expert * blob_bytes` with no padding.  That is
    // the layout the canonical kernels decode and the one FileExpertSource builds for a non-native pack
    // (expert_source.cpp's `!layout.native` branch); a store in any other form is refused, never guessed.
    std::ifstream in(st.dir + "/experts.json");
    if (!in) return refuse("cannot open " + st.dir + "/experts.json");
    std::string js((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    uint64_t blob_bytes = 0, file_layers = 0, total_bytes = 0;
    if (!json_u64(js, 0, "blob_bytes", blob_bytes) || !json_u64(js, 0, "n_layers", file_layers) ||
        !json_u64(js, 0, "total_bytes", total_bytes))
        return refuse(st.dir + "/experts.json is missing blob_bytes, n_layers or total_bytes");
    if (blob_bytes != (uint64_t) BLOB)
        return refuse(st.dir + "/experts.json's blob is " + std::to_string(blob_bytes) + " B, the canonical kernel decodes " +
                      std::to_string(BLOB) + " B (a different expert quantizer or geometry)");
    if (file_layers != (uint64_t) n_layers)
        return refuse(st.dir + "/experts.json holds " + std::to_string(file_layers) + " layers, the model has " +
                      std::to_string(n_layers));
    const uint64_t want = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
    if (total_bytes != want)
        return refuse(st.dir + "/experts.json's total_bytes is " + std::to_string(total_bytes) + ", the model's geometry is " +
                      std::to_string(want));
    // The canonical per-layer offset is pure arithmetic: total_bytes and blob_bytes above already pin the
    // store to layer * n_expert * BLOB with no padding, so the layer table is derived, not scanned (the
    // earlier JSON scan mis-keyed on the "layers" array's own name - F-360).
    std::vector<uint64_t> off((size_t) n_layers);
    for (int64_t l = 0; l < n_layers; ++l) off[(size_t) l] = (uint64_t) l * (uint64_t) n_expert * (uint64_t) BLOB;

    // The file itself, mapped whole: the blob pointer is base plus a multiply and the page fault that
    // follows is the read (the Phase-2 argument, expert_source.hpp).  The size check is not a formality:
    // a short file would fault at the END of a long sequence.
#if defined(_WIN32)
    const int wide = MultiByteToWideChar(CP_UTF8, 0, (st.dir + "/experts.bin").c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, (st.dir + "/experts.bin").c_str(), -1, wpath.data(), wide);
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return refuse("cannot open " + st.dir + "/experts.bin");
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) { CloseHandle(f); return refuse("cannot size " + st.dir + "/experts.bin"); }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf, "%s/experts.bin is %llu B, the canonical layout requires %llu B - it is not this "
                                       "model's Q2_0 store", st.dir.c_str(), (unsigned long long) sz.QuadPart,
                      (unsigned long long) want);
        CloseHandle(f);
        return refuse(buf);
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) { CloseHandle(f); return refuse("CreateFileMapping failed on " + st.dir + "/experts.bin"); }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) { CloseHandle(m); CloseHandle(f); return refuse("MapViewOfFile failed on " + st.dir + "/experts.bin"); }
    CloseHandle(f);   // the mapping holds the file
    st.base = (const uint8_t*) view;
#else
    const int fd = ::open((st.dir + "/experts.bin").c_str(), O_RDONLY);
    if (fd < 0) return refuse("cannot open " + st.dir + "/experts.bin");
    struct stat sb{};
    if (fstat(fd, &sb) != 0 || sb.st_size < 0 || (uint64_t) sb.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf, "%s/experts.bin is %lld B, the canonical layout requires %llu B - it is not this "
                                       "model's Q2_0 store", st.dir.c_str(),
                      (long long) (sb.st_size < 0 ? -1 : (long long) sb.st_size), (unsigned long long) want);
        ::close(fd);
        return refuse(buf);
    }
    // ENGINE-LEVEL 2 MiB backing (HOSTMEM-NO-REBOOT, engine-owned - no mount, no service, no root):
    // when the reserved pool covers the store whole, populate a hugetlb-backed memfd from this file
    // and serve the blobs from it - 2 MiB pages by construction, process-scoped, released with the
    // process.  ~15 GiB/s memcpy (the references-round2 measured class) makes the populate a ~2 s
    // boot step.  STRATA_SHADOW_MEMFD=0 keeps the plain file mapping; a short pool falls back to it
    // (the F-367 tmpfs tier, when present, is that same file path).
    void* view = MAP_FAILED;
    const char* mfd_note = "";
#if !defined(_WIN32) && defined(MFD_HUGETLB)
    const char* mfd_off = std::getenv("STRATA_SHADOW_MEMFD");
    // F4 (2026-11-09 bootstorm): ATTEMPT the memfd arm first and let the mapping's own reservation be
    // the pool check.  Probed on this box (kernel 7.2.12): ftruncate on a memfd MFD_HUGETLB file does
    // NOT reserve pages (Rsvd stayed 0); the MAP_SHARED mmap RESERVES the whole store atomically
    // (HugePages_Rsvd moved by exactly the store's page count, all-or-nothing, -ENOMEM when the pool
    // cannot cover).  The earlier huge_pool_covers() pre-check raced the pool's other drawers and its
    // all-or-nothing refusal is what landed the store on the swappable 4 KiB tmpfs tier in the 11:05
    // squeeze (log:667335) - the store now reserves its pages the moment they exist, ahead of the
    // complement's draw and every later organ.
    if (mfd_off == nullptr || mfd_off[0] != '0') {
        const auto t0 = std::chrono::steady_clock::now();
        int mfd = ::memfd_create("strata-shadow-q2_0", MFD_HUGETLB | MFD_HUGE_2MB);
        if (mfd >= 0) {
            const size_t hp = 2ull << 20;
            const size_t rounded = ((size_t) want + hp - 1) / hp * hp;   // hugetlb sizes are whole pages
            if (::ftruncate(mfd, (off_t) rounded) == 0) {
                void* src = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
                void* dst = src != MAP_FAILED
                                ? mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0)
                                : MAP_FAILED;
                if (src != MAP_FAILED && dst != MAP_FAILED) {
                    uint8_t* d = (uint8_t*) dst;
                    const uint8_t* s = (const uint8_t*) src;
                    // F2 (2026-11-09 bootstorm): fault the whole store in ONE kernel pass first
                    // (MADV_POPULATE_WRITE, kernel 5.14+; probed on this box: 4 GiB of pool pages in
                    // 0.163 s, 24.5 GiB/s), then copy fault-free; on EINVAL (kernel < 5.14) the chunked
                    // loop below stays the populate and faults in the copy as before.
                    if (::madvise(dst, rounded, MADV_POPULATE_WRITE) == 0) {
                        std::memcpy(d, s, (size_t) want);
                    } else {
                        constexpr size_t kChunk = 256ull << 20;
                        for (size_t o = 0; o < (size_t) want; o += kChunk) {
                            const size_t n = std::min(kChunk, (size_t) want - o);
                            std::memcpy(d + o, s + o, n);
                        }
                    }
                    ::munmap(src, (size_t) want);
                    ::mprotect(dst, rounded, PROT_READ);   // the store is read-only from here on
                    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    char buf[128];
                    std::snprintf(buf, sizeof buf, "; 2 MiB pool pages (memfd, reserved and populated in %.1f s)", secs);
                    mfd_note = buf;
                    view = dst;
                } else {
                    if (src != MAP_FAILED) ::munmap(src, (size_t) want);
                    if (dst != MAP_FAILED) ::munmap(dst, rounded);
                    char buf[192];
                    std::snprintf(buf, sizeof buf,
                                  "; the pool could not reserve the store (%.1f GiB free of %.1f GiB needed) - "
                                  "the swappable tmpfs tier serves every miss",
                                  (double) strata::platform::huge_pool_free_bytes() / 1073741824.0,
                                  (double) rounded / 1073741824.0);
                    mfd_note = buf;
                }
            } else {
                mfd_note = "; memfd ftruncate refused - file mapping";
            }
            ::close(mfd);   // the MAP_SHARED view holds the memfd, as it does a file
        } else {
            mfd_note = "; memfd_create refused - file mapping";
        }
    }
#endif
    if (view == MAP_FAILED) {
        view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
        if (view == MAP_FAILED) { ::close(fd); return refuse("mmap failed on " + st.dir + "/experts.bin"); }
    }
    ::close(fd);   // MAP_SHARED keeps the mapping valid after the fd closes
    st.base = (const uint8_t*) view;
#endif
    st.armed = true;
    st.n_layers = n_layers;
    st.n_expert = n_expert;
    st.bytes = want;
    st.layer_off = std::move(off);
    std::fprintf(stderr,
                 "strata: STRATA_SHADOW_EXPERTS=%s: shadow Q2_0 expert store mapped - %lld layers x %lld experts x "
                 "%llu B = %.2f GiB%s; decode CPU misses compute from it (hits, the PCIe share and the prompt path "
                 "do not change)\n",
                 st.dir.c_str(), (long long) n_layers, (long long) n_expert, (unsigned long long) BLOB,
                 (double) want / (double) (1ull << 30), mfd_note);
    return true;
}

bool armed() { return g_shadow.armed; }

const uint8_t* blob(int64_t layer, int64_t expert) {
    const ShadowState& st = g_shadow;
    if (!st.armed || layer < 0 || layer >= st.n_layers || expert < 0 || expert >= st.n_expert) return nullptr;
    return st.base + st.layer_off[(size_t) layer] + (uint64_t) expert * (uint64_t) strata::kernels::cpu::BLOB;
}

int64_t layers() { return g_shadow.armed ? g_shadow.n_layers : 0; }
int64_t experts() { return g_shadow.armed ? g_shadow.n_expert : 0; }
uint64_t mapped_bytes() { return g_shadow.armed ? g_shadow.bytes : 0; }
const char* dir() { return g_shadow.armed ? g_shadow.dir.c_str() : ""; }

}  // namespace strata::core::shadow
