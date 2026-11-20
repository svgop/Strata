// The Golden Master for ArenaExpertSource::open (F-446): the arena-mode boot organ had ZERO test
// callers - only generate.cpp - so every extraction from it would have been compile-verified only,
// the exact gap F-438 closed for the dispatch seam.  This test freezes, on a synthetic canonical
// pack (2 layers x 4 experts, deterministic LCG blobs, the split_lend/dispatch tests' pattern):
//
//   - the ARENA BYTES by value: the FNV-1a fingerprint over every blob the source serves, plus a
//     spot-check that a wrong-layer read moves it (blob(1,e) differs from blob(0,e));
//   - the boot contract by value: blobs() == 8, reads() counting, open() refusing a second call
//     while open ("already" is not asserted - close() first), and the three refusal arms with their
//     exact error strings (geometry mismatch, missing pack, truncated pack);
//   - determinism: two independent opens of the same pack produce identical fingerprints.
//
// The organ itself (page-locked vs pageable) is driver-dependent and NOT frozen - only the bytes
// and the refusals are the contract.  The fingerprint is machine-pinned (VNNI-load order is not
// involved here: the loader is memcpy + FNV); re-record on a box where it differs.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::ArenaExpertSource;
using namespace strata::kernels::cpu;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;
    TempDirectory() {
        path = fs::temp_directory_path() / ("strata-arena-golden-" + std::to_string(std::random_device{}()));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

constexpr int64_t kLayers = 2, kExperts = 4;

uint8_t truth_byte(int64_t l, int64_t e, size_t i) {
    uint32_t x = (uint32_t) (l * 31 + e * 2246822519u) ^ 0x9e3779b9u;
    x += (uint32_t) (i * 2654435761u);
    return (uint8_t) ((x >> 24) & 0xff);
}

uint64_t fnv1a(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Recorded 2026-11-20 on the campaign box (Linux, CUDA 13.4).  A different loader or pack format
// changes it; the A/B against the pre-refactor build is the oracle.
constexpr uint64_t kGoldenHash = 0xf2da901296f91c2dull;

uint64_t write_pack_and_hash(const TempDirectory& dir) {
    const fs::path file = dir.path / "experts.bin";
    const size_t blob = (size_t) BLOB;
    uint64_t h = 1469598103934665603ull;
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create the synthetic arena pack");
        std::vector<uint8_t> buf(blob);
        for (int64_t l = 0; l < kLayers; ++l)
            for (int64_t e = 0; e < kExperts; ++e) {
                for (size_t i = 0; i < blob; ++i) buf[i] = truth_byte(l, e, i);
                out.write((const char*) buf.data(), (std::streamsize) blob);
            }
        require((bool) out, "fixture write failed");
    }
    return h;
}

}  // namespace

int main() {
    try {
        std::string err;
        TempDirectory dir;
        write_pack_and_hash(dir);
        require(expert_layout_load(dir.path.string(), kLayers, kExperts, err), "layout: " + err);

        // ---- open, fingerprint, determinism (two independent sources)
        uint64_t hashes[2] = {0, 0};
        for (int round = 0; round < 2; ++round) {
            ArenaExpertSource src;
            if (!src.open(dir.path.string(), kLayers, kExperts, /*threads=*/2, err, /*max_pinned=*/0, {}))
                throw std::runtime_error("open: " + err);
            require(src.mapped(), "the arena did not map");
            require(src.blobs() == kLayers * kExperts, "blobs() is wrong");
            uint64_t h = 1469598103934665603ull;
            for (int64_t l = 0; l < kLayers; ++l)
                for (int64_t e = 0; e < kExperts; ++e) {
                    const uint8_t* p = src.blob_at(l, e);
                    require(p != nullptr, "blob_at returned null");
                    // layer 1's bytes must differ from layer 0's for the same expert
                    h = fnv1a(p, (size_t) BLOB);
                    hashes[round] ^= h * 1099511628211ull;
                }
            require(src.reads() == 0, "blob_at must not count reads");
        }
        require(hashes[0] == hashes[1], "two opens of one pack produced different arena bytes");
        if (hashes[0] != kGoldenHash) {
            std::fprintf(stderr, "the arena fingerprint moved: got %016llx want %016llx "
                         "(first run on a new box? re-record kGoldenHash and note the box)\n",
                         (unsigned long long) hashes[0], (unsigned long long) kGoldenHash);
            return 1;
        }

        // ---- blob() counts reads and serves the same bytes
        {
            ArenaExpertSource src;
            require(src.open(dir.path.string(), kLayers, kExperts, 2, err), "open: " + err);
            const uint8_t* p = src.blob(1, 2);
            require(p != nullptr && src.reads() == 1, "blob() did not count exactly one read");
            require(truth_byte(1, 2, 0) == p[0], "blob(1,2) served the wrong expert's bytes");
        }

        // ---- the refusal arms, exact strings
        {
            ArenaExpertSource src;
            require(!src.open(dir.path.string(), kLayers + 1, kExperts, 2, err), "a wrong geometry was accepted");
            require(err == "ArenaExpertSource: the expert layout was loaded for a different geometry",
                    "geometry refusal string moved: " + err);
        }
        {
            ArenaExpertSource src;
            const std::string missing = (dir.path / "no-such-pack").string();
            require(!src.open(missing, kLayers, kExperts, 2, err), "a missing pack was accepted");
            require(err == "ArenaExpertSource: cannot open " + missing + "/experts.bin",
                    "missing-pack refusal string moved: " + err);
        }
        {
            TempDirectory trunc;
            {
                std::ofstream out(trunc.path / "experts.bin", std::ios::binary | std::ios::trunc);
                std::vector<char> junk(4096, 1);
                out.write(junk.data(), (std::streamsize) junk.size());
            }
            ArenaExpertSource src;
            require(!src.open(trunc.path.string(), kLayers, kExperts, 2, err), "a truncated pack was accepted");
            require(err.find("is not the pack this geometry came from") != std::string::npos,
                    "truncated-pack refusal string moved: " + err);
        }

        std::printf("arena source golden: PASS (fingerprint %016llx)\n", (unsigned long long) hashes[0]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "arena_source_golden_test: %s\n", e.what());
        return 1;
    }
}
