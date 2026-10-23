// T35 A5: a small, self-contained lossless byte codec for the RESTORE-ONLY parked-state tier.
//
// The parked conversation images (--conversation-cache-mib) are never read by compute before they are restored, so
// compressing them cannot move an output bit: decompression reproduces the exact source bytes and the caller checks
// the stored FNV-1a hash (and, on first use, the whole payload) before the image is used.  This is the tier F-230
// cannot reach: no acceptance exposure, because the bytes handed back to the session are byte-identical.
//
// Format (self-describing, little-endian):
//   "SLC1" | u64 source_size | u64 source_hash | u8 method | payload
// method 0: the source bytes verbatim (when the LZ pass did not shrink them).
// method 1: an LZ4-class token stream: [token u8: hi nibble literal length (15 = extra varint), lo nibble
//   match length - 4 (15 = extra varint)][literals][if output < source_size: u32 distance][match bytes].  Matches
//   come from a single-probe 64K-entry hash table and may run any length; copies may overlap like LZ4's.
// Every read is bounds-checked; a malformed stream returns false (the caller keeps the image unuseable, never
// restores garbage).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::lossless {

inline uint64_t fnv1a64(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace detail {

constexpr uint8_t kMagic[4] = {'S', 'L', 'C', '1'};
constexpr size_t kHeader = 4 + 8 + 8 + 1;
constexpr int kHashBits = 16;
constexpr size_t kHashSize = 1u << kHashBits;
constexpr size_t kMinMatch = 4;

inline void put_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back((uint8_t) (v >> (8 * i)));
}
inline void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back((uint8_t) (v >> (8 * i)));
}
inline uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (8 * i);
    return v;
}
inline uint32_t get_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) p[i] << (8 * i);
    return v;
}
inline void put_varint(std::vector<uint8_t>& out, size_t v) {
    while (v >= 0x80) {
        out.push_back((uint8_t) (v | 0x80));
        v >>= 7;
    }
    out.push_back((uint8_t) v);
}
inline bool get_varint(const uint8_t*& p, const uint8_t* end, size_t& v) {
    v = 0;
    for (int shift = 0; shift <= 56; shift += 7) {
        if (p >= end) return false;
        const uint8_t b = *p++;
        v |= (size_t) (b & 0x7f) << shift;
        if ((b & 0x80) == 0) return true;
    }
    return false;
}
inline uint32_t hash4(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return (v * 2654435761u) >> (32 - kHashBits);
}
inline int token_len(size_t len, std::vector<uint8_t>& extra) {   // the 4-bit nibble, with the 15 escape
    if (len < 15) return (int) len;
    put_varint(extra, len - 15);
    return 15;
}

}  // namespace detail

/// Packs `src` into `out`.  Returns true when `out` holds a self-describing stream (always, unless n == 0, where out is
/// left empty).  The stream is either compressed or stored; `out.size()` is the caller's honest byte cost.
inline bool pack(const uint8_t* src, size_t n, std::vector<uint8_t>& out) {
    out.clear();
    if (n == 0) return false;
    std::vector<uint8_t> body;
    body.reserve(n / 2 + 64);
    std::vector<uint32_t> head(detail::kHashSize, 0xffffffffu);
    size_t pos = 0, anchor = 0;
    while (pos + detail::kMinMatch <= n) {
        const uint32_t h = detail::hash4(src + pos);
        const uint32_t cand = head[h];
        head[h] = (uint32_t) pos;
        if (cand != 0xffffffffu && pos > cand && pos - cand <= 0xffffffffu &&
            std::memcmp(src + cand, src + pos, detail::kMinMatch) == 0) {
            size_t match = detail::kMinMatch;
            while (pos + match < n && src[cand + match] == src[pos + match]) ++match;
            // LZ4 token order: token | literal-length escape | literals | match-length escape | distance
            const size_t lit = pos - anchor;
            std::vector<uint8_t> lext, mext;
            const int lit_nib = detail::token_len(lit, lext);
            const int match_nib = detail::token_len(match - detail::kMinMatch, mext);
            body.push_back((uint8_t) ((lit_nib << 4) | match_nib));
            body.insert(body.end(), lext.begin(), lext.end());
            body.insert(body.end(), src + anchor, src + pos);
            body.insert(body.end(), mext.begin(), mext.end());
            detail::put_u32(body, (uint32_t) (pos - cand));
            // the match itself is not written: the decoder copies from the distance
            pos += match;
            anchor = pos;
            if (pos + detail::kMinMatch <= n) {
                // index the first positions of the match too, so later matches can start inside it
                for (size_t i = pos - match + 1; i + detail::kMinMatch <= pos && i < pos; ++i)
                    head[detail::hash4(src + i)] = (uint32_t) i;
            }
        } else {
            ++pos;
        }
    }
    {   // the final literals (no match follows)
        const size_t lit = n - anchor;
        std::vector<uint8_t> extra;
        const int lit_nib = detail::token_len(lit, extra);
        body.push_back((uint8_t) (lit_nib << 4));
        body.insert(body.end(), extra.begin(), extra.end());
        body.insert(body.end(), src + anchor, src + n);
    }
    const bool compressed = body.size() + detail::kHeader < n;
    out.reserve(detail::kHeader + (compressed ? body.size() : n));
    out.insert(out.end(), detail::kMagic, detail::kMagic + 4);
    detail::put_u64(out, (uint64_t) n);
    detail::put_u64(out, fnv1a64(src, n));
    out.push_back(compressed ? 1 : 0);
    if (compressed) out.insert(out.end(), body.begin(), body.end());
    else out.insert(out.end(), src, src + n);
    return true;
}

/// Unpacks a stream `pack` produced.  False on a malformed stream or a truncated payload; `out` is then unspecified.
/// The caller checks the payload's size and hash (fnv1a64) against what it expects before using it.
inline bool unpack(const uint8_t* in, size_t n, std::vector<uint8_t>& out) {
    out.clear();
    if (in == nullptr || n < detail::kHeader || std::memcmp(in, detail::kMagic, 4) != 0) return false;
    const uint64_t want = detail::get_u64(in + 4);
    const uint8_t method = in[detail::kHeader - 1];
    const uint8_t* p = in + detail::kHeader;
    const uint8_t* const end = in + n;
    if (want > (uint64_t) (SIZE_MAX / 2) || want > (1ull << 40)) return false;
    out.reserve((size_t) want);
    if (method == 0) {
        if ((uint64_t) (end - p) != want) return false;
        out.insert(out.end(), p, end);
        return true;
    }
    if (method != 1) return false;
    while ((uint64_t) out.size() < want) {
        if (p >= end) return false;
        const uint8_t token = *p++;
        size_t lit = token >> 4;
        size_t match = token & 0x0f;
        if (lit == 15) {
            size_t extra = 0;
            if (!detail::get_varint(p, end, extra)) return false;
            lit += extra;
        }
        if (lit > (size_t) (end - p) || lit > (size_t) (want - out.size())) return false;
        out.insert(out.end(), p, p + lit);
        p += lit;
        if ((uint64_t) out.size() == want) break;
        if (match == 15) {
            size_t extra = 0;
            if (!detail::get_varint(p, end, extra)) return false;
            match += extra;
        }
        match += detail::kMinMatch;
        if ((size_t) (end - p) < 4) return false;
        const uint32_t distance = detail::get_u32(p);
        p += 4;
        if (distance == 0 || distance > out.size()) return false;
        if ((uint64_t) match > want - out.size()) return false;
        const size_t from = out.size() - distance;
        for (size_t i = 0; i < match; ++i) out.push_back(out[from + i]);   // overlapping copies are legal (LZ4)
    }
    return out.size() == want;
}

}  // namespace strata::core::lossless
