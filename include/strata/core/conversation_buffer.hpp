#pragma once

#include "strata/core/lossless.hpp"
#include "strata/core/progress.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // T35 A5 (STRATA_CONV_COMPRESS=1): replace the payload with a lossless packed stream, for the parked-state tier
    // only (a parked image is never read by compute before it is restored).  The logical size is unchanged, so every
    // size check and every reader is unchanged; the first visit/read after this decompresses transparently.  The
    // payload is round-tripped and compared before it is accepted, and a mismatch leaves the buffer untouched.
    // False when already packed, empty, or not worth it (no gain / the round trip did not match).
    bool pack() {
        if (packed_ || size_ == 0) return false;
        std::vector<uint8_t> flat(size_);
        if (!read(flat.data(), 0, size_)) return false;
        std::vector<uint8_t> packed;
        if (!strata::core::lossless::pack(flat.data(), flat.size(), packed)) return false;
        if (packed.size() >= flat.size()) return false;   // no gain: keep today's representation
        std::vector<uint8_t> check;
        const bool ok = strata::core::lossless::unpack(packed.data(), packed.size(), check) &&
                        check.size() == flat.size() && std::memcmp(check.data(), flat.data(), flat.size()) == 0;
        static std::atomic<uint64_t> verified{0}, rejected{0};
        if (!ok) {
            if (rejected.fetch_add(1, std::memory_order_relaxed) == 0)
                std::fprintf(stderr, "strata: conversation cache: STRATA_CONV_COMPRESS: a round trip did NOT match; "
                                     "the buffer stays uncompressed (a bad stream is never restored)\n");
            return false;
        }
        const uint64_t nth = verified.fetch_add(1, std::memory_order_relaxed);
        if (nth == 0)
            std::fprintf(stderr, "strata: conversation cache: STRATA_CONV_COMPRESS: first packed buffer %zu -> %zu "
                                 "bytes (%.2fx), round trip bit-exact\n", flat.size(), packed.size(),
                         (double) flat.size() / (double) std::max<size_t>(packed.size(), 1));
        segments_.clear();
        segments_.push_back(std::move(packed));
        packed_ = true;
        return true;
    }
    bool packed() const { return packed_; }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (packed_ && !ensure_unpacked()) {   // a packed buffer is never grown: unpack first (corruption: empty)
            segments_.clear();
            size_ = 0;
            return;
        }
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing, held));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        if (!ensure_unpacked() || !other.ensure_unpacked()) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    using Segment = std::vector<uint8_t>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    // T35 A5: a packed payload is decompressed on first use and stays uncompressed for the readers that follow (the
    // restore paths).  False (and the bytes are never handed out) when the stream does not decode.
    bool ensure_unpacked() const {
        if (!packed_) return true;
        std::vector<uint8_t> flat;
        if (segments_.size() != 1 ||
            !strata::core::lossless::unpack(segments_.front().data(), segments_.front().size(), flat) ||
            flat.size() != size_) {
            std::fprintf(stderr, "strata: conversation cache: a packed buffer did not unpack (%zu bytes expected)\n",
                         size_);
            return false;
        }
        segments_.clear();
        segments_.push_back(std::move(flat));
        packed_ = false;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (!self.ensure_unpacked()) return false;
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    mutable std::vector<Segment> segments_;
    mutable bool packed_ = false;   // T35 A5: segments_ holds one packed stream instead of the payload
    size_t size_ = 0;               // the logical payload size, packed or not
};

} // namespace strata::core
