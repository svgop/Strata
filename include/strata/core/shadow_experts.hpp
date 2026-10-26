// include/strata/core/shadow_experts.hpp - F-359: STRATA_SHADOW_EXPERTS, the shadow Q2_0 expert store.
//
// The CPU expert pool's misses stream the native pack's blobs (IQ3_S here: ~2,046,400 B a blob, ~150 us an
// entry, byte-bound).  The same model's Q2_0 blob store (tools/strata_pack.py's canonical experts.bin form)
// holds 1,382,400 B a blob and streams at ~75 us an entry.  The shadow store is that Q2_0 file mapped
// beside the running pack: the DECODE'S CPU MISSES compute from it, everything else is untouched.
//
// The shadow's blob form is the engine's compile-time canonical one (expert.hpp: BLOB = 1,382,400, the
// gate/up codes at 0, down codes at 819200, gate/up scales at 1228800, down scales at 1331200) - the same
// form FileExpertSource hands the canonical (non-native) path, decoded by ExpertPool::run_split_multi's
// row phases with Strata's own ActQ activations.  No kernel takes a second layout: the canonical kernel
// already IS the second layout.
//
// OFF BY DEFAULT.  With STRATA_SHADOW_EXPERTS unset or empty there is no mapping and no arm, and the
// dispatch's one armed() load reads false on every path it gates, so the engine's outputs are byte-identical.
#pragma once

#include <cstdint>

namespace strata::core::shadow {

/// Reads STRATA_SHADOW_EXPERTS=<dir> once.  Unset or empty: returns false, maps nothing, prints nothing.
/// Set: maps `<dir>/experts.bin`, validates `<dir>/experts.json` against the canonical layout (the same
/// geometry the running pack was loaded with: `n_layers` layers, 512 blobs a layer, 1,382,400 B a blob,
/// `layer offset == layer * n_expert * blob_bytes`), prints the one boot banner, and returns true.
/// A refusal (missing or mismatched file, a Hadamard-folded pack, whose experts the canonical Q2_0 form
/// cannot carry) prints one line and leaves the engine unshadowed - the native store serves every miss.
bool arm(int64_t n_layers, int64_t n_expert);

/// Whether arm() mapped a store.  One relaxed load; the dispatch checks it per window.
bool armed();

/// One shadow blob: `base + layer_offset(layer) + expert * BLOB`.  Null only when unarmed (never read
/// then) or on an out-of-range id, which the dispatch's null-blob failure path reports loudly.
const uint8_t* blob(int64_t layer, int64_t expert);

/// The armed store's geometry (0 when unarmed).
int64_t layers();
int64_t experts();
/// The mapping's size in bytes (0 when unarmed).
uint64_t mapped_bytes();
/// The directory the store came from ("" when unarmed) - for the banner and the summary line.
const char* dir();

}  // namespace strata::core::shadow
