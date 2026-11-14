// src/mm3/mm3_vq_host.hpp - the Baekpica VQ release's host-side contract: release.json + the GVQ1/GVR8
// matrix headers, spec-faithful to the pack's FORMAT.md (`minimax-m3-vq8`, format version 1/2).
//
// Phase 2 of the mm3 lane (.docs/research/digest/2026-11-14-mm3-import-map.md): the per-layer expert
// blobs this reader opens are the artifact of record on this device
// (data/models/VQ/MiniMax-M3.1-Flash/, SHA-256 verified 224/224) - unlike the Maya-format packs
// mm3_gguf.cu reads, these keep every expert at ~1.98 bpw as 13-bit codebook indices (GVQ1) with an
// 8-bit residual stage (GVR8) on the r13x57-r8x20 layers.
//
// THE READER DISPATCHES ON MAGIC BYTES, NEVER ON GGUF TYPE IDS.  The ecosystem repurposes type 42/43
// three ways (Q2_0 / ds4-vqblob / Synchopate; F8_E4M3 / fp4x32 / LUT-VQ3-64 - see the NVFP4-mainline
// digest); the payload's own magic is the only identity that holds.  Rejection follows the pack's own
// strict-reader law: noncanonical offsets, bad alignment, non-finite gains or scales, trailing bytes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mm3 { namespace vq {

// FORMAT.md: metadata in every part.  A reader that meets any other family key refuses.
inline constexpr const char* kArch = "minimax_m3_vq";
inline constexpr const char* kVariant = "minimax-m3-vq8";

// The GVQ1 matrix header (128 bytes, little endian; Python '<4sHBBIIIIIIQQQQ32s32s').
//   W[r, 8j+d] = float32(gain[r]) * float32(codebook[index[r, j]][d])          (13-bit indices)
// plus, on R13+8 parts, one GVR8 tensor per matrix:
//   W[r, 8j+d] = gain[r] * (C1[i1[r,j]][d] + C2[i2[r,j]][d])                    (shared row gain)
struct GvHeader {
    char magic[4];            // "GVQ1" or "GVR1"
    uint16_t version;         // 1
    uint8_t index_bits;       // 13 (GVQ1, container allows 12) / 8 (GVR8)
    uint8_t vec_dim;          // 8
    uint32_t rows, cols;      // R x C vectors (a [3072,6144] w1 is 3072 rows x 768 vectors)
    uint32_t codebook_entries;// 8192 (13-bit) / 4096 (12-bit) / 256 (GVR8)
    uint32_t codebook_prec;   // 1 = E4M3FN
    uint32_t codebook_id;     // 4*layer (+1 for the residual codebook)
    uint32_t flags;           // 0
    uint64_t gain_off;        // 128 (F16, one per row)
    uint64_t index_off;       // align128(128 + 2*R)          - the low12 stream
    uint64_t high_off;        // align128(low12 + R*C*3/16)   - the high1 plane; 0 when index_bits<13
    uint64_t total_bytes;     // align128 of the last region's end
    uint8_t cb_sha[32];       // SHA-256 of the linked codebook bytes
    uint8_t body_sha[32];     // SHA-256 of every byte after this header
};
static_assert(sizeof(GvHeader) == 128, "the GVQ1/GVR8 header is 128 bytes on the wire");

// One tensor of the tensor-manifest.json (the pack's per-tensor table; payload_sha256 over the
// matrix payload, source_* only on native tensors).
struct ManifestTensor {
    std::string name;         // original HF tensor names are kept
    std::string part;
    std::vector<uint64_t> stored_shape;
    uint32_t type_id;         // information only - identity is the payload magic
    uint64_t payload_bytes;
    std::string payload_sha256;
};

// release.json's parts table (the entry point of the release; independently readable files, never
// concatenated).  Roles: trunk (the layer-NN-experts.gguf blobs + body-nonexpert), embedding,
// vision, drafter.  container_version 2 + residual true marks the R13+8 parts (20 of the 57).
struct Part {
    std::string path;
    std::string role;         // trunk | embedding | vision | drafter
    std::string residency;    // resident | ssd-rows | mmap-on-demand
    bool required = false;
    int layer = -1;           // expert parts only
    std::string variant;      // "R13" (with "R8" residuals the parts carry residual=true)
    int container_version = 1;
    int index_bits = 13;
    bool residual = false;
};

struct Release {
    std::string format;           // "minimax-m3-vq8"
    int format_version = 1;
    std::string recipe_id;
    std::string source_revision;
    std::vector<Part> parts;
    std::vector<int> r13p8_layers; // the 20 layers whose parts are container_version 2
};

// Parse release.json (refuses unknown format/variant).  Loads the part table only; the files
// themselves are opened by the loader pass.
bool parse_release(const std::string& json_path, Release& out, std::string& err);

// Read + validate one GVQ1/GVR8 header at `payload` (mmap'd matrix start): magic, canonical offsets,
// alignment, total_bytes against the manifest's payload_bytes.  No allocation; the streams stay in
// the mapping and the device kernels walk them in place.
bool read_gv_header(const uint8_t* payload, uint64_t payload_bytes, GvHeader& out, std::string& err);

// The 13-bit index stream: for adjacent indices a, b the low12 plane holds
// [a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255]; the high1 plane carries bit 12 of
// index j as bit (j % 8) of byte (j / 8), LSB first, row-major vector order (GVR8: one plain byte
// per vector).  Kept as inline device helpers in mm3_vq.cu; declared here for the host self-check.
uint32_t gv_index(const GvHeader& h, const uint8_t* low12, const uint8_t* high1, uint64_t r, uint64_t v);

}}  // namespace mm3::vq
