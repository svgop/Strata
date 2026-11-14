// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.63).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/mm3/mm3_vq.cuh - the Baekpica VQ release's routed experts and matrices on the device (the pack of record on
// this device: data/models/VQ/MiniMax-M3.1-Flash, FORMAT.md's `minimax-m3-vq8`):
//
//   a matrix is a GVQ1 payload (128-byte header + F16 row gains + a 13-bit index stream + a high1 bit plane) over an
//   8-wide E4M3FN codebook (K 8192), and on the R13+8 layers a GVR8 residual payload (8-bit indices, K 256) shares
//   the row gain:
//     W[r, 8v + d] = f32(gain[r]) * (f32(C1[i1[r][v]][d]) [+ f32(C2[i2[r][v]][d])])
//   - 1.625 bpw of lane traffic (13 bits an 8-weight vector), the codebook-gather on the GPU; the sum of the two
//     E4M3 values is exact in F32 (the format's own arithmetic law).
//   - the low12 stream packs adjacent indices a,b as [a&255, ((a>>8)&15)|((b&15)<<4), (b>>4)&255] in row-major
//     vector order; bit 12 of index j is bit j%8 of byte j/8 of the high1 plane, LSB first.
//
// They stand behind the same interface as the EXL3 (fmt 0) and GGUF (fmt 4, kFmtGG) weights - mm3_exl3.hpp - as
// fmt 5 (kFmtVQ): a Mat's raw is the GVQ1 payload (its own 128-byte header carries the geometry; the fields vq_cb /
// vq_res / vq_cb2 point at the codebooks and the GVR8 payload, null when the matrix has no residual), and an
// ExpertLayout's blob is the pool's per-expert copy
//     [cb 65536 B][cb2 2048 B (R13+8)][w1][w3][w2][r1][r3][r2]   (each its GVQ1/GVR8 payload whole)
// with trellis[r] the GVQ1 offsets, res[r] the GVR8 ones (0: none) - the same staging-slot / promote machinery the
// GGUF experts ride, the layer's codebooks uploaded once (resident, 64 + 2 KB a layer).
//
// Activation law: the routed experts round the activations as the GGUF lane does (ggml's Q8_K, a scale a 256 - the
// CPU lane and the GPU give the same sums; STRATA_MM3_IQ_Q8K=0: a scale a 32), gate and up fused with MiniMax's
// swigluoai; the prompt path decodes FP16 rows and runs the same cuBLAS F32-accumulating GEMM the GGUF path runs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "mm3_exl3.hpp"

namespace mm3::vq {

namespace x3 = strata::mm3::exl3;

/// x3::Mat::fmt / x3::ExpertLayout::fmt of the Baekpica VQ weights.
constexpr int kFmtVQ = 5;

/// The payload magic a VQ matrix begins with (the reader's dispatch is ON THESE MAGIC BYTES, never on a GGUF type
/// id - the ecosystem repurposes type 42/43 three ways; see the NVFP4-mainline digest).
constexpr uint32_t kMagicGVQ1 = 0x31515647;   // "GVQ1" little endian
constexpr uint32_t kMagicGVR1 = 0x31525647;   // "GVR1" little endian

/// The 128-byte GVQ1/GVR8 header as the device reads it (FORMAT.md's wire layout; the strict-reader law - canonical
/// offsets, alignment, finite gains - is enforced by the host pass, read_gv_header in mm3_vq_host.hpp).
struct GvH {
    uint32_t rows, cols;          // R outputs, C VALUES a row (the wire's k; vectors a row = C / 8)
    __host__ __device__ __forceinline__ uint32_t vecs() const { return cols >> 3; }
    uint32_t entries;             // 8192 (13-bit) / 256 (GVR8)
    uint64_t gain_off, index_off, high_off, total;
};

/// Parse a payload's header into the fields the kernels use (magic + offsets only; the host pass validated the rest).
__host__ __device__ __forceinline__ GvH gv_header(const void* payload) {
    const uint8_t* p = (const uint8_t*) payload;
    GvH h;
    // '<4sHBBIIIIIIQQQQ32s32s': rows at 8, cols at 12, entries (K) at 16, gain at 32, index at 40, high at 48,
    // total at 56 - fixed offsets of the 128-byte wire header.
    h.rows = *(const uint32_t*) (p + 8);
    h.cols = *(const uint32_t*) (p + 12);
    h.entries = *(const uint32_t*) (p + 16);
    h.gain_off = *(const uint64_t*) (p + 32);
    h.index_off = *(const uint64_t*) (p + 40);
    h.high_off = *(const uint64_t*) (p + 48);
    h.total = *(const uint64_t*) (p + 56);
    return h;
}

/// The 13-bit index at flat stream position j (the packing law in the file comment).
__host__ __device__ __forceinline__ uint32_t idx13_flat(const uint8_t* low12, const uint8_t* high1, uint64_t j) {
    const uint64_t m = j >> 1;                       // the pair this index lives in
    const uint8_t* b = low12 + 3 * m;
    uint32_t i = (j & 1) == 0 ? (uint32_t) b[0] | ((uint32_t) (b[1] & 15) << 8)
                              : (uint32_t) (b[1] >> 4) | ((uint32_t) b[2] << 4);
    return i | (((uint32_t) high1[j >> 3] >> (j & 7)) & 1u) << 12;
}

/// E4M3FN decode (no infinities; the codebooks carry finite values only - the strict reader checked).
__host__ __device__ __forceinline__ float e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float mag = e == 0 ? (float) m * (1.0f / 512.0f)                     // (m/8) * 2^-6
                             : ldexpf((float) (8 + m) * (1.0f / 8.0f), e - 7); // (1+m/8) * 2^(e-7)
    return (b & 0x80) ? -mag : mag;
}

/// W[r, 8v + d] into out[8] (gain applied by the caller - a row's gain multiplies its every vector): the
/// codebook row C1[i1], plus the residual codebook's C2[i2] when the matrix has one.
__host__ __device__ __forceinline__ void w_vec(const uint8_t* low12, const uint8_t* high1, const uint8_t* idx8,
                                      const uint8_t* cb, const uint8_t* cb2, uint64_t j, float out[8]) {
    const uint32_t i1 = idx13_flat(low12, high1, j);
    const uint8_t* c1 = cb + 8 * (size_t) i1;
    #pragma unroll
    for (int d = 0; d < 8; ++d) out[d] = e4m3(c1[d]);
    if (idx8 != nullptr) {                          // the GVR8 residual: one plain index byte a vector
        const uint8_t* c2 = cb2 + 8 * (size_t) idx8[j];
        #pragma unroll
        for (int d = 0; d < 8; ++d) out[d] += e4m3(c2[d]);
    }
}

// ------------------------------------------------------------------ host interface (mirrors gg's)

/// y[r] = alpha * (x W)[r] (+ bias): a dense GEMV over a VQ matrix, the activations F32 (ncol activations as
/// MvJob takes; each column's sum its own).  Covered by a block a row-tile walking the vectors.
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s);

/// Decode [row0, row0 + rows) of m to FP16 rows (dst[r - row0][k], ldx apart): the prompt path's bands.
void f16_rows(const x3::Mat& m, int row0, int rows, uint16_t* dst, int ldx, cudaStream_t s);
/// Y[r] = beta Y[r] + (X[r] W): the same cuBLAS F32-accumulating FP16 GEMM the GGUF path runs, over the decoded
/// bands (a few column bands at a time on large matrices).
void gemm_rows(const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
               cudaStream_t s);

/// The routed experts of one token / a verify window / a prompt set - the gg contracts (mm3_exl3.hpp's MoeArgs,
/// WinArgs, SetArgs), swigluoai fused.  The blob layout is the file comment's.
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);
bool win_experts(const x3::WinArgs& a, cudaStream_t s);
void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s);
void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);

/// The per-stream scratch (the activations' Q8_K rows, the GEMMs' FP16 bands + cuBLAS handle): allocated at the
/// load like gg's, so the plan counts it.
bool prepare(cudaStream_t s);
size_t scratch_bytes();
int launch_errors();
unsigned wait_timeouts();

/// (diagnostics) the layer id the next win/moe call serves (window_post tags it; the content law prints it)
void trace_layer(int li);
/// (tests) rows of k values rounded as the routed experts' activations (gg's act_quant contract - shared with the
/// GGUF lane so a CPU lane gives the same sums).
void act_quant(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s);

}  // namespace mm3::vq
