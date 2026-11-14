// src/mm3/mm3_nvfp4.cuh - NVFP4 expert weights on the device for maya-mm3 (MiniMax-M3.1-Flash): NVIDIA's 4-bit
// format class behind the same interface as mm3_gguf.cuh's GGUF weights (an x3::Mat / x3::ExpertLayout with fmt
// kFmtNVFP4; mm3_exl3.cu's entry points hand them here).
//
// The format is NVIDIA's NVFP4 (type 40 upstream, llama.cpp MIT #19769; vLLM's nvfp4-pack-quantized; the ecosystem
// facts and the type-ID hazard census live in .docs/research/digest/2026-11-14-nvfp4-mainline.md): a weight is an
// E2M1 value a 4 bits, one E4M3 scale a 16 values, and one F32 global scale a matrix.  The decode math is ported
// from our vendored llama.cpp CUDA kernels (third_party/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:331-368,
// vec_dot_nvfp4_q8_1): the E2M1 codebook as integers doubled (ggml-common.h's kvalues_fp4: 0, 1, 2, 3, 4, 6, 8, 12
// and the signs) and the scale byte as E4M3 halved (common.cuh's ue4m3_to_fp32) - the doubled half pairs to the
// exact product, and the integer dot of the halves against the int8 activations runs through dp4a exactly, one F32
// sum a 16-value sub-block.  A nibble n decodes to sign(n >> 3) * {0, .5, 1, 1.5, 2, 3, 4, 6}[n & 7] and a scale
// byte b (bit 7 unused; 0x7F and 0xFF are the E4M3 NaNs, decoded as 0) to E4M3FN bits 0..6: exp (b >> 3) & 15,
// man b & 7, exp 0: man * 2^-9, else (1 + man / 8) * 2^(exp - 7).  The blob's F32 is the decode multiplier (the
// ecosystem's per-expert 1/global_scale class: llama.cpp PR #31192's `.scale` = 1.0 / weight_global_scale, applied
// on decode).  Dispatch keys on Mat::fmt / ExpertLayout::fmt (6) set by our loader - never on the ggml type id
// (the digest's hazard law: the id is a manifest claim, the payload is the truth).
//
// The EXPERT BLOB (and a dense Mat) - per matrix [n][k], k % 16 == 0, two planes after the header:
//   +0                   F32 gscale        the decode multiplier (see above)
//   +4                   nibbles [n][k/2]  uint8, row r at +4 + r*(k/2); value c in nibble c of the row
//                                          (byte c >> 1, low nibble when c is even - ggml's Q4 order)
//   +4 + n*(k/2)         scales [n][k/16]  uint8, row r at that offset + r*(k/16); the scale of values [16b, 16b+16)
// The whole matrix only (ld != 0 is refused): an expert blob is the three matrices gate | up | down back to back
// (ExpertLayout::trellis[r] = role r's byte offset from the blob base, multiple of 4; gg_type unused).  Every weight
// a row reads: y[r] = alpha * gscale * (dot of the decoded row r with the rounded activations) + bias[r].
//
// Activations as mm3_gguf.cu's (the same rounds on both sides of the CPU lane): the dense GEMVs (mv) rounded int8
// a 32 values (gg::q8_quant), the routed experts as the CPU lane rounds them (gg::act_quant: ggml's Q8_K when
// k % 256 == 0, else a scale a 32); the experts' gate and up then MiniMax's swigluoai fused (x3::kSwigluAlpha, the
// limit clamp), down over it, and the weighted sum in plan order with the CPU lane's part and the shared expert's -
// the EXL3 kernels' combine.  The prompt path's GEMMs (gemm_rows): the weight decoded to FP16 rows and the same
// cuBLAS FP16 GEMM with F32 accumulation, a few column bands at a time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "mm3_exl3.hpp"

namespace mm3::nf {

namespace x3 = strata::mm3::exl3;

/// x3::Mat::fmt / x3::ExpertLayout::fmt of an NVFP4 weight (the loader's claim; the ggml type id 40 stays a
/// manifest tag: type_of below maps it, the dispatch never reads it).
constexpr int kFmtNVFP4 = 6;

/// The manifest dtype tag the pack writes, the ggml type id upstream claims for it (GGML_TYPE_NVFP4, 40 in the
/// vendored enum and upstream's - the digest's drift hazard lives on the id, our dispatch keys on fmt 6).
constexpr const char* kDtypeName = "NVFP4";
constexpr int kGgmlTypeNVFP4 = 40;

/// The manifest's dtype name -> its id ("NVFP4"; -1: not this format), the bytes of a matrix row of k values
/// (nibbles + scales; 0: k % 16 != 0), the values one scale covers.
inline int type_of(const std::string& dtype) { return dtype == kDtypeName ? kGgmlTypeNVFP4 : -1; }
inline size_t row_bytes(int64_t k) { return k % 16 == 0 ? (size_t) (k / 2) + (size_t) (k / 16) : 0; }
constexpr int block_values() { return 16; }

/// mm3_exl3.cu's entry points for NVFP4 weights (the same contracts: mm3_exl3.hpp).  A Mat's fmt 6 with
/// raw = the matrix blob above, k % 16 == 0, ld == 0 (whole matrices).
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s);
void gemm_rows(const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
               cudaStream_t s);
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);
bool win_experts(const x3::WinArgs& a, cudaStream_t s);
void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s);
void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);
/// The per-stream scratch (the activations' int8 rows, a token's down outputs and swigluoai activations, the GEMMs'
/// FP16 weight bands and their cuBLAS handle): allocated here at the load, so the plan counts it.  scratch_bytes():
/// what prepare() takes a stream.
bool prepare(cudaStream_t s);
size_t scratch_bytes();
/// Launch failures of this family since the process started (each is also printed); the device's win_experts wait
/// on the host's word that gave up after 120 s.
int launch_errors();
unsigned wait_timeouts();

}  // namespace mm3::nf
