// bench/micro/moe_hit_sat.cu - H1's instrument (campaign/research/perf-hunt-2026-11-20.md, law 4:
// the cheapest falsifier first; law 11: instrument before optimize).
//
// A STANDALONE SATURATION MICROBENCH of the doorbell hit path's two weight kernels, included from
// the engine source itself (never a copy - the owner law): it measures the tree's actual
// `gu_pair_kernel` / `down_pair_kernel`, launched exactly as `launch_hit_gu` / `launch_hit_down`
// launch them (THREADS=256, one warp per row pair), with no doorbell, no window machinery, no
// CUDA graph - only the kernels and a timer.
//
// WHAT IT DECIDES (the perf-hunt decision table):
//   - saturated GB/s >= 600            -> the kernels are fine; the live path is issue-rate/wait
//                                         bound -> suspect #1 demoted, attack the doorbell shape.
//   - saturated GB/s ~ 185 (the live    -> the kernels are the wall; the layout/kernel arms price
//     derived figure)                     in (pack-time layout conversion first, law 9 order).
//   - low at live n_hits, high when     -> OCCUPANCY: the launch shape (warps in flight) starves
//     saturated                           HBM - the lever is batching/shape, not the kernel body.
//
// MODES, because the live footprint is under the L2: per launch the live path touches
// n_hits x 1.38 MB (a ~55 MB window at n_hits=40) on a part whose L2 is larger than that, so a
// single-mode bench would flatter (or slander) the kernel by cache accident:
//   ROTATE - every timed iteration reads FRESH slots (slot set advances through the blob array);
//            the footprint outruns L2, so the number is DRAM-honest.
//   HOT    - every iteration re-reads the SAME slots; the L2-resident ceiling (the live per-layer
//            regime's best case).
// ANCHOR - a grid-stride uint4 streaming read over the same allocation: the machine's own
// achievable number for this memory, measured, not the datasheet's.
//
// Usage:  /usr/local/cuda-13.4/bin/nvcc -O3 -arch=sm_120 -I include -I . \
//             bench/micro/moe_hit_sat.cu -o /tmp/moe_hit_sat && /tmp/moe_hit_sat
// Output: one SAT line per (n_hits, mode) point - GB/s of WEIGHT bytes (codes+scales; the
// activations are ~2% and are printed as touched-bytes for completeness), then the verdict inputs.

#include "../../src/kernels/cuda/s2_expert_grouped.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace {

constexpr int N_SLOTS = 2048;    // 2048 x 1.3824 MB = 2.83 GB of expert blobs, >> L2
constexpr int WARMUP = 4;
constexpr int ITERS = 64;

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

/// The ANCHOR: grid-stride uint4 read of the whole span, one add per vector, dummy-reduced so the
/// compiler cannot elide it. This is the part's achievable stream, not the datasheet.
__global__ void stream_read_kernel(const uint4* p, size_t n_vec, float* sink) {
    float acc = 0.f;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc += (float) (v.x ^ v.y ^ v.z ^ v.w);
    }
    if (acc == 1.e30f) *sink = acc;   // never true; keeps the reads live
}

struct Buffers {
    uint8_t* blobs = nullptr;       // N_SLOTS x blob_bytes, 256-B aligned
    int32_t* slot_index = nullptr;  // ROTATE: ITERS x max_hits, pre-shuffled
    int32_t* dst_index = nullptr;   // identity scatter: hit h -> out row h
    uint8_t* x_q8_0 = nullptr;      // one token's Q8_0 activation (H/32 blocks x 34 B)
    float* x_scales = nullptr;      // H/32 fp32 activation scales (R4.2h contract)
    uint8_t* h_q8_0 = nullptr;      // the down input intermediate, Q8_0 (max_hits x FF/32 x 34)
    float* h_scales = nullptr;      // max_hits x FF/32 fp32
    float* gate_up = nullptr;       // gu scratch: max_hits x 2 x FF fp32
    float* out = nullptr;           // down output: max_hits x H fp32
    float* sink = nullptr;
};

double time_kernel(const std::function<void(int)>& launch, int iters) {
    cudaEvent_t a, b;
    cuda_check(cudaEventCreate(&a), "event a");
    cuda_check(cudaEventCreate(&b), "event b");
    for (int i = 0; i < WARMUP; ++i) launch(i);
    cuda_check(cudaEventRecord(a), "record a");
    for (int i = 0; i < iters; ++i) launch(i);
    cuda_check(cudaEventRecord(b), "record b");
    cuda_check(cudaEventSynchronize(b), "sync b");
    float ms = 0.f;
    cuda_check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cuda_check(cudaEventDestroy(a), "destroy a");
    cuda_check(cudaEventDestroy(b), "destroy b");
    return (double) ms / iters;
}

}  // namespace

int main() {
    using strata::kernels::H;
    using strata::kernels::FF;
    using strata::kernels::ROW_GU;
    using strata::kernels::SC_GU;
    using strata::kernels::ROW_D;
    using strata::kernels::SC_D;
    using strata::kernels::O_D_SCALES;

    const size_t blob_bytes = O_D_SCALES + (size_t) H * SC_D * 2;
    const size_t gu_bytes_per_hit = (size_t) 2 * FF * ROW_GU + (size_t) 2 * FF * SC_GU * 2;  // 921,600
    const size_t dn_bytes_per_hit = (size_t) H * ROW_D + (size_t) H * SC_D * 2;              // 460,800
    std::printf("# moe_hit_sat: blob=%zu B (gu %zu + down %zu), N_SLOTS=%d, ITERS=%d, THREADS=%d\n",
                blob_bytes, gu_bytes_per_hit, dn_bytes_per_hit, N_SLOTS, ITERS, strata::kernels::THREADS);

    Buffers bu;
    cuda_check(cudaMalloc(&bu.blobs, N_SLOTS * blob_bytes), "blob alloc");
    const int max_hits = 3328;
    cuda_check(cudaMalloc(&bu.slot_index, (size_t) ITERS * max_hits * sizeof(int32_t)), "slot alloc");
    cuda_check(cudaMalloc(&bu.dst_index, (size_t) max_hits * sizeof(int32_t)), "dst alloc");
    cuda_check(cudaMalloc(&bu.x_q8_0, (size_t) (H / 32) * 34), "x alloc");
    cuda_check(cudaMalloc(&bu.x_scales, (size_t) (H / 32) * sizeof(float)), "xs alloc");
    cuda_check(cudaMalloc(&bu.h_q8_0, (size_t) max_hits * (FF / 32) * 34), "h alloc");
    cuda_check(cudaMalloc(&bu.h_scales, (size_t) max_hits * (FF / 32) * sizeof(float)), "hs alloc");
    cuda_check(cudaMalloc(&bu.gate_up, (size_t) max_hits * 2 * FF * sizeof(float)), "gu alloc");
    cuda_check(cudaMalloc(&bu.out, (size_t) max_hits * H * sizeof(float)), "out alloc");
    cuda_check(cudaMalloc(&bu.sink, sizeof(float)), "sink alloc");

    // Representative contents: arbitrary nonzero bytes for codes (dp4a timing is value-independent),
    // fp16 scales that decode to ordinary normals (0x3C00 = 1.0), activations likewise.
    {
        std::vector<uint8_t> host(blob_bytes);
        std::mt19937 rng(20261120);
        for (auto& b : host) b = (uint8_t) (rng() & 0xFF);
        for (size_t off = O_D_SCALES; off < blob_bytes; off += 2) { host[off] = 0x00; host[off + 1] = 0x3C; }
        for (int s = 0; s < N_SLOTS; ++s) cuda_check(cudaMemcpy(bu.blobs + (size_t) s * blob_bytes, host.data(), blob_bytes, cudaMemcpyHostToDevice), "blob fill");
        std::vector<uint8_t> x((size_t) (H / 32) * 34);
        for (auto& b : x) b = (uint8_t) (rng() & 0xFF);
        for (size_t off = 0; off < x.size(); off += 34) { x[off] = 0x00; x[off + 1] = 0x3C; }
        cuda_check(cudaMemcpy(bu.x_q8_0, x.data(), x.size(), cudaMemcpyHostToDevice), "x fill");
        std::vector<float> ones((size_t) (H / 32), 1.f);
        cuda_check(cudaMemcpy(bu.x_scales, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice), "xs fill");
        std::vector<uint8_t> h((size_t) max_hits * (FF / 32) * 34);
        for (auto& b : h) b = (uint8_t) (rng() & 0xFF);
        cuda_check(cudaMemcpy(bu.h_q8_0, h.data(), h.size(), cudaMemcpyHostToDevice), "h fill");
        std::vector<float> hones((size_t) max_hits * (FF / 32), 1.f);
        cuda_check(cudaMemcpy(bu.h_scales, hones.data(), hones.size() * 4, cudaMemcpyHostToDevice), "hs fill");
    }

    // ROTATE slot ladder: a shuffle of all N_SLOTS; iteration i uses hits [i*n, i*n+n) of the
    // flattened ITERS x max_hits plan, so every iteration reads fresh experts (footprint >> L2).
    {
        std::vector<int32_t> plan((size_t) ITERS * max_hits);
        std::vector<int32_t> perm(N_SLOTS);
        for (int i = 0; i < N_SLOTS; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), std::mt19937(4242));
        for (size_t i = 0; i < plan.size(); ++i) plan[i] = perm[i % N_SLOTS];
        cuda_check(cudaMemcpy(bu.slot_index, plan.data(), plan.size() * 4, cudaMemcpyHostToDevice), "plan fill");
        std::vector<int32_t> ids(max_hits);
        for (int i = 0; i < max_hits; ++i) ids[i] = i;
        cuda_check(cudaMemcpy(bu.dst_index, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "dst fill");
    }

    // ANCHOR: the same span, streamed.
    {
        const size_t n_vec = (size_t) N_SLOTS * blob_bytes / 16;
        const int blocks = 170 * 8;   // grid-stride; enough blocks to fill every SM
        const double ms = time_kernel([&](int) { stream_read_kernel<<<blocks, 256>>>(reinterpret_cast<const uint4*>(bu.blobs), n_vec, bu.sink); }, 16);
        const double gbs = (double) n_vec * 16 / (ms / 1e3) / 1e9;
        std::printf("ANCHOR stream_read %zu B in %.3f ms -> %.0f GB/s (the part's achievable stream)\n",
                    (size_t) N_SLOTS * blob_bytes, ms, gbs);
    }

    // THE SWEEP. Weight bytes are the headline (what the perf-hunt priced); touched adds the
    // activation side (~2%) for completeness. Grid math is `launch_hit_*`'s own.
    const int sweep[] = {13, 26, 40, 52, 104, 208, 416, 832, 1664, 3328};
    const int warps = strata::kernels::THREADS / 32;
    for (int n_hits : sweep) {
        for (int mode = 0; mode < 2; ++mode) {   // 0 = ROTATE (DRAM-honest), 1 = HOT (L2 ceiling)
            const int32_t* slots = mode == 0 ? bu.slot_index : bu.slot_index;  // ROTATE advances below; HOT pins hit 0
            const double gu_ms = time_kernel([&](int it) {
                const int32_t* s = mode == 0 ? slots + (size_t) it * max_hits : slots;
                const long long pairs = (long long) n_hits * FF;
                strata::kernels::gu_pair_kernel<<<(unsigned) ((pairs + warps - 1) / warps), strata::kernels::THREADS>>>(
                    bu.blobs, s, (long long) blob_bytes, bu.x_q8_0, bu.x_scales, bu.gate_up, n_hits, nullptr, nullptr, 0);
            }, ITERS);
            const double dn_ms = time_kernel([&](int it) {
                const int32_t* s = mode == 0 ? slots + (size_t) it * max_hits : slots;
                const long long pairs = (long long) n_hits * (H / 2);
                strata::kernels::down_pair_kernel<<<(unsigned) ((pairs + warps - 1) / warps), strata::kernels::THREADS>>>(
                    bu.blobs, s, bu.dst_index, (long long) blob_bytes, bu.h_q8_0, bu.h_scales, bu.out, n_hits, nullptr);
            }, ITERS);
            const double gu_gb = (double) n_hits * gu_bytes_per_hit / (gu_ms / 1e3) / 1e9;
            const double dn_gb = (double) n_hits * dn_bytes_per_hit / (dn_ms / 1e3) / 1e9;
            const double both_ms = gu_ms + dn_ms;
            const double both_gb = (double) n_hits * (gu_bytes_per_hit + dn_bytes_per_hit) / (both_ms / 1e3) / 1e9;
            cuda_check(cudaGetLastError(), "sweep launch");
            std::printf("SAT n_hits=%4d mode=%s gu=%.0f GB/s (%.3f ms) down=%.0f GB/s (%.3f ms) gu+down=%.0f GB/s\n",
                        n_hits, mode == 0 ? "ROTATE" : "HOT   ", gu_gb, gu_ms, dn_gb, dn_ms, both_gb);
        }
    }
    std::printf("# decision inputs: live n_hits ~13-40 per layer-launch; ANCHOR is the roofline;\n"
                "#   ROTATE at saturated n_hits decides ALU/BW vs issue-rate (perf-hunt table);\n"
                "#   the ROTATE->HOT and n_hits spread prices occupancy and L2 residency.\n");
    return 0;
}
