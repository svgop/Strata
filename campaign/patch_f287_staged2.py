#!/usr/bin/env python3
"""campaign/patch_f287_staged2.py — F-287 rung 0: the staged down read with a WEIGHT-side
cp.async ring (3 stages) + 4 independent fp32 accumulator chains (kHcStaged2=4, env
STRATA_HC_SPLIT=4; default stays kHcStaged=3 — byte-identical standing path).
Design per T37-kernel-math §3.1 rung 0: the measured deficits are outstanding bytes
(~0.7 MB vs the 1.36 MB BW requirement) and the 320-FMA single accumulator chain; this
kills both without mma.  Same grid (41 blocks), same row/lane mapping, same output
contract; the sum order changes (4 chains combined in fixed order) = the GDN_CHUNKED
reassociation class (~2.8e-7 RMS), gated by the F-230 ladder if it clears the bench.
Weight staging: each thread cp.asyncs exactly the 5 uint4 its own lane consumes, so
weight groups need no block sync (activations keep theirs)."""
from pathlib import Path

P = Path("/home/svg/LLM/strata-custom-2/src/kernels/cuda/fused_gr.cu")
src = P.read_text()

def rep(old, new):
    global src
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old[:80]!r}"
    src = src.replace(old, new)

# 1. the variant constant
rep("constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;",
    "constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3, kHcStaged2 = 4;   // F-287 rung 0: weight cp.async ring + 4 acc chains")

# 2. the kernel, after gr_down_staged_kernel (anchor: the closing of that kernel + the volta comment)
rep("""// the current device is Volta (sm_70): the fast norm/up is its default
bool cur_dev_volta() {""",
    """// F-287 rung 0 (kHcStaged2, STRATA_HC_SPLIT=4): the staged down read with the WEIGHTS
// staged by cp.async into a 3-slot ring (per warp: its own row's 160 uint4 per tile, each
// thread staging exactly the 5 it consumes - no cross-thread weight dependency) and the
// per-token accumulator split into 4 independent chains combined in fixed order at the
// end.  Kills the two measured deficits of the staged default (outstanding bytes, the
// 320-FMA chain) at the same 41-block grid and row mapping.  Bits: the chunk adds
// reassociate (GDN_CHUNKED class, ~2.8e-7 RMS) - gated by the F-230 ladder, default OFF.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_staged2_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];      // 2 act buffers x [T][2][160] float4, then the weight ring
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int k0 = EXACT_T ? (int) blockIdx.y * MAX_T : 0;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per ACT buffer (the second follows)
    uint4* wsm = reinterpret_cast<uint4*>(hbuf + 2 * buf_f4);   // [3][WARPS][H_TILE/8] ring slots
    constexpr int WT = H_TILE / 8;                      // uint4 of one row's tile (160)
    const int wbase = warp * WT;
    float acc[MAX_T][4];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k][0] = acc[k][1] = acc[k][2] = acc[k][3] = 0.0f;
    auto stage_wts = [&](int h, int slot) {
        if (!active) return;
#pragma unroll
        for (int q = 0; q < HQ; ++q)
            cp_async16(&wsm[(size_t) slot * (WARPS * WT) + wbase + lane + 32 * q],
                       w4 + (size_t) h * WT + lane + 32 * q);
    };
    stage_htile(m, T, 0, hbuf, t, k0);
    stage_wts(0, 0);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, k0);
    stage_wts(1, 1);
    cp_async_commit();
    stage_wts(2, 2);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        // needed data: weights(h) in group h; acts(h) in group h (h<2) or h+1 (the refill at h-2).
        // committed so far: 3 + h.  allowed outstanding keeps the needed group complete.
        if (h < 2) cp_async_wait2();
        else cp_async_wait1();
        __syncthreads();                                // the ACT buffer is block-shared
        const float4* cur = hbuf + (h & 1) * buf_f4;
        const uint4* wr = wsm + (size_t) (h % 3) * (WARPS * WT) + wbase;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wr[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k][q & 3] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // every warp is done with act buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, k0);
        stage_wts(h + 3 < N_HTILES ? h + 3 : h, (h + 3 < N_HTILES) ? (h % 3) : 0);   // refill the slot just used
        cp_async_commit();                              // possibly empty: keeps the group count uniform
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (EXACT_T || k < T) s[k] = ((acc[k][0] + acc[k][1]) + acc[k][2]) + acc[k][3];
        else s[k] = 0.0f;
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        float sv = warp_sum(s[k]);
        if (lane == k) {
            if (inject_block) {
                m.a[k0 + k].inject_out[row] = sv;
            } else {
                const float x = sv / (float) HC;
                m.a[k0 + k].lo[row] = x / (1.0f + __expf(-x));
            }
        }
    }
}

// the current device is Volta (sm_70): the fast norm/up is its default
bool cur_dev_volta() {""")

# 3. env mapping: STRATA_HC_SPLIT=4 selects the rung-0 variant
rep("""    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;""",
    """    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    if (std::atoi(e) >= 4) return kHcStaged2;           // F-287 rung 0 (bits-class gated; default OFF)
    return kHcStaged;""")

# 4. the launch: staged2 mirrors the staged switch with the weight ring's smem added
rep("""        const size_t smem = (size_t) ct * per_tok;
""",
    """        const size_t wts_ring = (size_t) 3 * WARPS * (H_TILE / 8) * sizeof(uint4);   // F-287: staged2's weight ring
        const size_t smem = (size_t) ct * per_tok + (variant == kHcStaged2 ? wts_ring : 0);
""")
rep("""        } else if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
""",
    """        } else if (variant == kHcStaged2) {
            // F-287 rung 0: the weight-ring staged read (STRATA_HC_SPLIT=4); the same launch
            // ladder as staged - exact-T instantiations 1..6, then the generic/max4 forms.
            if (exact_t && ct == 1) gr_down_staged2_kernel<1, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_staged2_kernel<2, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_staged2_kernel<3, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_staged2_kernel<4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_staged2_kernel<5, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_staged2_kernel<6, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (max4) gr_down_staged2_kernel<4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_staged2_kernel<><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
""")

P.write_text(src)
print("F-287 staged2 patch applied")
