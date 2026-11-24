// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include "strata/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;
bool g_gate_open = true;      // table 1's phase gate: open until the F-17 phase logic (per-table form) closes it
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr;
                   float* dir2 = nullptr; float* s2 = nullptr;      // dir2/s2: the optional second armed table
                   int* gate = nullptr; };                          // gate: table 1's phase gate (see cvec.hpp)
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host, g_dir2_host, g_s2_host;
int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}
// Allocate and fill this device's slot from the host mirrors.  The first table always rides; the second
// only when one is armed.  A failure frees every partial allocation (a leaked slot would masquerade as
// "uploaded" on the next call) and leaves the slot reset.
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    const bool two = !g_dir2_host.empty();
    if (t.dir != nullptr && (!two || t.dir2 != nullptr)) return true;   // this device already holds what it must
    const int flag = g_on_host ? 1 : 0, gate_flag = g_gate_open ? 1 : 0;
    const size_t bytes = g_dir_host.size() * sizeof(float), sbytes = g_s_host.size() * sizeof(float);
    const bool ok = cudaMalloc(&t.dir, bytes) == cudaSuccess &&
                    cudaMalloc(&t.s, sbytes) == cudaSuccess &&
                    cudaMalloc(&t.on, sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&t.gate, sizeof(int)) == cudaSuccess &&
                    (!two || cudaMalloc(&t.dir2, bytes) == cudaSuccess) &&
                    (!two || cudaMalloc(&t.s2, sbytes) == cudaSuccess) &&
                    cudaMemcpy(t.dir, g_dir_host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(t.s, g_s_host.data(), sbytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(t.gate, &gate_flag, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess &&
                    (!two || cudaMemcpy(t.dir2, g_dir2_host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess) &&
                    (!two || cudaMemcpy(t.s2, g_s2_host.data(), sbytes, cudaMemcpyHostToDevice) == cudaSuccess);
    if (!ok) {
        err = two ? "control vector: device allocation failed (two tables)" : "control vector: device allocation failed";
        cudaFree(t.dir); cudaFree(t.s); cudaFree(t.on); cudaFree(t.dir2); cudaFree(t.s2); cudaFree(t.gate);   // cudaFree(nullptr) is a no-op
        t = DevTables{};
        return false;
    }
    return true;
}

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// one block per (stream, token): the pending write, then h . v over the stream, then the update
__global__ void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                            const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                            const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj,
                            int64_t inj_ld, int write, const int* __restrict__ gate) {
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && *gate != 0 && s != 0.0f;   // uniform over the block (gate: table 1's phase)
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fmaf(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fmaf(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
        __shared__ float part[THREADS / 32];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fmaf(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

// Free every device's slot (both tables).  The one owner of teardown: a new first table replaces everything,
// and cvec_upload2 re-materialises both tables through one upload so no device slot is ever half-built.
void release_devices() {
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the tables
        cudaFree(g_dev[d].gate);
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        cudaFree(g_dev[d].on);
        cudaFree(g_dev[d].dir2);
        cudaFree(g_dev[d].s2);
        g_dev[d] = DevTables{};
    }
    cudaSetDevice(prev);
}

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    release_devices();
    g_dir_host = dir;
    g_s_host = s;
    g_dir2_host.clear();   // a first-table upload replaces both tables (cvec_upload2 re-arms the second)
    g_s2_host.clear();
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.gate = t.gate;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    g_cvec.dir2 = nullptr;   // a first-table upload disarms the second (cvec_upload2 re-arms it)
    g_cvec.s2 = nullptr;
    g_cvec.steered2.clear();
    g_gate_open = true;      // the phase gate starts open; the per-table phase logic closes it per request
    return true;
}

bool cvec_upload2(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                  int64_t n_embd, int64_t hc, std::string& err) {
    if (!g_cvec.loaded()) { err = "cvec_upload2: upload the first table first"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd || n_embd != g_cvec.n_embd || hc != g_cvec.hc) {
        err = "control vector: bad second-table sizes (must match the first table's geometry)"; return false;
    }
    g_dir2_host = dir;
    g_s2_host = s;
    release_devices();        // load time only (before any graph capture): rebuild both slots in one pass
    if (!upload_here(err)) { g_dir2_host.clear(); g_s2_host.clear(); return false; }
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;       // the rebuilt slot's first-table pointers (the device addresses changed)
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.gate = t.gate;
    g_cvec.dir2 = t.dir2;
    g_cvec.s2 = t.s2;
    g_cvec.mode2 = mode;
    g_cvec.first2 = first;
    g_cvec.last2 = last;
    g_cvec.steered2.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered2[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

// write one device int on every device that holds the tables (synchronized: nothing in flight may still read it)
void write_flag(int* DevTables::*slot, bool* host_mirror, bool v) {
    if (*host_mirror == v) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int flag = v ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaMemcpy(g_dev[d].*slot, &flag, sizeof(int), cudaMemcpyHostToDevice);   // the device buffer, never the pointer field
    }
    cudaSetDevice(prev);
    *host_mirror = v;
}

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded()) return;
    write_flag(&DevTables::on, &g_on_host, on);
}

void cvec_set_table1_gate(bool open) {
    if (!g_cvec.loaded()) return;
    write_flag(&DevTables::gate, &g_gate_open, open);
}

void cvec_set_phase_table1(bool on) { g_cvec.phase_table1 = on && g_cvec.loaded(); }

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

bool cvec_tables(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded()) return false;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) return false;
    *dir = t.dir; *s = t.s; *on = t.on;
    return true;
}

bool cvec_tables2(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded2()) return false;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir2 == nullptr) return false;
    *dir = t.dir2; *s = t.s2; *on = t.on;
    return true;
}

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const dim3 grid((unsigned) g_cvec.hc, (unsigned) T);
    const bool first = g_cvec.steered_at(layer);
    if (first)
        cvec_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(R, t.dir, t.s, t.on, g_cvec.mode, layer,
                                                                (int) g_cvec.n_embd, (int) g_cvec.hc, r_ld, bo, bo_ld,
                                                                inj, inj_ld, write ? 1 : 0, t.gate);
    if (g_cvec.steered2_at(layer))
        // after the first table, same stream: the ops compose sequentially (h += a; h -= s (h . v) v).
        // the pending FFN write rides only the FIRST launch - the second sees R already holding it.
        // when THIS table is the only one at the layer, the write is here - so bo/inj ride with it.
        cvec_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(R, t.dir2, t.s2, t.on, g_cvec.mode2, layer,
                                                                (int) g_cvec.n_embd, (int) g_cvec.hc, r_ld, bo, bo_ld,
                                                                inj, inj_ld, (!first && write) ? 1 : 0, t.on);   // no phase: gated by the request flag alone
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply: launch failed");
}

}  // namespace strata::kernels
