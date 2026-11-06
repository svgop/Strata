// bench/micro/cond_node.cu - the STRATA_WINDOW_GRAPH falsifier bench (digest 2026-11-05-condgraph-zero-host, section 3.5).
//
// Three probes, cheapest first, all GPU-synthetic, no model, cudart only:
//
//   (a) mapped-flag visibility on this driver (sm_120 / CUDA 13.x expected): the host STORES a mapped word, a
//       spin kernel sees it and stores a mapped answer back; the host read of the answer closes the round trip.
//       The engine's whole window handshake rides this cost - the gate is ~3 us median.  (Carried 11-04: 1-2 us
//       on PCIe5-class parts; this is the sm_120/CUDA-13 revalidation the brief demands before trusting the waits.)
//  (c-i) the conditional-WHILE machinery: one pass per launch across 10k launches (the cudaGraphCondAssignDefault
//       initializer re-arms with no host call and no drift), then a 3-iteration WHILE (the WHILE actually iterates
//       and terminates - proof the conditional node is live, not silently stripped).
//   (d) node-dispatch pricing: a straight-line body of N kernel nodes replayed, vs the same body wrapped in the
//       WHILE running exactly once - the delta is what one conditional node + one tail kernel cost per launch.
//
// Exit 0 = every gate met; 1 = a gate failed (print says which); 77 = conditional nodes unsupported by this
// runtime (CUDA < 12.4) - skip, not fail.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

__global__ void ring_kernel(const volatile unsigned* in, volatile unsigned* out, int rounds) {
    // One kernel, resident for every round - the engine's own shape (the window's wait kernels spin while the
    // host stores the flag; launch latency is NOT part of the handshake).  Round r: see *in >= r, fence, answer.
    for (int r = 1; r <= rounds; ++r) {
        while (*in < (unsigned) r) __nanosleep(64);
        __threadfence_system();
        *out = (unsigned) r;
    }
}

__global__ void tick_kernel(unsigned* counter) { atomicAdd(counter, 1u); }

__global__ void stub_kernel(unsigned* w) {
    if (threadIdx.x > 1024u) *w = 1u;   // never true: the node must stay, the work must stay nil
}

__global__ void tail_zero_kernel(cudaGraphConditionalHandle h) { cudaGraphSetConditional(h, 0u); }

__global__ void tail_passes_kernel(const unsigned* counter, cudaGraphConditionalHandle h) {
    cudaGraphSetConditional(h, *counter < 3u ? 1u : 0u);
}

int main() {
    int dev = 0;
    gpu(cudaGetDevice(&dev), "get device");
    cudaDeviceProp prop{};
    gpu(cudaGetDeviceProperties(&prop, dev), "device props");
    std::printf("cond_node: %s sm_%d%d, CUDA %d.%d\n", prop.name, prop.major, prop.minor,
                CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
#if CUDART_VERSION < 12040
    std::printf("SKIP: conditional nodes need CUDA >= 12.4, this runtime is %d\n", CUDART_VERSION);
    return 77;
#endif
    cudaStream_t cs = nullptr;
    gpu(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");

    // ---- (a) the mapped round trip ------------------------------------------------------------
    unsigned *h_in = nullptr, *h_out = nullptr, *d_in = nullptr, *d_out = nullptr;
    gpu(cudaHostAlloc(&h_in, 4, cudaHostAllocMapped), "hostalloc in");
    gpu(cudaHostAlloc(&h_out, 4, cudaHostAllocMapped), "hostalloc out");
    gpu(cudaHostGetDevicePointer((void**) &d_in, h_in, 0), "devptr in");
    gpu(cudaHostGetDevicePointer((void**) &d_out, h_out, 0), "devptr out");
    *h_in = 0;
    *h_out = 0;
    std::printf("(a) mapped-flag round trip (host store -> resident spin kernel -> host read):\n");
    std::vector<double> rtt;
    constexpr int kRounds = 2000, kWarm = 100;
    ring_kernel<<<1, 1, 0, cs>>>(d_in, d_out, kRounds + kWarm);   // resident for every round below
    gpu(cudaStreamQuery(cs), "ring launch");   // not a sync: the query only flushes the submit
    for (int i = 1; i <= kRounds + kWarm; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        *(volatile unsigned*) h_in = (unsigned) i;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (;;) {
            if (*(volatile unsigned*) h_out == (unsigned) i) break;
            if (ms_since(t0) > 100.0) {
                std::fprintf(stderr, "FAIL: round %d never came back - mapped visibility is broken\n", i);
                std::exit(1);
            }
        }
        const double us = ms_since(t0) * 1000.0;
        if (i > kWarm) rtt.push_back(us);
    }
    std::sort(rtt.begin(), rtt.end());
    const double p50 = rtt[rtt.size() / 2], p90 = rtt[(size_t) (rtt.size() * 0.9)], mx = rtt.back(), mn = rtt.front();
    std::printf("    %d rounds: min %.2f us, p50 %.2f us, p90 %.2f us, max %.2f us\n", (int) rtt.size(), mn, p50, p90, mx);
    const bool a_ok = p50 <= 3.0;
    std::printf("    %s (gate: p50 <= 3.0 us)\n", a_ok ? "PASS" : "FAIL");

    // ---- (c-i) one pass per launch, 10k launches, re-arm with no host call ----------------------
    unsigned* counter = nullptr;
    gpu(cudaMalloc(&counter, 4), "counter alloc");
    gpu(cudaMemset(counter, 0, 4), "counter zero");
    {
        cudaGraphExec_t exec = nullptr;
        cudaGraphConditionalHandle h = 0;   // an unsigned long long: no nullptr
        // (the tail writes the handle; the stubs make the body wide enough to price dispatch honestly)
        cudaGraph_t root = nullptr;
        gpu(cudaGraphCreate(&root, 0), "graph create");
        gpu(cudaGraphConditionalHandleCreate(&h, root, 1u, cudaGraphCondAssignDefault), "handle create");
        cudaGraphNodeParams p{};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = h;
        p.conditional.type = cudaGraphCondTypeWhile;
        p.conditional.size = 1;
        cudaGraphNode_t cond_node = nullptr;
        gpu(cudaGraphAddNode(&cond_node, root, nullptr, nullptr, 0, &p), "conditional add");
        cudaGraph_t body = p.conditional.phGraph_out[0];
        gpu(cudaStreamBeginCaptureToGraph(cs, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal),
            "begin capture to body");
        for (int i = 0; i < 16; ++i) stub_kernel<<<1, 32, 0, cs>>>(counter);
        tail_zero_kernel<<<1, 1, 0, cs>>>(h);
        gpu(cudaStreamEndCapture(cs, nullptr), "end capture");
        gpu(cudaGraphInstantiate(&exec, root, 0), "instantiate");
        gpu(cudaGraphUpload(exec, cs), "upload");
        gpu(cudaStreamSynchronize(cs), "post-upload sync");
        cudaGraphDestroy(root);
        constexpr int kLaunches = 10000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kLaunches; ++i) {
            gpu(cudaGraphLaunch(exec, cs), "launch");
        }
        gpu(cudaStreamSynchronize(cs), "10k sync");
        const double total = ms_since(t0);
        unsigned got = 0;
        gpu(cudaMemcpy(&got, counter, 4, cudaMemcpyDeviceToHost), "counter read");
        const bool ci_ok = got == (unsigned) kLaunches;
        std::printf("(c-i) %d launches of the one-pass WHILE: body ran %u times (%.2f us/launch incl. submit)\n",
                    kLaunches, got, total * 1000.0 / kLaunches);
        std::printf("      %s (gate: exactly once per launch - the AssignDefault initializer re-arms, no drift)\n",
                    ci_ok ? "PASS" : "FAIL");
        if (!ci_ok) std::exit(1);
        cudaGraphExecDestroy(exec);
    }

    // ---- (c-ii) the WHILE iterates and terminates: tail keeps the handle while counter < 3 -------------
    {
        gpu(cudaMemset(counter, 0, 4), "counter zero");
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t root = nullptr;
        cudaGraphConditionalHandle h = 0;   // an unsigned long long: no nullptr
        gpu(cudaGraphCreate(&root, 0), "graph create");
        gpu(cudaGraphConditionalHandleCreate(&h, root, 1u, cudaGraphCondAssignDefault), "handle create");
        cudaGraphNodeParams p{};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = h;
        p.conditional.type = cudaGraphCondTypeWhile;
        p.conditional.size = 1;
        cudaGraphNode_t cond_node = nullptr;
        gpu(cudaGraphAddNode(&cond_node, root, nullptr, nullptr, 0, &p), "conditional add");
        cudaGraph_t body = p.conditional.phGraph_out[0];
        gpu(cudaStreamBeginCaptureToGraph(cs, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal),
            "begin capture to body");
        tick_kernel<<<1, 1, 0, cs>>>(counter);
        tail_passes_kernel<<<1, 1, 0, cs>>>(counter, h);
        gpu(cudaStreamEndCapture(cs, nullptr), "end capture");
        gpu(cudaGraphInstantiate(&exec, root, 0), "instantiate");
        gpu(cudaGraphUpload(exec, cs), "upload");
        gpu(cudaStreamSynchronize(cs), "post-upload sync");
        cudaGraphDestroy(root);
        gpu(cudaGraphLaunch(exec, cs), "launch");
        gpu(cudaStreamSynchronize(cs), "sync");
        unsigned got = 0;
        gpu(cudaMemcpy(&got, counter, 4, cudaMemcpyDeviceToHost), "counter read");
        const bool cii_ok = got == 3u;
        std::printf("(c-ii) the iterating WHILE: one launch, body ran %u times\n", got);
        std::printf("       %s (gate: 3 - the WHILE is live and terminates on the body's own write)\n",
                    cii_ok ? "PASS" : "FAIL");
        if (!cii_ok) std::exit(1);
        cudaGraphExecDestroy(exec);
    }

    // ---- (d) dispatch pricing: N-node straight line vs the same body under the WHILE, one pass ---------
    {
        constexpr int kStubs = 512, kReplays = 2000;
        unsigned* sink = counter;
        gpu(cudaMemset(sink, 0, 4), "sink zero");
        cudaGraphExec_t exec_line = nullptr, exec_cond = nullptr;
        {   // straight line
            cudaGraph_t g = nullptr;
            gpu(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin line capture");
            for (int i = 0; i < kStubs; ++i) stub_kernel<<<1, 32, 0, cs>>>(sink);
            gpu(cudaStreamEndCapture(cs, &g), "end line capture");
            gpu(cudaGraphInstantiate(&exec_line, g, 0), "instantiate line");
            cudaGraphDestroy(g);
        }
        {   // wrapped, one pass
            cudaGraph_t root = nullptr;
            cudaGraphConditionalHandle h = 0;   // an unsigned long long: no nullptr
            gpu(cudaGraphCreate(&root, 0), "graph create");
            gpu(cudaGraphConditionalHandleCreate(&h, root, 1u, cudaGraphCondAssignDefault), "handle create");
            cudaGraphNodeParams p{};
            p.type = cudaGraphNodeTypeConditional;
            p.conditional.handle = h;
            p.conditional.type = cudaGraphCondTypeWhile;
            p.conditional.size = 1;
            cudaGraphNode_t cond_node = nullptr;
            gpu(cudaGraphAddNode(&cond_node, root, nullptr, nullptr, 0, &p), "conditional add");
            cudaGraph_t body = p.conditional.phGraph_out[0];
            gpu(cudaStreamBeginCaptureToGraph(cs, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal),
                "begin capture to body");
            for (int i = 0; i < kStubs; ++i) stub_kernel<<<1, 32, 0, cs>>>(sink);
            tail_zero_kernel<<<1, 1, 0, cs>>>(h);
            gpu(cudaStreamEndCapture(cs, nullptr), "end capture");
            gpu(cudaGraphInstantiate(&exec_cond, root, 0), "instantiate cond");
            cudaGraphDestroy(root);
        }
        gpu(cudaGraphUpload(exec_line, cs), "upload line");
        gpu(cudaGraphUpload(exec_cond, cs), "upload cond");
        gpu(cudaStreamSynchronize(cs), "post-upload sync");
        double t_line = 0, t_cond = 0;
        for (int pass = 0; pass < 2; ++pass) {   // two passes, report the second (cache-warm)
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kReplays; ++i) gpu(cudaGraphLaunch(exec_line, cs), "launch line");
            gpu(cudaStreamSynchronize(cs), "line sync");
            if (pass == 1) t_line = ms_since(t0);
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < kReplays; ++i) gpu(cudaGraphLaunch(exec_cond, cs), "launch cond");
            gpu(cudaStreamSynchronize(cs), "cond sync");
            if (pass == 1) t_cond = ms_since(t1);
        }
        std::printf("(d) %d-node body x %d replays: straight line %.2f ms (%.2f us/replay), "
                    "WHILE-wrapped %.2f ms (%.2f us/replay), conditional delta %.2f us/replay\n",
                    kStubs, kReplays, t_line, t_line * 1000.0 / kReplays, t_cond, t_cond * 1000.0 / kReplays,
                    (t_cond - t_line) * 1000.0 / kReplays);
        cudaGraphExecDestroy(exec_line);
        cudaGraphExecDestroy(exec_cond);
    }

    std::printf("cond_node: %s\n", a_ok ? "the mapped handshake is cheap enough for in-graph waits" :
                                         "mapped-flag visibility misses the gate - STOP before the window arm");
    cudaStreamDestroy(cs);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(counter);
    return a_ok ? 0 : 1;
}
