// FALSIFIER: does this card half-rate the fp32-accumulate tensor pipe like the
// Ampere FMLA32 lock (see the gpu-unlock / 3090 story, 2026-10-21)?
// Pure register mma.sync chains, no memory: warp-level m16n8k16, fp16 inputs,
// fp16-accumulate vs fp32-accumulate. Equal rates -> the 5090 is clean and the
// unlock is irrelevant to us; 2x -> the lock exists and the TC-hc plan is
// half-priced until it is cleared.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ITERS 200000  // inner loop count; each iter = 8 independent mma

__global__ void __launch_bounds__(256) mma_f16acc(float* out, int iters) {
    uint32_t a0 = threadIdx.x + 1, a1 = threadIdx.x + 2, a2 = threadIdx.x + 3, a3 = threadIdx.x + 4;
    uint32_t b0 = threadIdx.x + 5, b1 = threadIdx.x + 6;
    uint32_t c[8] = {a0, a1, a2, a3, b0, b1, a0 ^ b0, a1 ^ b1};
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < 8; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                : "+r"(c[j]), "+r"(c[j ^ 1])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    uint32_t s = 0;
    for (int j = 0; j < 8; j++) s += c[j];
    if (s == 0xdeadbeef) out[threadIdx.x] = 1.f;  // keep alive
}

__global__ void __launch_bounds__(256) mma_f32acc(float* out, int iters) {
    uint32_t a0 = threadIdx.x + 1, a1 = threadIdx.x + 2, a2 = threadIdx.x + 3, a3 = threadIdx.x + 4;
    uint32_t b0 = threadIdx.x + 5, b1 = threadIdx.x + 6;
    float c[4][4];
    for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) c[j][q] = (float)(a0 + q);
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < 4; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0;
    for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) s += c[j][q];
    if (s == 12345.678f) out[threadIdx.x] = s;
}

// bf16 inputs, fp32 accumulate - the form the TC-hc mma plan would use.
__global__ void __launch_bounds__(256) mma_bf16_f32acc(float* out, int iters) {
    uint32_t a0 = threadIdx.x + 1, a1 = threadIdx.x + 2, a2 = threadIdx.x + 3, a3 = threadIdx.x + 4;
    uint32_t b0 = threadIdx.x + 5, b1 = threadIdx.x + 6;
    float c[4][4];
    for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) c[j][q] = (float)(b0 + q);
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < 4; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0;
    for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) s += c[j][q];
    if (s == 12345.678f) out[threadIdx.x] = s;
}

static double run(void (*kp)(float*, int), const char* name, int mma_per_iter) {
    int blocks = 170 * 4, threads = 256;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    kp<<<blocks, threads>>>(nullptr, ITERS);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    kp<<<blocks, threads>>>(nullptr, ITERS);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double flop = (double)blocks * (threads / 32) * ITERS * mma_per_iter * 4096.0;  // 2*16*8*16
    double tflops = flop / (ms * 1e-3) / 1e12;
    printf("%-16s %8.2f ms  %7.2f TFLOPS\n", name, ms, tflops);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return tflops;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
    for (int leg = 0; leg < 2; leg++) {
        double f16 = run(mma_f16acc, "f16-accum", 8);
        double f32 = run(mma_f32acc, "f32-accum", 4);
        double bf  = run(mma_bf16_f32acc, "bf16-f32-accum", 4);
        printf("  ratio f16/f32 = %.2f   (1.0 = clean, ~2.0 = FMLA32-style lock)\n", f16 / f32);
    }
    return 0;
}
