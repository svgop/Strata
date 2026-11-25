#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s line %d\n", cudaGetErrorString(e), __LINE__); return 1;} } while(0)
int main() {
    const size_t N = 256ull<<20;
    char *h, *d;
    CK(cudaSetDevice(0));
    CK((cudaError_t)cudaHostAlloc((void**)&h, N, cudaHostAllocDefault));
    CK((cudaError_t)cudaMalloc((void**)&d, N));
    memset(h, 1, N);
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaMemcpy(d, h, N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost));
    CK(cudaDeviceSynchronize());
    for (int pass=0; pass<3; pass++) {
        for (int dir=0; dir<2; dir++) {
            const int reps = 40;
            CK(cudaEventRecord(a));
            for (int i=0;i<reps;i++) CK(cudaMemcpy(dir?h:d, dir?d:h, N, dir?cudaMemcpyDeviceToHost:cudaMemcpyHostToDevice));
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms, a, b));
            printf("pass%d %s big 256MiB x%d: %.2f GB/s\n", pass, dir?"D2H":"H2D", reps, (double)N*reps/1e9/(ms/1e3));
            fflush(stdout);
        }
    }
    const size_t S = 2<<20;
    CK(cudaEventRecord(a));
    for (int i=0;i<9900;i++) CK(cudaMemcpy(d, h, S, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    printf("H2D engine-fill pattern 2MiB x9900: %.2f GB/s (%.2f s total)\n", (double)S*9900/1e9/(ms/1e3), ms/1e3);
    fflush(stdout);
    printf("PROBE DONE\n");
    return 0;
}
