#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s line %d\n", cudaGetErrorString(e), __LINE__); return 1;} } while(0)
int main() {
    const size_t N = 256ull<<20, S = 2<<20;
    char *h = (char*) malloc(N);           // PAGEABLE source, like the engine's file-mapped fill
    memset(h, 1, N);                        // pages resident (as after prewarm)
    char *d; CK((cudaError_t)cudaMalloc((void**)&d, N));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaMemcpy(d, h, N, cudaMemcpyHostToDevice));   // warmup + link uptrain
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i=0;i<40;i++) CK(cudaMemcpy(d, h, N, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    printf("PAGEABLE big 256MiB x40: %.2f GB/s\n", (double)N*40/1e9/(ms/1e3)); fflush(stdout);
    CK(cudaEventRecord(a));
    for (int i=0;i<9900;i++) CK(cudaMemcpy(d, h, S, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("PAGEABLE engine-fill pattern 2MiB x9900: %.2f GB/s (%.2f s total)\n", (double)S*9900/1e9/(ms/1e3), ms/1e3);
    printf("PROBE2 DONE\n");
    return 0;
}
