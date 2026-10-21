// F-255/T27-1: cuBLASLt heuristic sweep on our prefill GEMM shapes (zero-boot microbench).
// Question: does the cuBLAS heuristic cublasGemmEx picks lose real TFLOPS on sm_120 for the
// exact (M,N,K) set our prefill runs, vs the best cublasLtMatmulAlgoGetHeuristic result?
// Shapes from T26-3's inventory: dense qkv/gate/out (fp16-in, fp32-accum, COMPUTE_32F) and the
// bf16 hc/router projections. M = prompt-chunk token counts we actually launch.
// Convention (identical for both paths): C[M,N] = act[M,K] . W[N,K]^T, all row-major, encoded
// column-major as C'[N,M] = opT(W_c[K,N]) . act_c[K,M] with ld = K / K / N.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <functional>

#define CK(x) do { auto e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) { \
  printf("CUBLAS ERR %d at %s:%d\n", (int)e_, __FILE__, __LINE__); exit(1); } } while (0)
#define CKC(x) do { auto e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA ERR %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct Shape { const char* name; int n, k; int type; };  // type 0 = fp16 in, 1 = bf16 in

static double timeit(int iters, const std::function<void()>& fn) {
  fn();  // warm
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) fn();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms;
}

static double bench_gemmex(cublasHandle_t h, const Shape& s, int M, void* act, void* w, void* c, int iters) {
  cudaDataType_t dt = s.type == 0 ? CUDA_R_16F : CUDA_R_16BF;
  const float one = 1.f, zero = 0.f;
  double ms = timeit(iters, [&] {
    CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, M, s.k, &one, w, dt, s.k, act, dt, s.k,
                    &zero, c, CUDA_R_32F, s.n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  });
  return 2.0 * M * s.n * s.k * iters / (ms * 1e-3) / 1e12;
}

static double bench_lt(cublasLtHandle_t lt, const Shape& s, int M, void* act, void* w, void* c,
                       void* ws, size_t wsSize, int iters, int& algoIdx) {
  cudaDataType_t dt = s.type == 0 ? CUDA_R_16F : CUDA_R_16BF;
  cublasLtMatmulDesc_t op;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tA = CUBLAS_OP_T, nB = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nB, sizeof(nB)));
  cublasLtMatrixLayout_t lw, la, lc;  // weights (K x N) opT, act (K x M), out (N x M)
  CK(cublasLtMatrixLayoutCreate(&lw, dt, s.k, s.n, s.k));
  CK(cublasLtMatrixLayoutCreate(&la, dt, s.k, M, s.k));
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, s.n, M, s.n));
  const float one = 1.f, zero = 0.f;
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
  cublasLtMatmulHeuristicResult_t res[8]; int nres = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, lw, la, lc, lc, pref, 8, res, &nres));
  double best = -1; int bestI = -1;
  for (int i = 0; i < nres; i++) {
    double ms = timeit(iters, [&] {
      CK(cublasLtMatmul(lt, op, &one, w, lw, act, la, &zero, c, lc, c, lc, &res[i].algo, ws, wsSize, 0));
    });
    double tf = 2.0 * M * s.n * s.k * iters / (ms * 1e-3) / 1e12;
    if (tf > best) { best = tf; bestI = i; }
  }
  algoIdx = bestI;
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(lw);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
  return best;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 200;
  const int Ms[] = {256, 512, 1024, 2048, 4096, 8192};
  std::vector<Shape> shapes = {
    {"qkv 10240x2560", 10240, 2560, 0},
    {"gate 6144x2560", 6144, 2560, 0},
    {"out 6144x2560", 6144, 2560, 0},
    {"hc_down 320x10240", 320, 10240, 1},
    {"hc_up 10240x320", 10240, 320, 1},
    {"router 512x2560", 512, 2560, 1},
  };
  cublasHandle_t h; CK(cublasCreate(&h));
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  size_t wsSize = 64 << 20; void *act, *w, *c, *ws;
  int maxN = 0, maxK = 0;
  for (auto& s : shapes) { if (s.n > maxN) maxN = s.n; if (s.k > maxK) maxK = s.k; }
  CKC(cudaMalloc(&act, (size_t)8192 * maxK * 2));
  CKC(cudaMalloc(&w, (size_t)maxN * maxK * 2));
  CKC(cudaMalloc(&c, (size_t)8192 * maxN * 4));
  CKC(cudaMalloc(&ws, wsSize));
  CKC(cudaMemset(act, 0x3c, (size_t)8192 * maxK * 2));
  CKC(cudaMemset(w, 0x3c, (size_t)maxN * maxK * 2));
  printf("%-18s %6s %10s %10s %8s\n", "shape", "M", "GemmEx", "Lt-best", "ratio");
  for (auto& s : shapes)
    for (int M : Ms) {
      double ge = bench_gemmex(h, s, M, act, w, c, iters);
      int ai = -1;
      double ltB = bench_lt(lt, s, M, act, w, c, ws, wsSize, iters, ai);
      printf("%-18s %6d %9.1f %9.1f %8.2fx  (algo %d of Lt)\n", s.name, M, ge, ltB, ltB / ge, ai);
    }
  return 0;
}
