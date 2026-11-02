// campaign/probe-bandwidth.c - A1 pre-flight: the achievable DRAM ceiling, and what the
// access SHAPE costs on top of it.
//
// The lane reads one expert matrix at a time (contiguous, ~8.5 MB at 1.75 bpw) and jumps to
// unrelated experts between reads. Two patterns are therefore priced separately:
//
//   seq   - every thread streams one contiguous slice: pure bandwidth (the ceiling).
//   chunk - 8.5 MB blocks visited in a deterministic shuffle, round-robin across threads:
//           the lane's shape. seq - chunk  =  TLB + prefetcher loss.
//
// Each pattern runs with 4 KB pages and with THP (MADV_HUGEPAGE + MADV_COLLAPSE, no root,
// no khugepaged dependency), so the hugepage arm is priced before it is written into the engine.
//
// build: gcc -O3 -mavx512f -mavx512bw -pthread -o probe-bandwidth probe-bandwidth.c
// run:   ./probe-bandwidth <GiB> <threads> <reps>
#define _GNU_SOURCE
#include <errno.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static char *base;
static size_t region;
static size_t chunk = 8704ull << 10;      /* 8.5 MB: one routed expert at 1.75 bpw */
static int nthreads;
static int mode;                          /* 0 = seq, 1 = chunk */
static uint32_t *perm;
static size_t nchunks;
static __thread uint64_t sink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static inline void eat(const char *p, size_t len) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 256 <= len; i += 256) {          /* 4 lines per iteration: read-latency bound */
        acc = _mm512_add_epi64(acc, _mm512_loadu_si512((const void *) (p + i)));
        acc = _mm512_add_epi64(acc, _mm512_loadu_si512((const void *) (p + i + 64)));
        acc = _mm512_add_epi64(acc, _mm512_loadu_si512((const void *) (p + i + 128)));
        acc = _mm512_add_epi64(acc, _mm512_loadu_si512((const void *) (p + i + 192)));
    }
    for (; i + 64 <= len; i += 64)
        acc = _mm512_add_epi64(acc, _mm512_loadu_si512((const void *) (p + i)));
    sink += (uint64_t) _mm512_reduce_add_epi64(acc) + (unsigned char) p[len ? len - 1 : 0];
}

static void *worker(void *arg) {
    long t = (long) arg;
    if (mode == 0) {
        size_t per = region / (size_t) nthreads;
        per &= ~(size_t) 255;
        size_t off = per * (size_t) t;
        size_t len = (t == nthreads - 1) ? region - off : per;
        eat(base + off, len);
    } else {
        for (size_t k = (size_t) t; k < nchunks; k += (size_t) nthreads) {
            size_t c = perm[k];
            size_t off = c * chunk;
            size_t len = (off + chunk <= region) ? chunk : region - off;
            eat(base + off, len);
        }
    }
    return NULL;
}

static void run(const char *what, double *out) {
    pthread_t th[64];
    double t0 = now_s();
    for (long i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, (void *) i);
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    double dt = now_s() - t0;
    *out = (double) region / dt / 1e9;
    printf("  %-6s %s  %.2f GB/s  (%.3f s)\n", what, "", *out, dt);
    fflush(stdout);
}

static uint64_t anon_huge_kb(void) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    char line[256];
    uint64_t kb = 0;
    if (!f) return 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "AnonHugePages: %lu kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

int main(int argc, char **argv) {
    double gib = argc > 1 ? atof(argv[1]) : 8.0;
    nthreads = argc > 2 ? atoi(argv[2]) : 8;
    int reps = argc > 3 ? atoi(argv[3]) : 3;
    if (nthreads < 1 || nthreads > 64) nthreads = 8;
    region = (size_t) (gib * 1073741824.0);
    region &= ~((size_t) (2u << 20) - 1);

    printf("region %.2f GiB, %d threads, THP policy: ", (double) region / 1073741824.0, nthreads);
    {
        FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
        char b[128] = "?";
        if (f) { if (fgets(b, sizeof b, f)) b[strcspn(b, "\n")] = 0; fclose(f); }
        printf("%s\n", b);
    }

    base = mmap(NULL, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    /* permutation for the chunk pattern (deterministic Fisher-Yates) */
    nchunks = region / chunk;
    perm = malloc(nchunks * sizeof(uint32_t));
    if (!perm) { perror("malloc"); return 1; }
    for (size_t i = 0; i < nchunks; i++) perm[i] = (uint32_t) i;
    {
        uint64_t s = 0x9e3779b97f4a7c15ull;
        for (size_t i = nchunks - 1; i > 0; i--) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;      /* xorshift64* */
            size_t j = (size_t) (s % (uint64_t) (i + 1));
            uint32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
    }

    /* --- pass 1: 4 KB pages ------------------------------------------------------------- */
    madvise(base, region, MADV_NOHUGEPAGE);
    memset(base, 1, region);                     /* first touch: small pages, by policy */
    printf("4k pages: anon huge = %lu kB\n", (unsigned long) anon_huge_kb());

    double seq4 = 0, ch4 = 0;
    for (int r = 0; r < reps; r++) run("seq", &seq4);
    for (int r = 0; r < reps; r++) run("chunk", &ch4);

    /* --- pass 2: THP ------------------------------------------------------------------- */
    if (madvise(base, region, MADV_HUGEPAGE) != 0) perror("madvise(HUGEPAGE)");
#ifdef MADV_COLLAPSE
    if (madvise(base, region, MADV_COLLAPSE) != 0) perror("madvise(COLLAPSE)");
#endif
    printf("THP: anon huge = %lu kB\n", (unsigned long) anon_huge_kb());

    double seqT = 0, chT = 0;
    for (int r = 0; r < reps; r++) run("seq", &seqT);
    for (int r = 0; r < reps; r++) run("chunk", &chT);

    /* --- permutation shared by the chunk pattern (deterministic) ----------------------- */
    printf("\nshape loss: 4k chunk/seq = %.3f, THP chunk/seq = %.3f\n", ch4 / seq4, chT / seqT);
    printf("hugepage gain: seq x%.3f, chunk x%.3f  (chunk is the lane's shape)\n", seqT / seq4, chT / ch4);
    printf("sink %llu\n", (unsigned long long) sink);
    return 0;
}
