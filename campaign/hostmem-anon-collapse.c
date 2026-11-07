// campaign/hostmem-anon-collapse.c - the box's 2 MiB health gauge (M1/M3 battery witness).
// A 256 MiB anon mapping faulted 4K, then MADV_COLLAPSE: the promoted fraction measures whether
// the box can still supply PMD folios at this moment (the degradation condition of
// campaign/HOSTMEM-NO-REBOOT.md; ENOMEM/0 kB promoted = the aged-box state).
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

static long rollup(const char *key) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, strlen(key))) { sscanf(line + strlen(key), "%ld", &kb); break; }
    fclose(f);
    return kb;
}

int main(void) {
    const size_t SZ = 256u << 20;
    char *m = mmap(NULL, SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    for (size_t o = 0; o < SZ; o += 4096) m[o] = 1;
    printf("gauge: faulted-4K AnonHugePages %ld kB; ", rollup("AnonHugePages:"));
    errno = 0;
    int rc = madvise(m, SZ, MADV_COLLAPSE);
    printf("collapse rc=%d errno=%d (%s) -> %ld kB of %u kB (%.0f%%)\n",
           rc, errno, rc ? strerror(errno) : "ok", rollup("AnonHugePages:"),
           (unsigned)(SZ >> 10), 100.0 * (rollup("AnonHugePages:") < 0 ? 0 : rollup("AnonHugePages:")) / (SZ >> 10));
    return 0;
}
