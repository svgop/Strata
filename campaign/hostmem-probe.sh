#!/bin/bash
# campaign/hostmem-probe.sh - M1 feasibility probe (ROOT, ~30 s, leaves nothing behind).
# The userns test of tmpfs huge=within_size was inconclusive (a user namespace charges folios
# differently), so this answers on the real kernel:
#   (a) does a fresh ftruncate+write-fault file on a huge=<policy> tmpfs get PMD folios directly?
#   (b) does MADV_COLLAPSE promote a pwrite()-populated (4K) file, and with what errno if not?
# tested under huge=within_size AND huge=always, plus the anon-collapse gauge (the box's 2 MiB
# health right now).  Run:  sudo bash campaign/hostmem-probe.sh
set -u
P=/mnt/strata-huge-probe
TMP=$(mktemp -d)
trap 'umount "$P" 2>/dev/null; rmdir "$P" 2>/dev/null; rm -rf "$TMP"' EXIT
mkdir -p "$P"

gcc -O2 -o "$TMP/probe" -xc - <<'C' || { echo "compile fail"; exit 1; }
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
static long file_pmd(const void *p) {
    FILE *f = fopen("/proc/self/smaps", "r");
    char line[256], want[64];
    unsigned long addr = (unsigned long) p;
    snprintf(want, sizeof want, "%lx-", addr >> 12 << 12);
    long kb = -1; int in = 0;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, want, strlen(want))) in = 1;
        else if (in && (line[0] < '0' || line[0] > 'f') && line[0] != ' ') break;
        if (in && !strncmp(line, "FilePmdMapped:", 14)) { sscanf(line + 14, "%ld", &kb); break; }
    }
    fclose(f);
    return kb;
}
static long rollup(const char *key) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, strlen(key))) { sscanf(line + strlen(key), "%ld", &kb); break; }
    fclose(f); return kb;
}
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--anon")) {          // the box gauge
        const size_t SZ = 256u << 20;
        char *m = mmap(NULL, SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        for (size_t o = 0; o < SZ; o += 4096) m[o] = 1;
        printf("anon gauge: faulted-4K AnonHugePages %ld kB; ", rollup("AnonHugePages:"));
        errno = 0;
        int rc = madvise(m, SZ, MADV_COLLAPSE);
        printf("collapse rc=%d errno=%d (%s) -> %ld kB of %u kB\n",
               rc, errno, rc ? strerror(errno) : "ok", rollup("AnonHugePages:"), (unsigned)(SZ >> 10));
        return 0;
    }
    const size_t SZ = 128u << 20;                          // 64 PMD folios
    const char *dir = argv[1];
    char path[512];
    // (a) fresh file, ftruncate, write-fault through the mapping
    snprintf(path, sizeof path, "%s/a.bin", dir);
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    ftruncate(fd, SZ);
    char *m = mmap(NULL, SZ, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    for (size_t o = 0; o < SZ; o += 4096) m[o] = (char) o;
    printf("(a) write-fault fresh file: FilePmdMapped %ld kB of %u kB -> %s\n",
           file_pmd(m), (unsigned)(SZ >> 10), file_pmd(m) > 0 ? "PASS" : "FAIL");
    munmap(m, SZ); close(fd); unlink(path);
    // (b) shell-populated (write() = 4K folios), then MADV_COLLAPSE
    snprintf(path, sizeof path, "%s/b.bin", dir);
    fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    char buf[4096]; memset(buf, 1, sizeof buf);
    for (size_t o = 0; o < SZ; o += 4096) pwrite(fd, buf, sizeof buf, o);
    m = mmap(NULL, SZ, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    errno = 0;
    int rc = madvise(m, SZ, MADV_COLLAPSE);
    printf("(b) pwrite-populated + MADV_COLLAPSE rc=%d errno=%d (%s): FilePmdMapped %ld kB of %u kB -> %s\n",
           rc, errno, rc ? strerror(errno) : "ok", file_pmd(m), (unsigned)(SZ >> 10), file_pmd(m) > 0 ? "PASS" : "FAIL");
    munmap(m, SZ); close(fd); unlink(path);
    return 0;
}
C

for pol in within_size always; do
  umount "$P" 2>/dev/null
  if mount -t tmpfs -o size=512M,huge=$pol strata-probe "$P"; then
    echo "== tmpfs huge=$pol =="
    "$TMP/probe" "$P"
  else
    echo "== tmpfs huge=$pol: MOUNT REFUSED =="
  fi
done
echo "== anon gauge (this box, now) =="
"$TMP/probe" --anon
echo "free order-9+ blocks: $(python3 - <<'PY'
t = 0
for line in open('/proc/buddyinfo'):
    parts = line.split()
    if 'zone' not in line: continue
    counts = [int(x) for x in parts[4:]]
    t += sum(c * (4096 << o) for o, c in enumerate(counts) if o >= 9)
print(f"{t >> 30} GiB")
PY
)"
