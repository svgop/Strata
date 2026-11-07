#!/bin/bash
# campaign/shm-shadow.sh - F-367/F-381: re-establish the shadow store's placement cache.
# M1 (campaign/HOSTMEM-NO-REBOOT.md, 2026-11-07): the preferred home is the huge-folio tmpfs
# /mnt/strata-huge (strata-huge-tmpfs.service, huge=within_size) - 2 MiB shmem folios where the
# kernel grants them.  /dev/shm never had one on this box (shmem_enabled=never; the F-365
# "THP-eligible" note was wrong here), which made the 38 GiB store one of the three 4K organs.
# No mount -> /dev/shm exactly as before.  Both are wiped on reboot; this script rebuilds the
# copy (12 s) and the launchers pick the best tier at start; until it has run, their guard falls
# back to the disk path so the lane still boots.
set -u
SRC=/home/svg/LLM/data/packs/q2_0
T0=$(date +%s)

DST_ROOT=/dev/shm
if mountpoint -q /mnt/strata-huge 2>/dev/null; then
  DST_ROOT=/mnt/strata-huge
fi
DST=$DST_ROOT/strata-shadow-q2_0
echo "shm-shadow: tier $DST_ROOT$([ "$DST_ROOT" = /mnt/strata-huge ] && echo ' (huge=within_size)')"

need=$(du -sB1 "$SRC" | cut -f1)
avail=$(df -B1 --output=avail "$DST_ROOT" | tail -1 | tr -dc '0-9')
if [ "$avail" -lt "$((need + 8 * 1024 * 1024 * 1024))" ]; then
  echo "shm-shadow: $DST_ROOT has $((avail / 2**30)) GiB free, need $((need / 2**30)) GiB + 8 GiB headroom - NOT copying"
  exit 1
fi
# fast path (the launcher now calls this on every start): an already-restored store with the source's
# file count and total size skips the copy and the verify walk entirely
fast_ok() {
  [ -f "$DST/experts.bin" ] || return 1
  local a b
  a=$(find "$SRC" -type f | wc -l); b=$(find "$DST" -type f | wc -l)
  [ "$a" = "$b" ] || return 1
  a=$(du -sB1 "$SRC" | cut -f1); b=$(du -sB1 "$DST" | cut -f1)
  [ "$a" = "$b" ] || return 1
  return 0
}
if fast_ok; then
  echo "shm-shadow: already restored ($(du -sh "$DST" | cut -f1)) - nothing to do"
  exit 0
fi
mkdir -p "$DST"
rsync -a "$SRC/" "$DST/"
# verify: every file in SRC exists in DST with identical size
bad=0
while read -r f; do
  rel="${f#"$SRC"/}"
  s1=$(stat -c %s "$f"); s2=$(stat -c %s "$DST/$rel" 2>/dev/null || echo -1)
  [ "$s1" = "$s2" ] || { echo "shm-shadow: SIZE MISMATCH $rel ($s1 vs $s2)"; bad=1; }
done < <(find "$SRC" -type f)
[ "$(find "$SRC" -type f | wc -l)" = "$(find "$DST" -type f | wc -l)" ] || { echo "shm-shadow: FILE COUNT MISMATCH"; exit 1; }
[ "$bad" = 0 ] || exit 1

# M1: rsync's write() path faults 4K folios even on the huge mount; MADV_COLLAPSE (kernel 6.1+)
# promotes them to 2 MiB where the box has supply.  Best-effort by design: a refusal leaves the
# 4K store (exactly the old class) and the engine maps whatever is in the page cache either way.
if [ "$DST_ROOT" = /mnt/strata-huge ]; then
  pmd0=$(awk '/^ShmemPmdMapped:/{print $2}' /proc/meminfo)
  rc=0
  python3 - "$DST/experts.bin" <<'PY' 2>/dev/null || rc=$?
import ctypes, mmap, os, sys
path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "rb+") as f:                       # PROT_READ|PROT_WRITE, MAP_SHARED
    m = mmap.mmap(f.fileno(), size)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), 25)   # 25 = MADV_COLLAPSE
    sys.exit(rc)
PY
  pmd1=$(awk '/^ShmemPmdMapped:/{print $2}' /proc/meminfo)
  echo "shm-shadow: MADV_COLLAPSE pass rc=$rc: ShmemPmdMapped ${pmd0} -> ${pmd1} kB"
fi
echo "shm-shadow: verified $((need / 2**30)) GiB at $DST in $(( $(date +%s) - T0 ))s - restart the lane to pick it up"
