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
hp0=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo)
echo "shm-shadow: tier $DST_ROOT ($(stat -f -c %T "$DST_ROOT" 2>/dev/null || echo fs))"

need=$(du -sB1 "$SRC" | cut -f1)
if [ "$(stat -f -c %T "$DST_ROOT" 2>/dev/null)" = hugetlbfs ]; then
  # the dedicated mount: its size= cap is the bound (2 GiB headroom), and the copy runs only on
  # the FULL boot-armed pool - a partial runtime reservation (24369 of 45056 on 2026-11-07) must
  # stay free for the engine arena that shares the pool (store ~19000 + IQ2_XS arena 20164 + KV
  # shadow > a partial pool).  Exit 0 = planned deferral, the launcher falls to the /dev/shm tier.
  pool=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo)
  if [ "$pool" -lt 40000 ]; then
    echo "shm-shadow: pool ${pool} pages < 40000 (partial runtime reservation) - store stays on /dev/shm, the pool is the arena's; the boot-armed pool (sysctl.d 91-strata-hugepages) lifts this gate"
    exit 0
  fi
  head=2
else
  head=8
fi
avail=$(df -B1 --output=avail "$DST_ROOT" | tail -1 | tr -dc '0-9')
if [ "$avail" -lt "$((need + head * 1024 * 1024 * 1024))" ]; then
  echo "shm-shadow: $DST_ROOT has $((avail / 2**30)) GiB free, need $((need / 2**30)) GiB + ${head} GiB headroom - NOT copying"
  exit 1
fi
# fast path (the launcher now calls this on every start): an already-restored store with the source's
# file count and total size skips the copy and the verify walk entirely
fast_ok() {
  [ -f "$DST/experts.bin" ] || return 1
  local a b
  a=$(find "$SRC" -type f | wc -l); b=$(find "$DST" -type f | wc -l)
  [ "$a" = "$b" ] || return 1
  # stat-based: identical on tmpfs; on hugetlbfs the dest rounds UP to whole 2 MiB pages
  # (du misreports there, and exact-sum would never match) - accept src <= dest < src + one page
  local hf=0
  [ "$(stat -f -c %T "$DST_ROOT" 2>/dev/null)" = hugetlbfs ] && hf=1
  local f rel s1 s2
  while read -r f; do
    rel="${f#"$SRC"/}"
    s1=$(stat -c %s "$f"); s2=$(stat -c %s "$DST/$rel" 2>/dev/null || echo -1)
    if [ "$hf" = 1 ]; then
      { [ "$s2" -ge "$s1" ] && [ $((s2 - s1)) -lt $((2 * 1024 * 1024)) ]; } || return 1
    else
      [ "$s1" = "$s2" ] || return 1
    fi
  done < <(find "$SRC" -type f)
  return 0
}
if fast_ok; then
  echo "shm-shadow: already restored ($(du -sh "$DST" | cut -f1)) - nothing to do"
  exit 0
fi
mkdir -p "$DST"
# hugetlbfs has NO write(2) path (no .write_iter in the kernel: cp/rsync/dd/sendfile/copy_file_range
# all fail EINVAL/EXDEV - measured, references-round2 digest) and ftruncate demands 2 MiB alignment.
# Population there is ftruncate+mmap+memcpy, measured 15.16 GiB/s (the 38 GiB store in ~2.5 s);
# files round UP to whole 2 MiB pages (the tail padding is inert - the engine reads to blob offsets,
# never past st_size of the source) and the verifier below accepts dest >= src on this tier.
if [ "$(stat -f -c %T "$DST_ROOT" 2>/dev/null)" = hugetlbfs ]; then
  python3 - "$SRC" "$DST" <<'PY'
import mmap, os, sys
src_root, dst_root = sys.argv[1], sys.argv[2]
HP, CH = 2 << 20, 256 << 20
copied = 0
for dirpath, _dirs, files in os.walk(src_root):
    rel = os.path.relpath(dirpath, src_root)
    os.makedirs(os.path.join(dst_root, rel), exist_ok=True)
    for f in sorted(files):
        s, d = os.path.join(dirpath, f), os.path.join(dst_root, rel, f)
        sz = os.path.getsize(s)
        rounded = max(HP, (sz + HP - 1) // HP * HP)
        with open(s, "rb") as fs, open(d, "wb+") as fd:
            fd.truncate(rounded)
            md = mmap.mmap(fd.fileno(), rounded)
            if sz:
                ms = mmap.mmap(fs.fileno(), sz, prot=mmap.PROT_READ)
                for off in range(0, sz, CH):
                    n = min(CH, sz - off)
                    md[off:off + n] = ms[off:off + n]
                ms.close()
            md.flush()
            md.close()
        copied += sz
print(f"shm-shadow: hugetlbfs populate {copied / 2**30:.1f} GiB by mmap+memcpy")
PY
else
  rsync -a "$SRC/" "$DST/"
fi
# verify: every file in SRC exists in DST with the right size - identical on tmpfs, dest >= src
# (2 MiB page rounding) on hugetlbfs
hf=0
[ "$(stat -f -c %T "$DST_ROOT" 2>/dev/null)" = hugetlbfs ] && hf=1
bad=0
while read -r f; do
  rel="${f#"$SRC"/}"
  s1=$(stat -c %s "$f"); s2=$(stat -c %s "$DST/$rel" 2>/dev/null || echo -1)
  if [ "$hf" = 1 ]; then
    { [ "$s2" -ge "$s1" ] && [ $(( (s2 - s1) < 2 * 1024 * 1024 )) = 1 ]; } || { echo "shm-shadow: SIZE MISMATCH $rel ($s1 vs $s2)"; bad=1; }
  else
    [ "$s1" = "$s2" ] || { echo "shm-shadow: SIZE MISMATCH $rel ($s1 vs $s2)"; bad=1; }
  fi
done < <(find "$SRC" -type f)
[ "$(find "$SRC" -type f | wc -l)" = "$(find "$DST" -type f | wc -l)" ] || { echo "shm-shadow: FILE COUNT MISMATCH"; exit 1; }
[ "$bad" = 0 ] || exit 1

# The store's page class, by mount type.  hugetlbfs: pages are 2 MiB by construction - nothing to
# promote; the witness is the pool draw (HugePages_Free delta across the copy).  tmpfs (the
# /dev/shm tier, or a future kernel where shmem folios actually work): rsync's write() path
# faults 4K folios; MADV_COLLAPSE promotes where the box has supply - best-effort by design, a
# refusal leaves exactly the old 4K class.
if [ "$DST_ROOT" = /mnt/strata-huge ] && [ "$(stat -f -c %T "$DST_ROOT" 2>/dev/null)" = hugetlbfs ]; then
  hp1=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo)
  echo "shm-shadow: store on hugetlbfs - 2 MiB by construction (pool draw $((hp0 - hp1)) pages, ${hp1} free)"
elif [ "$DST_ROOT" = /mnt/strata-huge ]; then
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
