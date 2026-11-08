#!/bin/bash
# campaign/shm-shadow.sh - F-367: re-establish the shadow store's /dev/shm placement cache.
# ENGINE-LEVEL NOTE (HOSTMEM-NO-REBOOT, 2026-11-08): when the reserved 2 MiB pool covers the store,
# the engine now builds it itself as a hugetlb-backed memfd at arm time (shadow_experts.cpp - the
# FlexKV pattern; no file, no mount, no service, released with the process).  This tmpfs copy is
# the FALLBACK tier for pool-short boots: the engine then maps this copy instead of the page-cached
# canonical pack (F-365: 2.6M minor faults per probe leg on the disk copy).  The earlier
# hugetlbfs-mount tier was removed - it double-drew the pool against the engine's memfd (caught
# live: 19225 pages held after engine death; the engine owns the store now).
set -u
SRC=/home/svg/LLM/data/packs/q2_0
T0=$(date +%s)
DST=/dev/shm/strata-shadow-q2_0

need=$(du -sB1 "$SRC" | cut -f1)
avail=$(df -B1 --output=avail /dev/shm | tail -1 | tr -dc '0-9')
if [ "$avail" -lt "$((need + 8 * 1024 * 1024 * 1024))" ]; then
  echo "shm-shadow: /dev/shm has $((avail / 2**30)) GiB free, need $((need / 2**30)) GiB + 8 GiB headroom - NOT copying"
  exit 1
fi
# fast path (the launcher now calls this on every start): an already-restored store with the source's
# file count and total size skips the copy and the verify walk entirely
fast_ok() {
  [ -f "$DST/experts.bin" ] || return 1
  local a b
  a=$(find "$SRC" -type f | wc -l); b=$(find "$DST" -type f | wc -l)
  [ "$a" = "$b" ] || return 1
  a=$(find "$SRC" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  b=$(find "$DST" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
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
echo "shm-shadow: verified $((need / 2**30)) GiB at $DST in $(( $(date +%s) - T0 ))s - restart the lane to pick it up"
