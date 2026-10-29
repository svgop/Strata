#!/bin/bash
# campaign/shm-shadow.sh - F-367: re-establish the tmpfs shadow store after reboot.
# /dev/shm is tmpfs: RAM-resident, THP-eligible, never evicted (the F-365 fix - 2,635,199 minor
# faults per probe leg on the disk page cache), and WIPED on reboot. Run this once after a reboot;
# until it has, the launcher's guard falls back to the disk path so the lane still boots.
set -u
SRC=/home/svg/LLM/data/packs/q2_0
DST=/dev/shm/strata-shadow-q2_0
T0=$(date +%s)

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
echo "shm-shadow: verified $((need / 2**30)) GiB at $DST in $(( $(date +%s) - T0 ))s - restart the lane to pick it up"
