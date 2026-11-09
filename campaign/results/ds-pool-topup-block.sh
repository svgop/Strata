#!/bin/bash
# ds-pool-topup-block.sh — raise the huge-page pool 45076 -> 54272 WITHOUT a reboot (2026-11-09).
#
# OWNER OF RECORD: campaign/91-strata-hugepages.conf (deployed as /etc/sysctl.d/91-strata-hugepages.conf)
# SIZING LAW:      .docs/research/digest/2026-11-08-pool-sizing-tools.md  (54272 = DS all 110.85 GB
#                  = 52,860 pages + 2 GiB slack, round up 512 -> 54,272 = 106 GiB)
# REMEDY LAW:      .docs/research/digest/2026-11-07-preboot-ds-pool-sizing.md 5 + the sysctl conf header:
#                  engines down -> drop page cache -> compact_memory -> re-raise in steps -> READ BACK OR FAIL
#                  (DPDK discipline: the kernel may silently grant fewer pages; declared != actual is a FAIL).
#
# RUN AS ROOT, no sudo inside:   sudo bash campaign/results/ds-pool-topup-block.sh
#
# Why it matters today: HugePages_Free 45076 x 2 MiB = 94.53 GB < DS all 110.85 GB -> huge_arena OFF
# (ds4_experts_rule.hpp:173-177) AND the copy budget's pool term drops out (ds4_model.cu:4674),
# can = avail - 8 GiB < all -> all-or-none budget 0 -> DS copies NOTHING and pins NOTHING.
set -u

TARGET=54272            # pages, = 106 GiB (512-page granules, Oracle-style)
PASSES=3                # compact + raise, up to 3 passes
HP_SYSCTL=/proc/sys/vm/nr_hugepages

page_gib() { echo $(( $1 * 2 / 1024 )); }          # 2 MiB pages -> GiB
count()    { awk '/^HugePages_Total:/{print $2}' /proc/meminfo; }
freec()    { awk '/^HugePages_Free:/{print $2}'  /proc/meminfo; }

echo "== ds-pool-topup: target $TARGET pages ($(page_gib $TARGET) GiB), $PASSES passes =="

# ---- Gate 1: engines DOWN (port law: 8081 family alternates; a live engine holds pool pages
#      and would convert them to surplus on any later shrink) ----
if curl -s -m 2 http://127.0.0.1:8081/health >/dev/null 2>&1; then
    echo "FAIL: an engine answers on 8081 — stop it first (launch-custom-dsv41-8081.sh / iq3s occupant), then re-run." >&2
    exit 1
fi
if ss -ltn 2>/dev/null | grep -q ':8081 '; then
    echo "FAIL: port 8081 has a listener but /health is dark — kill it and re-run." >&2
    exit 1
fi
PIDS="$(pgrep -f 'serve/server\.py' 2>/dev/null; pgrep -x strata 2>/dev/null; pgrep -f '/build-fresh/strata' 2>/dev/null)"
if [ -n "${PIDS}" ]; then
    echo "FAIL: engine processes alive (strata / serve/server.py): ${PIDS}" >&2
    exit 1
fi
echo "engines: down (8081 free, no strata / serve/server.py processes)"

# ---- Gate 2: declared target still stands (drift check: the sysctl file owns the number) ----
grep -q "^vm.nr_hugepages=${TARGET}$" /etc/sysctl.d/91-strata-hugepages.conf \
    || echo "WARN: /etc/sysctl.d/91-strata-hugepages.conf does not declare $TARGET — top-up will not survive a reboot without it." >&2

BEFORE=$(count)
AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
echo "before: HugePages_Total=$BEFORE ($(page_gib $BEFORE) GiB), short $((TARGET-BEFORE)) pages ($(( (TARGET-BEFORE)*2/1024 )) GiB), MemAvailable ${AVAIL_KB} kB"

# ---- Step 3: drop the page cache (frees cache pages back to the buddy as free 2 MiB blocks) ----
sync
echo 1 > /proc/sys/vm/drop_caches
echo "page cache dropped (vm.drop_caches=1)"

# ---- Step 4: compact + raise + read-back, up to $PASSES passes ----
BEST=0
pass=1
while [ "$pass" -le "$PASSES" ]; do
    echo "== pass $pass/$PASSES: compact_memory (can take minutes on 184 GiB) =="
    echo 1 > /proc/sys/vm/compact_memory
    echo "$TARGET" > "$HP_SYSCTL"
    GOT=$(count)
    echo "pass $pass: asked $TARGET, got $GOT ($(page_gib $GOT) GiB), free $(freec)"
    [ "$GOT" -gt "$BEST" ] && BEST=$GOT
    [ "$GOT" -ge "$TARGET" ] && break
    pass=$((pass+1))
done

# ---- Step 5: read-back or fail (the DPDK message shape) ----
FINAL=$(count)
FREE=$(freec)
if [ "$FINAL" -ge "$TARGET" ]; then
    echo "OK: HugePages_Total=$FINAL / Free=$FREE ($(page_gib $FINAL) GiB pool)."
    echo "OK: pool_free = $(awk -v p=$FINAL 'BEGIN{printf "%.2f", p*2097152/1e9}') GB >= DS all 110.85 GB -> huge_arena arms at the next DS boot; pin arms with it (arena_all_huge)."
    echo "next: bash campaign/batteries/ds-verify.sh   (2 boots + witness grep; NOT run by this block)"
    exit 0
fi
echo "FAIL: got $FINAL pages instead of $TARGET (best pass $BEST; short $((TARGET-FINAL)) pages = $(( (TARGET-FINAL)*2/1024 )) GiB)" >&2
echo "FAIL: at $FINAL pages DS stays disarmed (pool_free < all 110.85 GB -> budget 0 -> no copy, no pin). The guaranteed arm is a REBOOT with the sysctl file in place (pristine buddy, M5 law)." >&2
exit 1
