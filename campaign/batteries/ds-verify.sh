#!/bin/bash
# ds-verify.sh — DS huge-arena + pin verification battery (SKELETON, do not run until the pool is armed).
#
# PRECONDITION: campaign/results/ds-pool-topup-block.sh has completed OK (HugePages_Total >= 54272).
#               DS and Qwen cannot coexist in RAM (port law, 2026-11-07 digest §3.3) — stop any 8081
#               occupant first; this script refuses to run while the pool is short.
# LAW:          .docs/research/digest/2026-11-07-preboot-ds-pool-sizing.md §5 (expected witnesses 1-4)
#               + 2026-11-08-pool-sizing-tools.md §Outputs 3 (read-back or fail).
# CODE ANCHORS: src/dsv41/ds4_model.cu:4670-4672 (auto-ON line), :4683 (all-or-none FAIL line),
#               :4745 (pool-ran-out FAIL line), :4846-4852 (pin line), ds4_experts_rule.hpp:166-177.
#
# TWO BOOTS: boot 1 lights the mechanisms; boot 2 proves "engines stop/start return whole pages —
# restart is a perfect re-arm" (sysctl conf header) and that the pin relights on a fresh arena.
# NOT RUN — the pool is 45076/54272 today; DS needs the full pool first.
set -u

REPO=/home/svg/LLM/strata-custom-2
LAUNCH="$REPO/launch-custom-dsv41-8081.sh"
LOG="$REPO/logs/custom-dsv41-8081.log"
OUT="$REPO/campaign/results/ds-verify-$(date +%Y%m%d-%H%M).txt"
TARGET=54272
HEALTH_TIMEOUT=900        # s; DS load copies ~110 GB — allow 15 min
BOOT=0
FAILED=0

total() { awk '/^HugePages_Total:/{print $2}' /proc/meminfo; }
free_p() { awk '/^HugePages_Free:/{print $2}' /proc/meminfo; }
die() { echo "FAIL: $*" >&2; echo "FAIL: $*" >> "$OUT"; exit 1; }
note() { echo "$*" | tee -a "$OUT"; }

# ---- Guard: refuse to run while the pool is short (DS boots would land in-place and the bank lies) ----
[ "$(total)" -ge "$TARGET" ] || die "pool is $(total)/$TARGET — run campaign/results/ds-pool-topup-block.sh (or reboot) first"
curl -s -m 2 http://127.0.0.1:8081/health >/dev/null 2>&1 && die "an engine answers on 8081 — stop it first (no DS+Qwen coexistence)"

note "== ds-verify $(date): pool Total=$(total) Free=$(free_p) =="

boot_one() {
    BOOT=$((BOOT+1))
    local off fre_before fre_after delta
    fre_before=$(free_p)
    off=$(( $(wc -l < "$LOG") ))
    note "--- boot $BOOT: launch (log offset $off, HugePages_Free before $fre_before) ---"
    ( cd "$REPO" && nohup "$LAUNCH" >> "$OUT" 2>&1 & )
    local t=0
    until curl -s -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1; do
        sleep 5; t=$((t+5))
        [ "$t" -ge "$HEALTH_TIMEOUT" ] && die "boot $BOOT: no /health in ${HEALTH_TIMEOUT}s"
    done
    fre_after=$(free_p)
    delta=$(( fre_before - fre_after ))
    note "boot $BOOT: HugePages_Free $fre_before -> $fre_after (delta $delta pages = $(awk -v d=$delta 'BEGIN{printf "%.1f", d*2097152/1e9}') GB)"

    # Witness 1 — auto-ON line (ds4_model.cu:4670-4672): pool holds the whole copy, arena on 2 MB pages.
    tail -n +"$off" "$LOG" | grep -aE "the reserved huge-page pool \([0-9.]+ GB free\) holds the whole copy \([0-9.]+ GB\): the arena on 2 MB pages" >> "$OUT" \
        || { FAILED=1; note "boot $BOOT MISS W1: no auto-ON line (huge_arena did not arm — pool short at load?)"; }
    # Witness 2 — the copy happened, all-or-none cleared: "ds4: --mmap-experts auto: ... fit the RAM ... copied in at load" + "N of ...N experts copied into RAM (110.x GB ...)".
    tail -n +"$off" "$LOG" | grep -aE "fit the RAM this process can get .*copied in at load|[0-9]+ of the CPU lane's [0-9]+ experts copied into RAM \(1[01][0-9]\.[0-9] GB" >> "$OUT" \
        || { FAILED=1; note "boot $BOOT MISS W2: no copy lines (budget 0 — check for the :4683 fail line)"; }
    # Witness 3 — the pin (ds4_model.cu:4846-4852): "the copied experts pinned in X.X s (read-write|read-only): the prompt path DMAs them in place".
    tail -n +"$off" "$LOG" | grep -aE "the copied experts pinned in [0-9.]+ s \((read-write|read-only)\): the prompt path DMAs them in place" >> "$OUT" \
        || { FAILED=1; note "boot $BOOT MISS W3: no pin line (arena_all_huge false — partial pool ate the arena mid-load?)"; }
    # Pool delta — expect ~52,900-53,900 pages consumed (all 110.85 GB / 2 MiB + per-layer rounding + spares), Free left ~400-1,400.
    [ "$delta" -ge 52000 ] && [ "$delta" -le 53900 ] || { FAILED=1; note "boot $BOOT MISS W4: HugePages_Free delta $delta outside 52000-53900 (arena not wholly from the pool)"; }

    # Fail lines — ANY of these means the pool did not hold (2026-11-07 digest §5 fail lines).
    for pat in "the 2 MB huge-page arena was asked for but the pool ran out" "do not all fit the RAM this process can get" "pinning the copied experts failed" "no RAM for layer"; do
        tail -n +"$off" "$LOG" | grep -aF "$pat" >> "$OUT" && { FAILED=1; note "boot $BOOT FAIL LINE present: $pat"; }
    done

    note "--- boot $BOOT: stop, expect perfect re-arm (Free back to $(total)) ---"
    pkill -f 'serve/server\.py' 2>/dev/null; pkill -x strata 2>/dev/null; pkill -f '/build-fresh/strata' 2>/dev/null
    sleep 5
    local fre_stop
    fre_stop=$(free_p)
    [ "$fre_stop" -ge "$TARGET" ] || { FAILED=1; note "boot $BOOT RE-ARM FAIL: HugePages_Free $fre_stop < $TARGET after stop (pages not returned whole — investigate before boot 2)"; }
    note "boot $BOOT: after stop Free=$fre_stop"
}

boot_one
boot_one

if [ "$FAILED" -eq 0 ]; then
    note "== ds-verify PASS: both boots lit W1+W2+W3, pool delta in band, re-arm whole after each stop =="
    exit 0
fi
note "== ds-verify FAIL: see MISS/FAIL lines above; pool verdict: Total=$(total) Free=$(free_p) =="
exit 1
