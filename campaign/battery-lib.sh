#!/bin/bash
# campaign/battery-lib.sh — the shared battery infrastructure (source, never execute).
#
# The 31-battery fleet + the two ladders carried ~1,040 lines of this as copy-paste (26
# cleanup_engine() blocks were byte-identical), and the copies drifted three ways that bit us:
#   - the GPU kill sweep matched the SUBSTRING "strata" in any compute process's cmdline, which
#     killed whichever engine the OTHER lane had up (both launchers exec "server.py --engine
#     strata"; both engine binaries live in this repo) — 27 sites wide;
#   - the warm-in: 21 batteries still called campaign/prewarm2.sh and t64 called prewarm-ds4.sh
#     after those scripts were absorbed into the server itself (commit cc05319f) — a silent
#     no-op, so a battery could measure legs while the native warm-in was still replaying;
#   - the health-wait patience and the final-serve restore differed per copy for no reason.
#
# What belongs HERE is the machinery every boot pays; what stays in each battery is its
# experiment (arms, leg schedule, labels, probe flags).  Source it:
#     cd campaign && . ./battery-lib.sh                       # a ladder or a campaign/ script
#     . "$(dirname "$0")/../battery-lib.sh"                   # a battery in campaign/batteries/
# PORT must already be set by the caller (the fleet is 8081); REPO defaults to this checkout.

REPO=${REPO:-/home/svg/LLM/strata-custom-2}
PORT=${PORT:-8081}

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }

cleanup_engine() {
  # Port-scoped TERM first — the server owns its engine process and closes it in order.  The GPU
  # sweep that follows is scoped to THIS repo's engine binaries only: never a bare "strata"
  # substring (it matched both engines' cmdlines and any in-repo parity bench), but the actual
  # paths this checkout launches (engine/ for the Qwen lane, build-fresh/ for the DS lane).
  local p pid c
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null)
    case "$c" in *"$REPO/engine/"*|*"$REPO/build-fresh/"*) kill -TERM "$pid" 2>/dev/null ;; esac
  done
  # then the port itself must go free — a boot racing the old listener's socket release is the
  # port-8081 race every long session has paid at least once (up to 30 s, then through anyway)
  local t
  for t in $(seq 1 30); do
    [ -z "$(port_pid)" ] && break
    sleep 1
  done
}

wait_health() {  # wait_health [tries-of-3s] — any /health body means the API answers
  local t
  for t in $(seq 1 "${1:-120}"); do
    [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ] && return 0
    sleep 3
  done
  return 1
}

wait_prewarm() {  # wait_prewarm [tries-of-5s] — the server's own warm-in must reach a terminal
  # state before any leg runs (campaign law 11: warmup is part of the bench).  "done" it warmed;
  # "failed"/"yielded" it ended on its own terms (live traffic pre-empted it); "" the config has
  # no prewarm section — all three are terminal.  Non-terminal: idle/files/replay.
  local t s
  for t in $(seq 1 "${1:-240}"); do
    s=$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | sed -n 's/.*"prewarm": "\([a-z]*\)".*/\1/p')
    case "$s" in done|failed|yielded|"") return 0 ;; esac
    sleep 5
  done
  echo "wait_prewarm: no terminal warm state in 20 min (continuing)"
}

final_serve() {  # final_serve <launcher> <out-log> — leave the standing server up for the operator
  ( cd "$REPO" && nohup bash "$1" >"$2" 2>&1 & )
  wait_health || echo "final_serve: the server did not answer /health in 6 min"
}
