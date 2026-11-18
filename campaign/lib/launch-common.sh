#!/bin/bash
# campaign/lib/launch-common.sh — the ONE exec ladder + shadow-store ladder for every launcher.
# Sourced (never executed) by launch-custom-*.sh at the repo root.
#
# Why this exists (2026-11-09 refactor, the launcher-fleet audit): the five launchers carried the
# same three blocks by copy-paste and they DRIFTED - the taskset-propagation repair landed in one
# twin malformed (engine args dropped), the swap guard then landed in a different ORDER per launcher,
# and the two wrappers RACE: on iq3s, setting STRATA_ENGINE_CPUS exec'd taskset FIRST and silently
# bypassed the swap guard; on the other four, the default-on guard made the taskset branch dead
# code (the knob was silently ignored).  One ladder here composes them; the argv exists in exactly
# one place (launch::engine_argv) so no branch can retype it.
#
# STRATA_LAUNCH_DRY=1 prints the composed argv instead of exec'ing - the verification harness.
# LAUNCH LAW stays: the engine is started as `.venv/bin/python serve/server.py ...` resolved
# relative to the repo root (this lib cds there), never through a wrapper script.

LAUNCH_REPO="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"   # the CALLER launcher's dir = repo root
STRATA_HUGEPAGES_MIN=17000    # single-owner threshold: 31.64 GiB store = 16568 pages + headroom

launch::hugepages_free() {    # -> int: HugePages_Free from /proc/meminfo
  awk '/^HugePages_Free/{print $2}' /proc/meminfo
}

launch::engine_argv() {       # --config <file> [--port N] [--host H] -> the engine argv
  local config="" port=8081 host=0.0.0.0
  while [ $# -gt 0 ]; do
    case "$1" in
      --config) config="$2"; shift 2 ;;
      --port)   port="$2"; shift 2 ;;
      --host)   host="$2"; shift 2 ;;
      *) echo "launch::engine_argv: unknown arg $1" >&2; return 2 ;;
    esac
  done
  [ -n "$config" ] || { echo "launch::engine_argv: --config required" >&2; return 2; }
  # NOTE: tokens are space-free by repo convention; exec_engine word-splits this string, so
  # no embedded quoting (a quoted echo hands the exec a filename with literal " marks in it).
  echo ".venv/bin/python serve/server.py --engine strata --config $config --port $port --host $host"
}

launch::swap_guard_wanted() { # 0 = arm the systemd-run scope
  [ "${STRATA_SWAP_GUARD-1}" != "0" ] && command -v systemd-run >/dev/null 2>&1 && [ -n "${XDG_RUNTIME_DIR:-}" ]
}

launch::exec_engine() {       # --config <file> [--port N] [--host H]
  local argv; argv="$(launch::engine_argv "$@")" || return 2
  cd "$LAUNCH_REPO" || return 1
  if launch::swap_guard_wanted; then
    # Composed (the audit's D1 fix): the pin works UNDER the guard - neither wrapper is dropped.
    if [ -n "${STRATA_ENGINE_CPUS-}" ]; then
      [ -n "${STRATA_LAUNCH_DRY-}" ] && { echo "exec systemd-run --user --scope --collect -q -p MemorySwapMax=0 --unit=\"strata-engine-$$\" taskset -c \"$STRATA_ENGINE_CPUS\" $argv"; return 0; }
      exec systemd-run --user --scope --collect -q -p MemorySwapMax=0 --unit="strata-engine-$$" taskset -c "$STRATA_ENGINE_CPUS" $argv
    fi
    [ -n "${STRATA_LAUNCH_DRY-}" ] && { echo "exec systemd-run --user --scope --collect -q -p MemorySwapMax=0 --unit=\"strata-engine-$$\" $argv"; return 0; }
    exec systemd-run --user --scope --collect -q -p MemorySwapMax=0 --unit="strata-engine-$$" $argv
  fi
  if [ -n "${STRATA_ENGINE_CPUS-}" ]; then
    [ -n "${STRATA_LAUNCH_DRY-}" ] && { echo "exec taskset -c \"$STRATA_ENGINE_CPUS\" $argv"; return 0; }
    exec taskset -c "$STRATA_ENGINE_CPUS" $argv
  fi
  # plain exec: containers / non-systemd hosts (no guard there, serving works)
  [ -n "${STRATA_LAUNCH_DRY-}" ] && { echo "exec $argv"; return 0; }
  exec $argv
}

# ---- Qwen-family shadow store (DS4 launchers never call these) ----

launch::shadow_canon() {      # the store's one owner-path (packs live with the estate, like data/mtp)
  echo /mnt/DATA/LLM/Models/GGUF/strata/data/packs/q2_0
}

launch::shadow_export() {     # THE METHOD (2026-11-18 operator ruling): the canonical pack is the
  # primary lane's store - no ladder, no system-dir fallbacks. The pack's absence means no store
  # (native path serves every miss), never a degraded tier. The tmpfs fast tier (F-367, +5.5% ABAB)
  # is an explicit opt-in: STRATA_SHADOW_TMPFS=1 arms the RAM copy and lets it linger; the tier that
  # corrupted under memory pressure (M1: the pool-short boot fell to the swappable copy) is never
  # automatic. An explicit env value always wins - including the "" rollback (the engine treats
  # unset and empty alike - shadow_experts.hpp).
  local canon=$(launch::shadow_canon) tmpfs=/dev/shm/strata-shadow-q2_0
  if [ -n "${STRATA_SHADOW_EXPERTS+set}" ]; then
    :                                                 # explicit value wins (incl. "" rollback)
  elif [ "${STRATA_SHADOW_TMPFS:-0}" = "1" ]; then
    export STRATA_SHADOW_EXPERTS="$tmpfs"             # opt-in fast tier: RAM copy lingers; the
                                                      # reestablish below builds it before the exec
  elif [ -f "$canon/experts.bin" ]; then
    export STRATA_SHADOW_EXPERTS="$canon"             # THE method: the estate artifact (like data/mtp)
  else
    export STRATA_SHADOW_EXPERTS=""
    echo "launcher: shadow pack $canon MISSING - no store, native path serves every miss" >&2
  fi
  [ -n "$STRATA_SHADOW_EXPERTS" ] && echo "launcher: shadow store $STRATA_SHADOW_EXPERTS" >&2
}

launch::shadow_reestablish() {  # the tmpfs placement copy is OPT-IN (2026-11-18 operator ruling: the
  # primary lane has no fallbacks): default = no copy, no system-dir lingering. STRATA_SHADOW_TMPFS=1
  # rebuilds the F-367 fast tier (the copy lingers in RAM; the prewarm then skips it as "RAM already").
  # The old every-start copy is what left a 39.3 GiB unused store squatting /dev/shm through the
  # rollback. Repo-anchored (the original call was cwd-relative and silently skipped from other cwds).
  [ "${STRATA_SHADOW_TMPFS:-0}" = "1" ] || return 0
  if [ "$(launch::hugepages_free)" -lt "$STRATA_HUGEPAGES_MIN" ]; then
    bash "$LAUNCH_REPO/campaign/shm-shadow.sh" >/dev/null 2>&1 || true
  fi
}
