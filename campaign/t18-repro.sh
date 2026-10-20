#!/bin/bash
# campaign/t18-repro.sh — T18 SIGSEGV repro harness (the CPU-pool crash under 6-stream load).
#
# Fires repeated probe-shaped batches (one single + one 6-concurrent thinking-budget batch,
# fresh prompts each round, so every round prefills) against the STANDING launcher, and
# detects the engine death the T18 knob legs caught: SIGSEGV in strata::kernels::cpu::vnni::
# iq4nl_nt with multiple pool threads at one IP, fault addrs 0x2/0x3/0x5/0xa, the server
# printing "stopped unexpectedly (exit code -11)".
#
# Usage:
#   campaign/t18-repro.sh --label pre    --max-iters 60     # pre-fix hunt (default)
#   campaign/t18-repro.sh --label fixed  --max-iters 60     # post-fix proof
#   campaign/t18-repro.sh --label fixed  --max-iters 60 --resume-iter 0   (dry helper)
#
# One FRESH BOOT per invocation (campaign law). On a crash it captures server output,
# journalctl segfault lines, the newest coredumpctl entry and the engine-log tail into
# campaign/results/t18-crashes/, waits for the server's auto-restart, and keeps counting.
# ALWAYS leaves the box stopped (no engine listening).
set -u
REPO=/home/svg/LLM/strata-custom-2
LAUNCH=$REPO/launch-custom-iq3s-8081.sh
RES=$REPO/campaign/results
CRASHDIR=$RES/t18-crashes
PORT=8081
LABEL=pre
MAXIT=60
SHAPE=light
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL=$2; shift 2;;
    --max-iters) MAXIT=$2; shift 2;;
    --shape) SHAPE=$2; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
mkdir -p "$CRASHDIR"
SUMMARY=$RES/t18-$LABEL-summary.txt
echo "T18 REPRO ($LABEL) start $(date -Is) max_iters=$MAXIT" | tee "$SUMMARY"

port_pid() { ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1; }
engine_pids() { pgrep -f "$REPO/engine/strata --serve" 2>/dev/null; }
segfault_lines() { journalctl -k --since "$1" 2>/dev/null | grep -i "segfault" | grep -v grep; }

cleanup_all() {
  local p pid
  p=$(port_pid); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  sleep 4
  for pid in $(engine_pids); do kill -TERM "$pid" 2>/dev/null; done
  sleep 3
  for pid in $(engine_pids); do kill -KILL "$pid" 2>/dev/null; done
  pkill -KILL -f "serve/server.py --engine strata" 2>/dev/null
  sleep 1
}
trap cleanup_all EXIT

cleanup_all
sleep 2
SERVER_OUT=$RES/t18-$LABEL-server.out
: > "$SERVER_OUT"
( cd "$REPO" && nohup "$LAUNCH" > "$SERVER_OUT" 2>&1 & )
echo "booting standing launcher (server out: $SERVER_OUT)" | tee -a "$SUMMARY"
ok=0
for t in $(seq 1 120); do
  if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then ok=1; break; fi
  sleep 3
done
if [ "$ok" != 1 ]; then echo "BOOT FAILED" | tee -a "$SUMMARY"; tail -30 "$SERVER_OUT" | tee -a "$SUMMARY"; exit 1; fi
echo "boot healthy after ~$((t*3))s $(date -Is)" | tee -a "$SUMMARY"

crashes=0
it=1
while [ "$it" -le "$MAXIT" ]; do
  sleep 2
  T0=$(date -Is)
  J0=$(date -d "$T0" +%s 2>/dev/null || echo 0)
  SEGF0=$(segfault_lines "@$((J0 - 1))" | wc -l)
  EPID0=$(engine_pids | head -1)
  OUT=$RES/t18-$LABEL-iter$(printf %03d "$it").txt
  # One round: --shape light = 1 single + 6 concurrent thinking streams (fresh prompts).
  #            --shape probe = the probe-lane lane sequence in miniature: M1(2000) + pair(2x900)
  #                            + conc6(600) + 6 fresh short prefills - the shape the original
  #                            sightings died in (full probe legs).
  PORT="$PORT" ITER="$it" SHAPE="$SHAPE" python3 - <<'PYEOF' > "$OUT" 2>&1
import json, os, sys, threading, time, urllib.request
port = os.environ["PORT"]; it = int(os.environ["ITER"]); shape = os.environ["SHAPE"]
key = "3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8"
base = f"http://127.0.0.1:{port}/v1/chat/completions"
SEEDS = [
 ("Write a technical post-mortem of a {i}th-century bridge failure, covering the materials, the loads, the inspection gaps, and what modern engineers changed afterwards.", 900),
 ("Explain how a {i}-engine aircraft's systems are designed for redundancy: hydraulics, electrics, flight controls, and the certification tests that prove them.", 850),
 ("Give a detailed walkthrough of {i} classic distributed-systems failure modes (partitions, clocks, retries, cascades) with concrete protocols and mitigations.", 880),
 ("Analyse the logistics of supplying a {i}-month polar research station: power, food, fuel, communications, and the failure modes of each supply line.", 860),
 ("Describe how a city of {i} million people keeps its water safe: sources, treatment stages, distribution monitoring, and contamination response drills.", 870),
 ("Walk through the design of a {i}-storey timber high-rise: fire engineering, structural joins, moisture control, and the code changes that allowed it.", 840),
 ("Write a rigorous comparison of {i} database replication strategies (sync, semi-sync, group replication, CRDTs) with their failure and recovery behaviour.", 890),
 ("Explain the full chain of a {i}-day weather forecast: observation, assimilation, the physics core, ensembles, and where the skill is lost.", 830),
]
FRESH_TMPL = ("Section {i} of the incident review. At step {i} the operator observed a distinct change in "
              "behaviour and recorded it against the runbook entry number {i}. The measurement taken at "
              "that point was catalogued with identifier {i} and compared against the baseline that had "
              "been established before the change was introduced.")
def probe(prompt, max_tok):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tok, "temperature": 0.7}).encode()
    req = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json"})
    req.add_header("Authorization", "Bearer " + key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1500) as r:
            d = json.load(r)
        return True, d["usage"]["completion_tokens"], time.time() - t0
    except Exception as e:
        print(f"  req ERROR: {type(e).__name__}: {e}", flush=True)
        return False, 0, time.time() - t0
ok, tok, dt = probe(SEEDS[it % len(SEEDS)][0].format(i=it), SEEDS[it % len(SEEDS)][1])
print(f"single: ok={ok} tok={tok} {dt:.1f}s", flush=True)
def run_batch(n, max_tok, seed_off):
    out = []
    def work(i):
        p, mt = SEEDS[(seed_off + i) % len(SEEDS)]
        out.append(probe(p.format(i=f"{it}x{i}"), max_tok))
    ths = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    agg = sum(o[1] for o in out); nerr = sum(0 if o[0] else 1 for o in out)
    print(f"batch{n}/{max_tok}: agg_tok={agg} wall={wall:.1f}s agg_rate={agg/max(wall,0.001):.1f} t/s errors={nerr}", flush=True)
    return nerr
errs = 0
if shape == "probe":
    p, mt = SEEDS[it % len(SEEDS)]
    ok, tok, dt = probe(p.format(i=it), 2000)
    print(f"m1: ok={ok} tok={tok} {dt:.1f}s", flush=True)
    errs += run_batch(2, 900, it)
    errs += run_batch(6, 600, it * 7)
    for j in range(6):   # fresh short prefills (the prompt-churn lane)
        p2 = FRESH_TMPL.format(i=it * 100 + j)
        ok, tok, dt = probe(p2, 120)
        errs += 0 if ok else 1
    print(f"fresh6 done (errors so far {errs})", flush=True)
else:
    p, mt = SEEDS[it % len(SEEDS)]
    ok, tok, dt = probe(p.format(i=it), mt)
    print(f"single: ok={ok} tok={tok} {dt:.1f}s", flush=True)
    errs += run_batch(6, 700, it * 7)
sys.exit(1 if errs > 0 else 0)
PYEOF
  RC=$?
  sleep 3
  J1=$(date +%s)
  SEGF1=$(segfault_lines "@$J1" | wc -l)
  EPID1=$(engine_pids | head -1)
  DEATH=$(grep -c "stopped unexpectedly" "$SERVER_OUT" 2>/dev/null || true)
  CRASH=""
  if [ "$RC" != 0 ]; then CRASH="client-errors"; fi
  if [ -n "$EPID0" ] && [ -z "$EPID1" ]; then CRASH="$CRASH engine-gone"; fi
  if [ -n "$EPID0" ] && [ -n "$EPID1" ] && [ "$EPID0" != "$EPID1" ]; then CRASH="$CRASH engine-pid-changed($EPID0->$EPID1)"; fi
  if [ "${SEGF1:-0}" -gt "${SEGF0:-0}" ]; then CRASH="$CRASH kernel-segfault"; fi
  if [ "$DEATH" != "${DEATH0:-0}" ]; then CRASH="$CRASH server-death-note"; fi
  if [ "$RC" = 0 ] && [ -z "$CRASH" ]; then
    echo "iter $it OK  $(tail -1 "$OUT")" | tee -a "$SUMMARY"
  else
    crashes=$((crashes + 1))
    STAMP=$(date +%Y%m%d-%H%M%S)
    CDIR=$CRASHDIR/t18-$LABEL-iter$(printf %03d "$it")-$STAMP
    mkdir -p "$CDIR"
    cp "$OUT" "$CDIR/client.txt"
    cp "$SERVER_OUT" "$CDIR/server-tail.out" 2>/dev/null
    segfault_lines "@$((J0 - 1))" > "$CDIR/journal-segfault.txt" 2>&1
    coredumpctl info 2>/dev/null | tail -80 > "$CDIR/coredumpctl-info.txt"
    tail -400 "$REPO/custom-iq3_s.log" > "$CDIR/engine-log-tail.txt" 2>/dev/null
    tail -120 "$SERVER_OUT" > "$CDIR/server-last120.txt" 2>/dev/null
    echo "iter $it CRASH ($CRASH) -> $CDIR" | tee -a "$SUMMARY"
    echo "iter $it CRASH flags:$CRASH rc=$RC $(tail -2 "$OUT" | tr '\n' ' ')" | tee -a "$SUMMARY"
    # wait out the server's auto-restart so the next round starts healthy
    sleep 5
    rok=0
    for t in $(seq 1 120); do
      if [ -n "$(curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null)" ]; then rok=1; break; fi
      sleep 3
    done
    echo "  post-crash health=$rok (waited ~$((t*3))s)" | tee -a "$SUMMARY"
  fi
  DEATH0=$DEATH
  it=$((it + 1))
done

echo "T18 REPRO ($LABEL) end $(date -Is): iters=$((it - 1)) crashes=$crashes rate=$(python3 -c "print(f'{100.0*$crashes/max(1,$it-1):.1f}')")%" | tee -a "$SUMMARY"
cleanup_all
trap - EXIT
sleep 2
if [ -n "$(port_pid)" ] || [ -n "$(engine_pids)" ]; then echo "WARNING: something still listening" | tee -a "$SUMMARY"; else echo "box stopped (no engine listening) $(date -Is)" | tee -a "$SUMMARY"; fi
