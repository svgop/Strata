#!/bin/bash
# campaign/upstream/test-window.sh — the v0.2.4.2 adoption test window (F-329):
# stock-MTP boot (3 legs) -> stock+DFlash2 boot (3 legs) -> custom server restored ALWAYS.
# Single-engine law: takes 8081 down for the window. Evidence: campaign/upstream/*.out
set -u
cd /home/svg/LLM/strata-custom-2
KEY=3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8
restore_custom() {
  (nohup bash launch-custom-iq3s-8081.sh > campaign/upstream/restore-custom.out 2>&1 &)
  for i in $(seq 1 60); do
    sleep 10
    curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -q 200 && { echo "CUSTOM SERVER RESTORED"; return 0; }
  done
  echo "RESTORE FAILED — check campaign/upstream/restore-custom.out"; return 1
}
trap 'restore_custom' EXIT
server_pid() { ss -ltnp 2>/dev/null | grep ':8081 ' | grep -oP 'pid=\K[0-9]+' | head -1; }
PID=$(server_pid)
[ -n "$PID" ] && { echo "stopping custom server (pid $PID)"; kill "$PID"; sleep 6; }
boot() {
  .venv/bin/python serve/server.py --engine strata --config "$1" --port 8081 --host 0.0.0.0 > "$2" 2>&1 &
  SPID=$!
  for i in $(seq 1 45); do
    sleep 10
    curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -q 200 && { echo "up after ~$((i*10))s"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "BOOT DIED — log:"; tail -5 "$2"; return 1; }
  done
  echo "TIMEOUT"; return 1
}
for CFG in s1-mtp s2-dflash2; do
  echo "=== BOOT $CFG $(date +%H:%M:%S) ===" | tee -a campaign/upstream/$CFG.out
  if boot campaign/upstream/$CFG.json campaign/upstream/boot-$CFG.out 2>&1 | tee -a campaign/upstream/$CFG-clean.out; then
    for L in 1 2; do
      echo "--- $CFG leg $L $(date +%H:%M:%S) ---" >> campaign/upstream/$CFG-clean.out
      timeout 300 .venv/bin/python campaign/probe-lane.py 2>&1 | grep -E 'M1 single|M2 pair|M2 6-conc|M3b rate' >> campaign/upstream/$CFG-clean.out
    done
    grep -E 'accept|draft' campaign/upstream/$CFG.log 2>/dev/null | tail -3 >> campaign/upstream/$CFG-clean.out
  fi
  BPID=$(server_pid); [ -n "$BPID" ] && { kill "$BPID"; sleep 6; }
done
echo "WINDOW COMPLETE"
