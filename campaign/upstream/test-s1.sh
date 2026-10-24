#!/bin/bash
set -u
cd /home/svg/LLM/strata-custom-2
KEY=3f0940df8efb9c363e4189956ae907d3e4acac221aab39f44804fe0b0df1b8
restore_custom() {
  (nohup bash launch-custom-iq3s-8081.sh > campaign/upstream/restore-custom2.out 2>&1 &)
  for i in $(seq 1 60); do sleep 10; curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -q 200 && { echo "CUSTOM SERVER RESTORED"; return 0; }; done
  echo "RESTORE FAILED"; return 1
}
trap 'restore_custom' EXIT
PID=$(ss -ltnp 2>/dev/null | grep ':8081 ' | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$PID" ] && { kill "$PID"; sleep 6; }
.venv/bin/python serve/server.py --engine strata --config campaign/upstream/s1-mtp.json --port 8081 --host 0.0.0.0 > campaign/upstream/boot-s1b.out 2>&1 &
for i in $(seq 1 45); do
  sleep 10
  curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8081/v1/models 2>/dev/null | grep -q 200 && { echo "S1 up ~$((i*10))s"; break; }
done
sleep 5
for L in 1 2 3; do echo "--- S1 leg $L $(date +%H:%M:%S) ---" >> campaign/upstream/s1-mtp.out; timeout 300 .venv/bin/python campaign/probe-lane.py 2>&1 | grep -E 'M1 single|M2 pair|M2 6-conc|M3b rate' >> campaign/upstream/s1-mtp.out; done
grep -iE 'mtp|accept' campaign/upstream/s1-mtp.log 2>/dev/null | tail -4 >> campaign/upstream/s1-mtp.out
