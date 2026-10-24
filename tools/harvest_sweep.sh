#!/bin/bash
# tools/harvest_sweep.sh — the first-cycle harvest sweep: one CLI boot per stream,
# --dump-final-r writes records-<shard>.f32 (the F-321 patch). Server must be down.
# Usage: bash tools/harvest_sweep.sh <corpus-dir> <records-dir> [mtp-dir]  (default stock rt)
set -u
REPO=/home/svg/LLM/strata-custom-2
CORPUS=${1:?corpus dir}
OUT=${2:?records dir}
MTP=${3:-/home/svg/LLM/data/mtp-en/rt}
mkdir -p "$OUT"
port_busy() { ss -ltn 2>/dev/null | grep -q ":8081 "; }
if port_busy; then echo "ABORT: the 8081 server is up — stop it first (single-engine law)"; exit 1; fi
for f in "$CORPUS"/stream-*.ids; do
  name=$(basename "$f" .ids)
  [ -s "$OUT/records-$name.f32" ] && { echo "skip $name (exists)"; continue; }
  echo "=== harvest $name ($(wc -c < "$f") bytes of ids) $(date +%H:%M:%S) ==="
  timeout 1800 "$REPO/engine/strata" \
    --pack /home/svg/LLM/data/packs/iq3_s \
    --native /home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --ple-gguf /home/svg/LLM/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
    --expert-cache auto --spec 6 --max-context 131072 --kv int8 --kv-resident 32768 \
    --spec-min-p 0.5 --prefill auto:32768 --mtp-q4 all --mtp-hnorm stream \
    --pcie-frac 0.35 --pcie-mode dma --vram-reserve-mib 2000 \
    --control-vector-scaled refusal-r.gguf:1.0 --control-vector-layer-range 4 24 \
    --cvec-mode project --cvec-dir per-layer --ple-io mmap --adapt-decay 0.85 \
    --expert-profile "$REPO/data/expert-profile.bin" --resident-experts \
    --mtp "$MTP" --batch 6 \
    --tokens-file "$f" --max-new 1 \
    --dump-final-r "$OUT/records-$name.f32" > "$OUT/$name.out" 2>&1
  echo "  exit=$? size=$(stat -c %s "$OUT/records-$name.f32" 2>/dev/null)"
done
echo "sweep done $(date +%H:%M:%S)"
