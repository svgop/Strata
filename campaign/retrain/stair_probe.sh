#!/bin/bash
# campaign/retrain/stair_probe.sh — the T14C G0 M3d stair read: one CLI boot per drafter
# (stock rt vs trained rt2), STRATA_SPEC_DEPTH=1, same corpus stream, same args — the
# per-depth accepted/tested lines are the gate (v1: pos-2/3 +>=8pp, pos-0 <=1pp regression).
# Usage: bash campaign/retrain/stair_probe.sh <label> <mtp-dir>   (server must be DOWN)
set -u
REPO=/home/svg/LLM/strata-custom-2
LABEL=${1:?label}
MTP=${2:?mtp dir}
OUT=$REPO/campaign/results/f325-stair-$LABEL.out
ss -ltn 2>/dev/null | grep -q ":8081 " && { echo "ABORT: 8081 up (single-engine law)"; exit 1; }
STRATA_SPEC_DEPTH=1 timeout 900 "$REPO/engine/strata" \
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
  --tokens-file /home/svg/LLM/Custom-data/harvest/corpus/stream-0.ids --max-new 2048 \
  2>&1 | tee "$OUT" | grep -E 'spec depth|tokens/s|accept' | tail -8
echo "stair written: $OUT"
