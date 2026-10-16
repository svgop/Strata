#!/bin/bash
# F-220 pre-warm (smoothness agent T8.B, knob #1): fully populate the page cache with the
# mmap'd weight files before the server boots, so every battery round starts from the same
# warm state. The morning-vs-evening gap (181 vs 196-208) is host-side page-cache warm-in of
# the mmap'd ~85GB (pack + shards), not GPU thermal — a warm GPU reading FASTER contradicts
# throttling. No privileges needed; pinning (mlock) would additionally guard against eviction
# under memory pressure, which our 184GB box does not exercise (~55-85GB resident, no pressure).
# Idempotent; ~10-25s on warm NVMe, worth it once per boot and before every battery.
set -u
T0=$(date +%s)
for f in /home/svg/LLM/Custom-data/packs/iq3_s/* \
         /home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
         /home/svg/LLM/Custom-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf; do
  [ -f "$f" ] || continue
  dd if="$f" of=/dev/null bs=8M status=none 2>/dev/null || cat "$f" > /dev/null 2>/dev/null || true
done
echo "[$(date +%H:%M:%S)] prewarm done in $(( $(date +%s) - T0 ))s (page cache now carries pack+shards)"
