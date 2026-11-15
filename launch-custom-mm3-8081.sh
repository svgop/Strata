#!/usr/bin/env bash
# launch-custom-mm3-8081.sh - the MiniMax-M3.1-Flash lane on 8081 (the third family, src/mm3 +
# strata-mm3 from Project Maya v1.1.63): the Baekpica VQ pack (GVQ1 13-bit codebook experts +
# GVR8 residuals, dense Q4_K/Q8_0/BF16, DSpark drafter, 1M-ctx model run at 131072 here).
# Alternate occupant of 8081 (the port law: swap by stopping one, starting the other).
# The serving stack is our serve/server.py with the minimax-m3 family port (the family EOS,
# model-card sampling, the pack tokenizer stub with the M2-lineage pre-pattern + NFC).
# VQ quality receipts: KL 0.150 / 88.2% top-1 vs the NVFP4 QAT source (the pack's own).
source "$(dirname "$0")/campaign/lib/launch-common.sh"
cd "$(dirname "$0")"
launch::exec_engine --config config/custom-mm3-8081.json --port 8081 --host 0.0.0.0
