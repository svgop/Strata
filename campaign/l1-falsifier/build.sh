#!/usr/bin/env bash
# build.sh — GDN commit falsifier microbench (SCOUT2 L1 commit/kill, FALSIFIER SPEC v2)
set -euo pipefail
cd "$(dirname "$0")"
/usr/local/cuda-13.4/bin/nvcc -O3 -arch=sm_120 -lineinfo -std=c++17 \
    -Xptxas -v -o l1_commit_falsifier l1_commit_falsifier.cu 2>&1 | tee build.log
echo "build OK -> ./l1_commit_falsifier"
