#!/bin/bash
# F-209 settle round 2: 3 interleaved old/new pairs, one continuous run.
# Binary paths are deliberately ONLY in this file's contents (never in any
# caller's cmdline) - battery-boot.sh pkills by a 'build/strata ' pattern.
set -u
cd /home/svg/LLM/strata-custom-2
OLD=build/strata.pre-f209
NEW=build/strata.f209-merged
CUR=build/strata

for round in 1 2 3; do
  cp "$OLD" "$CUR"
  campaign/battery-boot.sh "f209-r2-old$round" || exit 1
  cp "$NEW" "$CUR"
  campaign/battery-boot.sh "f209-r2-new$round" || exit 1
done

# leave the merged binary in place afterwards
cp "$NEW" "$CUR"
echo "=== ROUND-2 SUMMARY ==="
for f in campaign/results/f209-r2-*.txt; do
  printf '%-28s ' "$(basename "$f" .txt)"
  grep -hE 'M1 single-stream|M2 pair |M2 6-conc|M3b rate' "$f" | tr '\n' ' ' | sed 's/  */ /g'
  echo
done
