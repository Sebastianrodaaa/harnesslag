#!/usr/bin/env bash
# Small-scale CPU run (SmolLM2-135M-Instruct): 5 arms x 3 seeds, 48 GRPO steps each.
# Arm A evolves the harness live; B-E replay the edits A accepted for the same seed,
# so every arm of a seed trains under the same harness schedule.
set -euo pipefail
OUT=${OUT:-results}
JOBS=${JOBS:-4}
ARGS="--steps 48 --edit-every 12 --eval-every 12 --threads ${THREADS:-2} --out $OUT"
mkdir -p "$OUT/logs"
for s in 0 1 2; do
  python -m harnesslag.train --arm A --seed $s $ARGS > "$OUT/logs/A_s$s.log" 2>&1 &
done
wait
for s in 0 1 2; do for arm in B C D E; do echo "$arm $s"; done; done |
  xargs -P "$JOBS" -L 1 sh -c "python -m harnesslag.train --arm \$0 --seed \$1 $ARGS --schedule $OUT/A_s\$1.json > $OUT/logs/\$0_s\$1.log 2>&1"
