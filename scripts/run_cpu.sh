#!/usr/bin/env bash
# Small-scale CPU run (SmolLM2-135M-Instruct): 5 arms x 3 seeds, 48 GRPO steps each.
# Arm A evolves the harness live; B-E replay the edits A accepted for the same seed,
# so every arm of a seed trains under the same harness schedule.
# Resumable: finished runs are skipped, a run already in progress is waited for,
# and B-E of a seed start once A of that seed has finished. ~8.6 GB RAM per job.
set -euo pipefail
export OUT=${OUT:-results}
export ARGS="--steps 48 --edit-every 12 --eval-every 12 --threads ${THREADS:-2} --out $OUT"
mkdir -p "$OUT/logs"
run() {
  local arm=$1 s=$2 extra=""
  finished() { grep -q '"final"' "$OUT/$1_s$2.json" 2>/dev/null; }
  while pgrep -f "harnesslag.train --arm $arm --seed $s " >/dev/null; do sleep 60; done
  finished "$arm" "$s" && return 0
  if [ "$arm" != A ]; then
    while pgrep -f "harnesslag.train --arm A --seed $s " >/dev/null; do sleep 60; done
    finished A "$s" || { echo "arm A seed $s did not finish" >&2; return 1; }
    extra="--schedule $OUT/A_s$s.json"
  fi
  echo "start $arm s$s"
  python -m harnesslag.train --arm "$arm" --seed "$s" $ARGS $extra > "$OUT/logs/${arm}_s$s.log" 2>&1
}
export -f run
for s in 0 1 2; do echo "A $s"; done > /tmp/queue
for s in 1 2 0; do for arm in B C D E; do echo "$arm $s"; done; done >> /tmp/queue
xargs -P "${JOBS:-3}" -L 1 bash -c 'run $0 $1' < /tmp/queue
