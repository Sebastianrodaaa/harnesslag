#!/usr/bin/env bash
# Full-scale run from the brief. NOT RUN in this repo (CPU only); every committed number is small-scale CPU.
# Qwen2.5-0.5B-Instruct, 150 GRPO steps, 4 prompts x 8 samples, harness edits every 25 steps,
# 5 arms x 3 seeds on one 24 GB GPU (L4 / A10G class). The brief estimates 12-16 GPU-hours;
# the first command times 5 steps so the real figure is known before launching all 15 runs.
#   bash scripts/full_scale_gpu.sh
set -euo pipefail
OUT=results/gpu
COMMON=(--model Qwen/Qwen2.5-0.5B-Instruct --device cuda --steps 150 --edit-every 25 --eval-every 25
        --prompts 4 --group 8 --buffer 4 --select-n 200 --test-n 200 --mb 8 --eval-batch 128
        --label "full-scale GPU run")
python -m harnesslag.train --arm D --seed 0 "${COMMON[@]}" --steps 5 --final-test-n 0 --out "$OUT/timing"
python - <<'PY'
import json; t = json.load(open("results/gpu/timing/D_s0.json"))["time"]
print(f"5 steps: {t['total']:.0f}s -> est. {t['total'] / 5 * 150 * 15 / 3600:.1f} GPU-hours for 15 runs (plus evals)")
PY
read -r -p "launch all 15 runs? [y/N] " ok && [ "$ok" = y ] || exit 0
for s in 0 1 2; do
  python -m harnesslag.train --arm A --seed $s "${COMMON[@]}" --out "$OUT"
  for arm in B C D E; do
    python -m harnesslag.train --arm $arm --seed $s "${COMMON[@]}" --schedule "$OUT/A_s$s.json" --out "$OUT"
  done
done
python scripts/plots.py "$OUT"
