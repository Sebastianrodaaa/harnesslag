ARM ?= D
SEED ?= 0
STEPS ?= 48
SCHED = $(if $(filter-out A,$(ARM)),--schedule results/A_s$(SEED).json)

test:
	python3 test_harnesslag.py

repro:  # one small-scale CPU run; B-E replay arm A's harness edits for the same seed
	python3 -m harnesslag.train --arm $(ARM) --seed $(SEED) --steps $(STEPS) --edit-every 12 --eval-every 12 $(SCHED) --out results/repro

all-cpu:
	scripts/run_cpu.sh

figures:
	python3 scripts/plots.py && python3 scripts/demo_gif.py && python3 scripts/dashboard.py

.PHONY: test repro all-cpu figures
