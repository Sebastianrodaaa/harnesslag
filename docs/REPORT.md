# harnesslag: report

**Scale label: small-scale CPU run.** SmolLM2-135M-Instruct, 48 GRPO steps per run, 5 arms x 3 seeds, one 8-core
CPU machine with no GPU. Every number here is read from `results/*.json` by `scripts/plots.py`. None was typed by
hand. The full-scale GPU configuration in `scripts/full_scale_gpu.sh` has not been run.

## 1. Question

In an agent RL loop whose harness keeps changing (system prompt, tool-output format, reminders), can rollouts
sampled under an old harness be reused, so that fewer fresh rollouts are needed? The hypothesis from the brief:
harness-aware correction (arm D) reaches arm A's final accuracy with at least 25% fewer fresh rollouts, and beats
naive reuse (arm B) by at least 2 points at equal fresh rollouts. It fails if the saving is under 10% or inside
3-seed variance, or if re-rendering without weights (arm C) matches D.

## 2. Method

**Raw traces.** A rollout is stored as a list of sampled action-token IDs per turn, the raw calculator results,
the sampler's per-token log-probabilities and the harness config it was sampled under. It is never stored as
rendered text. `harness.render(trace, h, tok)` builds the token sequence for any harness `h`. The action tokens
are inserted unchanged, so token positions line up across harnesses. Rendering under v1, then v2, then v1 again
gives byte-identical output (tested).

**Harness space.** Three system prompts, tool-output format (raw, truncated to 6 significant digits, JSON), an
error reminder after a failed tool call, an answer reminder, and calculator rounding. Rounding is a semantic edit:
it changes what the tool returns. `renderable()` therefore rejects any trace with a tool result under a harness
whose calculator semantics differ. Traces with no tool calls stay renderable.

**Exact split.** For a stale action token with sampler log-prob `s` (old weights, bf16 sampler, old context):

- `t_oo`: trainer fp32 log-prob under the weights at sampling time, on the old context. It is computed right after
  sampling and stored in the trace.
- `t_no`: trainer log-prob under the current weights, on the old context.
- `t_nn`: trainer log-prob under the current weights, on the current context.

numerics = `t_oo - s`, policy lag = `t_no - t_oo`, harness lag = `t_nn - t_no`. The sum is `t_nn - s` by
construction. The test checks it to 1e-4 in float32 on real forward passes, so a bookkeeping error that misaligns
action positions would fail the test.

**Numerics are real.** After each step the sampler is a copy of the trainer with its weights rounded to bf16, and
it decodes with a KV cache in batches. The trainer scores full sequences in fp32. So the numerics term measures a
real engine mismatch, not an injected one.

**Weights.** Arms A and E: `min(exp(total), 2)`. Arm D: `min(exp(n - n_bar), 2) * min(exp(p), 2) * min(exp(h), 2)`,
where `n_bar` is the mean numerics term over the batch. Arms B and C: weight 1. The weight multiplies a
token-level GRPO policy-gradient loss with group-normalized advantages (group size 8, no KL, no critic, AdamW
lr 1e-5).

**Harness evolution.** Every 12 steps, arm A proposes 4 single-component edits, scores each with greedy rollouts on
200 held-out train problems, and accepts the best one if its mean reward is higher and a one-sided paired t-test
gives p < 0.1. Arms B to E replay arm A's accepted edits for the same seed (`--schedule`). The harness schedule
is therefore identical across arms and only the use of stale data differs.

**Reward.** 1 for a correct final answer and 0.1 for using the `#### <answer>` format. Without the format bonus a
135M model gets almost no learning signal. Reported accuracy counts correctness only.

## 3. Related work

See [INSPIRATIONS.md](../INSPIRATIONS.md). In short, prior work splits training-inference mismatch into numerics and
policy lag, and co-evolves harnesses with policies. Our search of abstracts found none that keeps rollouts
across harness versions by re-rendering them and correcting the context shift as its own importance-weight term.

## 4. Results

All numbers are from **small-scale CPU runs**: SmolLM2-135M-Instruct, 48 GRPO steps × 32 rollouts, 5 arms × 3 seeds,
104.6 CPU-hours in total on one 8-core machine. Raw per-step logs are in `results/*.json` and `results/logs/`.

### 4.1 Final accuracy (full GSM8K test split, n = 1,319, greedy)

| arm | seed 0 | seed 1 | seed 2 | mean ± 95% CI |
|---|---|---|---|---|
| A discard stale | 1.82 | 1.82 | 1.59 | 1.74 ± 0.33 |
| B naive reuse | 2.35 | 1.44 | 1.74 | 1.84 ± 1.15 |
| C re-render, no IS | 2.50 | 1.67 | 1.74 | 1.97 ± 1.15 |
| D harnesslag | 2.58 | 1.97 | 1.82 | 2.12 ± 1.00 |
| E re-render, one cap | 2.12 | 1.59 | 1.59 | 1.77 ± 0.76 |

Paired differences of D against the other arms, per seed, in points (paired t-test, 3 seeds):

| comparison | seed 0 | seed 1 | seed 2 | p |
|---|---|---|---|---|
| D − A | +0.76 | +0.15 | +0.23 | 0.19 |
| D − B | +0.23 | +0.53 | +0.08 | 0.17 |
| D − C | +0.08 | +0.30 | +0.08 | 0.18 |
| D − E | +0.45 | +0.38 | +0.23 | 0.03 |

### 4.2 Verdict on the hypothesis: not supported at this scale

- *D beats B by ≥ 2 points:* **fails.** The gap is +0.28 points.
- *D reaches A's final accuracy with ≥ 25% fewer fresh rollouts:* the script reports a 75% saving, but **it is not
  meaningful**. The accuracy curve uses a 200-problem subset evaluated every 12 steps, so one problem is
  0.5 points. A's curve peaks at 384 rollouts and drops to 1.17% by the end, so any arm "matches" it at the first
  eval. The saving sits inside 3-seed variance, and by the brief's own failure criterion that counts as a fail.
- *C matches D:* C and D differ by 0.15 points on average (p = 0.18), so **we cannot rule out that C matches D**.
- D is above A, B and C on every seed, and above E with p = 0.03. That is consistent with the idea, but these are
  4 comparisons on 3 seeds, the gaps are 1 to 10 problems out of 1,319, and absolute accuracy is about 2%. We do
  not count it as evidence.

### 4.3 What the decomposition measured

Mean |term| per stale action token, averaged over all steps and seeds (nats):

| arm | numerics | policy lag | harness lag | total |
|---|---|---|---|---|
| B | 0.202 | 0.057 | n/a | 0.251 |
| C | 0.194 | 0.052 | 0.002 | 0.236 |
| D | 0.170 | 0.036 | 0.003 | 0.198 |
| E | 0.159 | 0.033 | 0.003 | 0.185 |

- **Numerics dominates.** At step 1 it is about 1e-6, because the checkpoint is stored in bf16 and the bf16 sampler
  copy is exact. It grows to roughly 0.2 nats as fp32 updates (lr 1e-5) move the trainer's weights off the bf16 grid,
  which the sampler's rounding then undoes (our interpretation of `mismatch.png`). Centering the numerics term
  (arm D) takes out its mean (n̄ ≈ −0.03 to −0.12) but not its spread.
- **Policy lag** stays around 0.03 to 0.06 with a 4-batch buffer.
- **Harness lag** is about 0.1 nats per token for the 3 steps after an accepted system-prompt edit, while pre-edit
  batches are still in the buffer. Averaged over all steps it is small: the buffer is short, and only 2 of 9 edit
  points accepted an edit (system prompt, seeds 0 and 1).
- **Clipping.** D clips 1.37% of stale tokens and E clips 1.02%, both at cap 2.

### 4.4 Honest failures

- **The tool was barely used.** The 135M model calls the calculator in roughly 0.1% of rollouts. The rounding
  (semantic) edit was never accepted, and **no trace was dropped as non-renderable** in any run. The semantic-edit
  logic is covered by `test_harnesslag.py` but was not exercised by training.
- **Harness lag was rarely present.** Seed 2 accepted no edit, so its arm C is the same algorithm as B (both 1.74%).
- **Absolute accuracy is about 2%.** That is too close to the floor for arm differences to be measured.

The pipeline runs end to end on real data, and the decomposition is exact (tested to 1e-4) and cheap. The accuracy
question needs the full-scale run (`scripts/full_scale_gpu.sh`, Qwen2.5-0.5B, 150 steps), which **was not run**.

## 5. Limitations

- **Scale.** 135M parameters and 48 steps of 32 rollouts. GSM8K accuracy for this model is in the low single
  digits, so accuracy differences between arms are small next to seed noise. This experiment cannot support or
  refute the accuracy claim at the scale the brief intends (Qwen2.5-0.5B, 150 steps). It does measure the
  mismatch terms on a real model.
- **Harness schedule.** Edits come from arm A's live selection and are replayed. If A accepts few edits, there is
  little harness lag to correct, and D and C converge to near-identical algorithms.
- **Curve resolution.** The accuracy curve is evaluated every 12 steps on a fixed 200-problem test subset, so the
  rollout-saving estimate is quantized to 12-step (384-rollout) units.
- **Single cap value** (2) and a single buffer size (4 batches). No sweep.
- **CPU-hours** are wall time x threads per job, counting only the training process.
