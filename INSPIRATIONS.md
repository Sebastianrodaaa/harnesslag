# Inspirations and related work

Checked on arXiv, Hugging Face papers and GitHub on 2026-11-02 (from the project brief).

- **On the Origin and Evolution of Training-Inference Mismatch in LLM RL** (https://huggingface.co/papers/2610.40945).
  An exact split into engine numerics and policy lag, with a correction for each source. harnesslag adds a third
  source, the context, and borrows the finding that numerics behave like a near-constant bias.
- **bf16 Rounding Silently Discards Most RL Policy Updates** (https://arxiv.org/abs/2610.41073v1) and **CIS**
  (https://arxiv.org/abs/2609.32444v1). Fixes for numerical and lag mismatch. Both sides see the same context.
  Our sampler serves bf16-rounded weights for the same reason.
- **CoTrace** (https://arxiv.org/abs/2610.10426v1). Routes RL data by harness version and trains only on fresh
  interactions. That is close to our arm A.
- **HarnessAdapt-RL** (https://arxiv.org/abs/2610.28069v1), **Self-HPO** (https://arxiv.org/abs/2610.22791v1),
  **WHALE** (https://arxiv.org/abs/2609.00196v1), **HarnessForge** (https://arxiv.org/abs/2606.01779v1),
  **CoEvo-RSI** (https://arxiv.org/abs/2610.25078v1). These co-evolve the harness and the policy. None of their
  abstracts re-renders or importance-corrects rollouts across harness versions.
- **MemoryWalker** (https://arxiv.org/abs/2609.00865v1). Corrects the mismatch that context compression causes
  inside one harness.
- **Evolving Together, Failing Apart** (https://arxiv.org/abs/2610.20845v1). Shows that co-evolved policies break
  under skill-bank drift. It is an evaluation protocol, not a training correction.
- **TinyAstral**: accept a harness edit only if a paired test on a selection split passes.
- **Truncated importance sampling** (Ionides 2008, and the TIS correction used in open RL stacks). It caps each
  importance weight from above.
- **GRPO** (Shao et al., DeepSeekMath, https://arxiv.org/abs/2402.03300). Group-normalized advantages with no critic.
- **athanor** (https://github.com/ghgoodreau/athanor). Gave the README its shape and the banner its dithered plates.
