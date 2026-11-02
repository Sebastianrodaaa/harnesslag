"""Harness edit proposals and paired-test acceptance on the held-out selection split."""
import numpy as np
from scipy.stats import ttest_rel

from harnesslag.env import rollout
from harnesslag.harness import describe, edits


def paired_p(cand, base):
    d = np.asarray(cand) - np.asarray(base)
    if d.std() == 0:
        return 0.0 if d.mean() > 0 else 1.0
    return float(ttest_rel(cand, base, alternative="greater").pvalue)


def select(model, tok, problems, h, rng, k=4, alpha=0.1):
    """Propose k single-component edits, score each with greedy rollouts on `problems`, and accept the
    best one whose per-problem reward beats the current harness in a one-sided paired t-test (p < alpha)."""
    base = [t["reward"] for t in rollout(model, tok, problems, h, greedy=True)]
    cands = []
    for c in rng.sample(edits(h), k):
        r = [t["reward"] for t in rollout(model, tok, problems, c, greedy=True)]
        cands.append({"harness": c, "edit": describe(h, c), "mean": float(np.mean(r)),
                      "delta": float(np.mean(r) - np.mean(base)), "p": paired_p(r, base)})
    ok = [c for c in cands if c["p"] < alpha and c["delta"] > 0]
    best = max(ok, key=lambda c: c["delta"]) if ok else None
    return (best["harness"] if best else None), {"base_mean": float(np.mean(base)), "candidates": cands,
                                                 "accepted": best["edit"] if best else None}
