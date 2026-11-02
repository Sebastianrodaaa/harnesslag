"""Exact three-term split of the per-token log-ratio between the current trainer policy in the
current rendering and the sampler that produced the token in the old rendering:

  log pi(theta_now, c_new) - log mu(theta_old, c_old)
    = [log pi(theta_old, c_old) - log mu(theta_old, c_old)]   numerics     (trainer vs sampler)
    + [log pi(theta_now, c_old) - log pi(theta_old, c_old)]   policy lag   (weights moved)
    + [log pi(theta_now, c_new) - log pi(theta_now, c_old)]   harness lag  (context re-rendered)
"""
import torch


def action_logprobs(model, seqs):
    """Trainer log-probs of the action tokens of each (ids, mask) rendering, one 1-D tensor per row."""
    L = max(len(ids) for ids, _ in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (x, _) in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    ids, att = ids.to(model.device), att.to(model.device)
    logits = model(input_ids=ids, attention_mask=att).logits
    out = []
    for i, (x, m) in enumerate(seqs):
        pos = torch.tensor([j for j, v in enumerate(m) if v], device=ids.device)
        out.append(logits[i, pos - 1].float().log_softmax(-1).gather(1, ids[i, pos, None]).squeeze(1))
    return out


def decompose(s, t_oo, t_no, t_nn):
    """-> numerics, policy lag, harness lag, total (all per token). The three terms telescope to total."""
    return t_oo - s, t_no - t_oo, t_nn - t_no, t_nn - s


def weights(arm, n, p, h, total, n_bar, cap):
    """Per-token importance weights (detached inputs).
    A, E: one truncated weight on the total ratio. B, C: none. D: numerics centered, then each source
    truncated on its own."""
    if arm in "BC":
        return torch.ones_like(total), torch.zeros_like(total, dtype=torch.bool)
    if arm in "AE":
        r = total.exp()
        return r.clamp(max=cap), r > cap
    rn, rp, rh = (n - n_bar).exp(), p.exp(), h.exp()
    return rn.clamp(max=cap) * rp.clamp(max=cap) * rh.clamp(max=cap), (rn > cap) | (rp > cap) | (rh > cap)
