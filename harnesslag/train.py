"""Plain PyTorch GRPO with a replay buffer and an evolving harness.

Arms (same fresh rollouts and optimizer steps; they differ only in what they do with stale data):
  A  discard rollouts collected under an older harness (same-harness replay keeps one truncated IS weight)
  B  naive reuse: train on stale rollouts in their old rendering, no IS weight
  C  re-render under the current harness, no IS weight
  D  harnesslag: re-render, split the log-ratio into numerics / policy lag / harness lag,
     center numerics, truncate each source separately
  E  re-render, one combined truncated IS weight (no decomposition)
C, D and E drop traces whose tool semantics changed (not re-renderable).

  python -m harnesslag.train --arm D --seed 0
"""
import argparse
import copy
import json
import os
import random
import time
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harnesslag import evolve
from harnesslag.env import load, rollout
from harnesslag.harness import DEFAULT, describe, render, renderable
from harnesslag.mismatch import action_logprobs, decompose, weights


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def sync_sampler(sampler, model):
    """The sampler serves bf16-rounded weights, as an inference engine would after a weight sync."""
    with torch.no_grad():
        for ps, pt in zip(sampler.parameters(), model.parameters()):
            ps.copy_(pt.to(torch.bfloat16).float())


@torch.no_grad()
def accuracy(model, tok, problems, h, batch):
    tr = rollout(model, tok, problems, h, greedy=True, batch=batch)
    return mean([float(t["correct"]) for t in tr]), mean([float(t["reward"] > 0) for t in tr])


def grpo_advantages(traces, group):
    for g in range(0, len(traces), group):
        r = torch.tensor([t["reward"] for t in traces[g:g + group]])
        a = (r - r.mean()) / (r.std() + 1e-6) if r.std() > 0 else torch.zeros_like(r)
        for t, v in zip(traces[g:g + group], a):
            t["adv"] = float(v)


def train_step(model, opt, tok, buffer, h, arm, cap, mb):
    """One optimizer step on the fresh batch plus the replay buffer. Returns mismatch stats."""
    items, st = [], {"dropped_old_harness": 0, "dropped_nonrenderable": 0, "stale_items": 0, "rerendered_items": 0}
    for bi, batch in enumerate(buffer):
        stale = bi < len(buffer) - 1
        for t in batch:
            if t["adv"] == 0:
                continue
            same = t["harness"] == h
            if not same and arm == "A":
                st["dropped_old_harness"] += 1
                continue
            if not same and arm in "CDE" and not renderable(t, h):
                st["dropped_nonrenderable"] += 1
                continue
            old = render(t, t["harness"], tok)
            rerender = not same and arm != "B"
            items.append({"t": t, "stale": stale, "old": old, "new": render(t, h, tok) if rerender else old,
                          "rerender": rerender})
    st["items"] = len(items)
    if not items:
        return st
    n_bar = float(torch.cat([torch.tensor(it["t"]["t_oo"]) - torch.tensor(it["t"]["logp"]) for it in items]).mean())
    need = [it for it in items if it["rerender"]]
    with torch.no_grad():
        for k in range(0, len(need), mb):
            for it, lp in zip(need[k:k + mb], action_logprobs(model, [it["old"] for it in need[k:k + mb]])):
                it["t_no"] = lp
    ntok = sum(len(it["t"]["logp"]) for it in items)
    acc = {k: [] for k in ("n_fresh", "absn_fresh", "absn", "absp", "absh", "abstot", "w", "clip")}
    demo = None
    items.sort(key=lambda it: len(it["new"][0]))
    for k in range(0, len(items), mb):
        chunk = items[k:k + mb]
        loss = 0.0
        for it, lp in zip(chunk, action_logprobs(model, [it["new"] for it in chunk])):
            t, tnn = it["t"], lp.detach()
            n, p, hl, total = decompose(torch.tensor(t["logp"], device=lp.device), torch.tensor(t["t_oo"], device=lp.device),
                                        it.get("t_no", tnn), tnn)
            w, clipped = weights(arm, n, p, hl, total, n_bar, cap)
            loss = loss - (w * t["adv"] * lp).sum() / ntok
            if it["stale"]:
                acc["absn"].append(n.abs().mean().item())
                acc["absp"].append(p.abs().mean().item())
                if arm != "B":
                    acc["absh"].append(hl.abs().mean().item())
                acc["abstot"].append(total.abs().mean().item())
                acc["clip"].append(clipped.float().mean().item())
                acc["w"].append(w.mean().item())
                if it["rerender"] and demo is None:
                    demo = {"question": t["question"], "old_harness": t["harness"], "new_harness": h,
                            "old_context": tok.decode(it["old"][0]), "new_context": tok.decode(it["new"][0]),
                            "tokens": [tok.decode([x]) for a in t["actions"] for x in a],
                            "numerics": n.tolist(), "policy_lag": p.tolist(), "harness_lag": hl.tolist(),
                            "weight": w.tolist(), "reward": t["reward"]}
            else:
                acc["n_fresh"].append(n.mean().item())
                acc["absn_fresh"].append(n.abs().mean().item())
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    st.update({k: mean(v) for k, v in acc.items()})
    st["n_bar"] = n_bar
    st["stale_items"] = sum(it["stale"] for it in items)
    st["rerendered_items"] = len(need)
    st["demo"] = demo
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list("ABCDE"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--buffer", type=int, default=4, help="batches kept, fresh one included")
    ap.add_argument("--edit-every", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--select-n", type=int, default=200)
    ap.add_argument("--test-n", type=int, default=200, help="fixed test subset for the curve")
    ap.add_argument("--final-test-n", type=int, default=1319)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--schedule", default=None, help="replay accepted edits from another run's JSON")
    ap.add_argument("--out", default="results")
    ap.add_argument("--label", default="small-scale CPU run")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    sampler = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    train = load("train")
    random.Random(1234).shuffle(train)
    select_split, pool = train[:a.select_n], train[a.select_n:]
    random.Random(a.seed).shuffle(pool)
    test = load("test")
    curve_set = random.Random(0).sample(test, a.test_n)
    schedule = None
    if a.schedule:
        schedule = {e["step"]: e["harness"] for e in json.load(open(a.schedule))["edits"] if e["harness"]}

    h = dict(DEFAULT)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.arm}_s{a.seed}.json")
    log = {"config": vars(a), "steps": [], "evals": [], "edits": [], "demo": None, "time": {}}
    t0 = time.time()
    tm = {"rollout": 0.0, "train": 0.0, "evolve": 0.0, "eval": 0.0}

    def evaluate(step):
        s = time.time()
        acc, fmt = accuracy(model, tok, curve_set, h, a.eval_batch)
        tm["eval"] += time.time() - s
        log["evals"].append({"step": step, "fresh_rollouts": step * a.prompts * a.group, "acc": acc,
                             "format": fmt, "harness": dict(h)})
        print(f"[{a.arm} s{a.seed}] eval step {step} acc {acc:.3f} format {fmt:.3f}", flush=True)

    evaluate(0)
    buffer = deque(maxlen=a.buffer)
    for step in range(a.steps):
        if step > 0 and step % a.edit_every == 0:
            s = time.time()
            if schedule is not None:
                new, info = schedule.get(step), {"replayed_from": a.schedule}
            else:
                new, info = evolve.select(model, tok, select_split, h, rng)
            tm["evolve"] += time.time() - s
            log["edits"].append({"step": step, "harness": new, "edit": describe(h, new) if new else None, **info})
            print(f"[{a.arm} s{a.seed}] step {step} edit: {describe(h, new) if new else 'rejected'}", flush=True)
            if new:
                h = new
        sync_sampler(sampler, model)
        s = time.time()
        batch = pool[step * a.prompts:(step + 1) * a.prompts]
        gen = torch.Generator(device=a.device).manual_seed(a.seed * 100003 + step)
        traces = rollout(sampler, tok, [p for p in batch for _ in range(a.group)], h, gen=gen)
        grpo_advantages(traces, a.group)
        with torch.no_grad():
            for k in range(0, len(traces), a.mb):
                ch = traces[k:k + a.mb]
                for t, lp in zip(ch, action_logprobs(model, [render(t, h, tok) for t in ch])):
                    t["t_oo"] = lp.tolist()
        buffer.append(traces)
        tm["rollout"] += time.time() - s
        s = time.time()
        st = train_step(model, opt, tok, buffer, h, a.arm, a.cap, a.mb)
        tm["train"] += time.time() - s
        if st.get("demo") and log["demo"] is None:
            log["demo"] = {"step": step, **st["demo"]}
        st.pop("demo", None)
        st.update({"step": step, "reward": mean([t["reward"] for t in traces]),
                   "correct": mean([float(t["correct"]) for t in traces]),
                   "tool_calls": mean([len(t["tools"]) for t in traces]), "elapsed": time.time() - t0})
        log["steps"].append(st)
        print(f"[{a.arm} s{a.seed}] step {step} reward {st['reward']:.3f} items {st['items']} "
              f"|n| {st.get('absn')} |p| {st.get('absp')} |h| {st.get('absh')} {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.eval_every == 0:
            evaluate(step + 1)
        log["time"] = {**tm, "total": time.time() - t0}
        json.dump(log, open(path, "w"))
    if not a.final_test_n:
        return
    s = time.time()
    acc, fmt = accuracy(model, tok, test[:a.final_test_n], h, a.eval_batch)
    tm["final_test"] = time.time() - s
    log["final"] = {"acc": acc, "format": fmt, "n": len(test[:a.final_test_n]), "harness": h}
    log["time"] = {**tm, "total": time.time() - t0}
    json.dump(log, open(path, "w"))
    print(f"[{a.arm} s{a.seed}] final test acc {acc:.4f} on {log['final']['n']}", flush=True)


if __name__ == "__main__":
    main()
