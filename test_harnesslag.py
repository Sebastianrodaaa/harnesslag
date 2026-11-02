"""python3 test_harnesslag.py  (CPU, 16 real GSM8K problems, SmolLM2-135M-Instruct; also runs under pytest)"""
import copy
import random
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harnesslag.env import calculator, load, rollout, score
from harnesslag.harness import DEFAULT, edits, render, renderable
from harnesslag.mismatch import action_logprobs, decompose
from harnesslag.train import grpo_advantages, sync_sampler, train_step

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_cache = {}


def setup():
    if not _cache:
        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        sampler = copy.deepcopy(model).eval()
        problems = load("train")[:16]
        gen = torch.Generator().manual_seed(0)
        old_h = dict(DEFAULT)
        traces = rollout(sampler, tok, [p for p in problems[:4] for _ in range(4)], old_h, gen=gen, max_new=96)
        with torch.no_grad():
            for t, lp in zip(traces, action_logprobs(model, [render(t, old_h, tok) for t in traces])):
                t["t_oo"] = lp.tolist()
        _cache.update(tok=tok, model=model, sampler=sampler, problems=problems, traces=traces, old_h=old_h)
    return _cache


def test_calculator_and_score():
    assert calculator("2*(3+4)") == {"ok": True, "value": "14"}
    assert calculator("10/3", rounding=True)["value"] == "3.33"
    assert not calculator("__import__('os')")["ok"] and not calculator("1/0")["ok"]
    assert score("so #### 1,234", "1234") == (1.1, True) and score("no answer", "3") == (0.0, False)


def test_render_roundtrip_and_semantics():
    c = setup()
    tok, t = c["tok"], c["traces"][0]
    v1, v2 = dict(DEFAULT), {**DEFAULT, "system": 2, "tool_format": "json", "answer_reminder": True}
    a = render(t, v1, tok)
    render(t, v2, tok)
    assert render(t, v1, tok) == a, "v1 -> v2 -> v1 must be byte-identical"
    assert [x for x, m in zip(*a) if m] == [x for x, m in zip(*render(t, v2, tok)) if m], "actions must align"
    assert renderable(t, v2)
    assert renderable({**t, "tools": []}, {**v1, "rounding": True})
    assert not renderable({**t, "tools": [calculator("1+1")]}, {**v1, "rounding": True})
    assert len(edits(DEFAULT)) == 7


def test_three_terms_sum_to_total():
    c = setup()
    tok, model, traces, old_h = c["tok"], copy.deepcopy(c["model"]), c["traces"], c["old_h"]
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    (-sum(lp.sum() for lp in action_logprobs(model, [render(t, old_h, tok) for t in traces[:4]]))).backward()
    opt.step()  # weights move: policy lag is now non-zero
    new_h = {**old_h, "system": 1, "tool_format": "truncated", "answer_reminder": True}
    with torch.no_grad():
        t_no = action_logprobs(model, [render(t, old_h, tok) for t in traces])
        t_nn = action_logprobs(model, [render(t, new_h, tok) for t in traces])
    for t, no, nn in zip(traces, t_no, t_nn):
        s, too = torch.tensor(t["logp"]), torch.tensor(t["t_oo"])
        assert len(s) == len(too) == len(no) == len(nn)
        n, p, h, total = decompose(s, too, no, nn)
        assert (n + p + h - (nn - s)).abs().max() < 1e-4
        assert (total - (nn - s)).abs().max() == 0
        assert n.abs().max() < 0.5 and p.abs().max() > 0 and h.abs().max() > 0


def test_one_rl_step_per_arm():
    c = setup()
    tok, problems, old_h = c["tok"], c["problems"], c["old_h"]
    sem_h = {**old_h, "rounding": True}  # a semantic edit: all old traces become non-renderable
    render_h = {**old_h, "system": 1}
    stale = copy.deepcopy(c["traces"])
    grpo_advantages(stale, 4)
    for arm in "ABCDE":
        for h in (render_h, sem_h):
            model = copy.deepcopy(c["model"])
            sampler = copy.deepcopy(model)
            sync_sampler(sampler, model)
            fresh = rollout(sampler, tok, [p for p in problems[4:8] for _ in range(4)], h,
                            gen=torch.Generator().manual_seed(1), max_new=64)
            grpo_advantages(fresh, 4)
            with torch.no_grad():
                for t, lp in zip(fresh, action_logprobs(model, [render(t, h, tok) for t in fresh])):
                    t["t_oo"] = lp.tolist()
            before = [p.clone() for p in model.parameters()][:3]
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            st = train_step(model, opt, tok, deque([stale, fresh]), h, arm, 2.0, 8)
            n_stale = sum(t["adv"] != 0 for t in stale)
            n_tool = sum(t["adv"] != 0 and bool(t["tools"]) for t in stale)
            if arm == "A":
                assert st["dropped_old_harness"] == n_stale
            if arm in "CDE" and h is sem_h:
                assert st["dropped_nonrenderable"] == n_tool and st["stale_items"] == n_stale - n_tool
            if arm in "CDE" and h is render_h:
                assert st["rerendered_items"] == n_stale and st["absh"] > 0
            if arm == "B":
                assert st["stale_items"] == n_stale
            if st["items"]:
                assert any((a != b).any() for a, b in zip(before, model.parameters())), arm


if __name__ == "__main__":
    random.seed(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
