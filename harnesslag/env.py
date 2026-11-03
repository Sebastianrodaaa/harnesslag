"""GSM8K calculator-tool environment. Rollouts are stored as raw traces (sampled action ids,
sampler log-probs, raw tool results), never as rendered text, so they can be re-rendered later."""
import ast
import json
import operator
import os
import re
import urllib.request

import torch

from harnesslag.harness import render

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"
MAX_TOOL_CALLS = 3
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def load(split):
    """GSM8K (MIT, openai/grade-school-math) as [{'question', 'answer'}], downloaded once into data/."""
    path = os.path.join(DATA, f"gsm8k_{split}.jsonl")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split), path)
    rows = [json.loads(line) for line in open(path)]
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in rows]


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 64:
            raise ValueError("exponent too large")
        return OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expr, rounding=False):
    """Safe arithmetic only (no names, no calls). Returns a raw tool result."""
    try:
        v = float(_eval(ast.parse(expr.strip(), mode="eval")))
        if abs(v) > 1e15:
            raise ValueError("result too large")
        if rounding:
            v = round(v, 2)
        return {"ok": True, "value": str(int(v)) if v.is_integer() else repr(v)}
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return {"ok": False, "value": str(e)[:60] or type(e).__name__}


def extract_answer(text):
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    return m[-1].replace(",", "") if m else None


def score(text, gold):
    """reward = 1 for a correct final answer + 0.1 for using the #### answer format at all."""
    pred = extract_answer(text)
    correct = pred is not None and abs(float(pred) - float(gold)) < 1e-6
    return float(correct) + 0.1 * (pred is not None), correct


@torch.no_grad()
def sample(model, tok, prompts, max_new, greedy, gen=None):
    """Batched KV-cache decoding with left padding. Stops a row at <|im_end|> or a closed </calc>.
    Returns (action ids, sampler log-probs of those ids)."""
    eos, B = tok.convert_tokens_to_ids("<|im_end|>"), len(prompts)
    L = max(map(len, prompts))
    ids = torch.full((B, L), eos, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p)
        att[i, L - len(p):] = 1
    ids, att = ids.to(model.device), att.to(model.device)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
    acts, lps, done, rows = [[] for _ in range(B)], [[] for _ in range(B)], [False] * B, list(range(B))
    for _ in range(max_new):
        logp = out.logits[:, -1].float().log_softmax(-1)
        nxt = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
        for j, i in enumerate(rows):
            acts[i].append(int(nxt[j]))
            lps[i].append(float(logp[j, nxt[j]]))
            done[i] = int(nxt[j]) == eos or "</calc>" in tok.decode(acts[i][-6:])
        keep = [j for j, i in enumerate(rows) if not done[i]]
        if not keep:
            break
        cache = out.past_key_values
        if len(keep) < len(rows):  # drop finished rows from the batch and the KV cache
            cache.batch_select_indices(torch.tensor(keep, device=model.device))
            nxt, att, pos, rows = nxt[keep], att[keep], pos[keep], [rows[j] for j in keep]
        att = torch.cat([att, att.new_ones((len(rows), 1))], 1)
        pos = pos[:, -1:] + 1
        out = model(input_ids=nxt[:, None], attention_mask=att, position_ids=pos,
                    past_key_values=cache, use_cache=True)
    return acts, lps


def rollout(model, tok, problems, h, greedy=False, gen=None, max_new=160, batch=64):
    """Run the tool loop for each problem under harness h; returns raw traces."""
    traces = [{"question": p["question"], "gold": p["answer"], "actions": [], "logp": [], "tools": [],
               "harness": dict(h)} for p in problems]
    active = list(range(len(traces)))
    while active:
        nxt = []
        for s in range(0, len(active), batch):
            chunk = active[s:s + batch]
            acts, lps = sample(model, tok, [render(traces[i], h, tok)[0] for i in chunk], max_new, greedy, gen)
            for i, a, lp in zip(chunk, acts, lps):
                t = traces[i]
                t["actions"].append(a)
                t["logp"].extend(lp)
                text = tok.decode(a)
                if text.rstrip().endswith("</calc>") and len(t["tools"]) < MAX_TOOL_CALLS:
                    t["tools"].append(calculator(text.rsplit("<calc>", 1)[-1].split("</calc>")[0], h["rounding"]))
                    nxt.append(i)
        active = nxt
    for t in traces:
        t["text"] = tok.decode([x for a in t["actions"] for x in a], skip_special_tokens=True)
        t["reward"], t["correct"] = score(t["text"], t["gold"])
    return traces
