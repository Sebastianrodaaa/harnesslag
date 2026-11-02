"""Harness component space and deterministic rendering of a raw env trace into token ids.

A harness is a plain dict. Rendering is a pure function of (trace, harness, tokenizer):
context segments are tokenized one by one and the sampled action ids are pasted in
unchanged, so the action tokens line up across any two renderings.
"""
import json

SYSTEM_PROMPTS = [
    "Solve the math problem step by step. You can use a calculator by writing "
    "<calc>expression</calc>. End with #### followed by the final number.",
    "You are a careful math tutor. For any arithmetic, call the calculator like this: "
    "<calc>2*(3+4)</calc>. Give the final answer as #### <number>.",
    "Answer the grade-school math question. The tool <calc>...</calc> evaluates arithmetic "
    "and returns the result. Final line: #### <answer>.",
]
COMPONENTS = {
    "system": [0, 1, 2],
    "tool_format": ["raw", "truncated", "json"],
    "error_reminder": [False, True],
    "answer_reminder": [False, True],
    "rounding": [False, True],  # semantic: changes what the tool returns, not how it is shown
}
SEMANTIC = {"rounding"}
DEFAULT = {"system": 0, "tool_format": "raw", "error_reminder": False, "answer_reminder": False, "rounding": False}
ERROR_REMINDER = "\nReminder: only numbers and + - * / ** ( ) are allowed inside <calc>."
ANSWER_REMINDER = "\nWhen you are done, write #### followed by the final number."


def edits(h):
    """All single-component edits of harness h."""
    return [{**h, k: v} for k, vals in COMPONENTS.items() for v in vals if v != h[k]]


def describe(old, new):
    return ", ".join(f"{k}: {old[k]} -> {new[k]}" for k in COMPONENTS if old[k] != new[k]) or "none"


def renderable(trace, h):
    """A trace can be re-rendered only if the tool semantics it was collected under still hold
    (a trace that never called the tool does not depend on them)."""
    return not trace["tools"] or all(trace["harness"][k] == h[k] for k in SEMANTIC)


def tool_text(res, h):
    fmt = h["tool_format"]
    if fmt == "json":
        text = json.dumps({"result": res["value"]} if res["ok"] else {"error": res["value"]})
    elif fmt == "truncated":
        v = res["value"]
        if res["ok"]:
            try:
                v = f"{float(v):.6g}"
            except ValueError:
                pass
        text = ("Result: " if res["ok"] else "Error: ") + v[:40]
    else:
        text = res["value"] if res["ok"] else "error: " + res["value"]
    if h["error_reminder"] and not res["ok"]:
        text += ERROR_REMINDER
    if h["answer_reminder"]:
        text += ANSWER_REMINDER
    return text


def render(trace, h, tok):
    """-> (ids, action_mask). Renders every action and every tool result in the trace;
    a trace with as many tool results as actions ends in an open assistant turn (a generation prompt)."""
    ids, mask = [], []

    def ctx(text):
        t = tok.encode(text, add_special_tokens=False)
        ids.extend(t)
        mask.extend([0] * len(t))

    ctx(f"<|im_start|>system\n{SYSTEM_PROMPTS[h['system']]}<|im_end|>\n"
        f"<|im_start|>user\n{trace['question']}<|im_end|>\n<|im_start|>assistant\n")
    for i, a in enumerate(trace["actions"]):
        ids.extend(a)
        mask.extend([1] * len(a))
        if i < len(trace["tools"]):
            close = "" if a and a[-1] == tok.convert_tokens_to_ids("<|im_end|>") else "<|im_end|>"
            ctx(f"{close}\n<|im_start|>user\n{tool_text(trace['tools'][i], h)}<|im_end|>\n<|im_start|>assistant\n")
    return ids, mask
