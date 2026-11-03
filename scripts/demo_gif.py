"""docs/screenshots/demo.gif from a real run: one stale trajectory under the old and the new harness,
tokens shaded by |harness lag| with the per-token IS weight, next to the mismatch curves growing step by step."""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
plt.rcParams.update({"figure.facecolor": "#000", "axes.facecolor": "#000", "text.color": "#ddd", "axes.edgecolor": "#444",
                     "xtick.color": "#888", "ytick.color": "#888", "font.family": "monospace", "font.size": 8})


def pick():
    best = None
    for s in (0, 1, 2):
        p = os.path.join(RES, f"D_s{s}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("demo") and (best is None or max(map(abs, r["demo"]["harness_lag"])) > max(map(abs, best["demo"]["harness_lag"]))):
                best = r
    if best is None:
        sys.exit("no arm D run with a re-rendered stale trajectory yet")
    return best


def system_text(ctx):
    return ctx.split("<|im_start|>system\n")[-1].split("<|im_end|>")[0]


def frame(r, upto, ntok):
    d = r["demo"]
    fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
    ax_t = fig.add_axes([0.02, 0.04, 0.52, 0.92])
    ax_t.axis("off")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    old, new = d["old_harness"], d["new_harness"]
    diff = ", ".join(f"{k} {old[k]}->{new[k]}" for k in old if old[k] != new[k])
    ax_t.text(0, 0.98, f"arm D · seed {r['config']['seed']} · stale rollout re-rendered at step {d['step']}", va="top", color="#fff")
    ax_t.text(0, 0.93, f"edit: {diff}", va="top", color="#999")
    y = 0.86
    for lab, ctx in (("old system prompt", d["old_context"]), ("new system prompt", d["new_context"])):
        ax_t.text(0, y, lab, va="top", color="#777")
        y -= 0.04
        for line in textwrap.wrap(system_text(ctx), 62)[:4]:
            ax_t.text(0, y, line, va="top", color="#ccc", fontsize=7)
            y -= 0.035
        y -= 0.02
    ax_t.text(0, y, "sampled tokens · shade = |harness lag|", va="top", color="#777")
    y -= 0.045
    x = 0.0
    hl = np.abs(np.array(d["harness_lag"]))
    for i, tok in enumerate(d["tokens"][:ntok]):
        t = tok.replace("\n", "⏎")
        w = len(t) * 0.0125 + 0.004
        if x + w > 1:
            x, y = 0.0, y - 0.045
        if y < 0.05:
            break
        v = min(1.0, hl[i] / max(hl.max(), 1e-6)) if i < upto else 0.0
        g = 0.08 + 0.85 * v
        ax_t.text(x, y, t, va="top", fontsize=7.5, color="#000" if g > 0.55 else "#ddd",
                  bbox=dict(boxstyle="square,pad=0.08", fc=(g, g, g), ec="none"))
        if i == upto - 1:
            ax_t.text(0.0, 0.0, f"token {i} '{t}'  numerics {d['numerics'][i]:+.4f}  policy {d['policy_lag'][i]:+.4f}  "
                      f"harness {d['harness_lag'][i]:+.4f}  w {d['weight'][i]:.3f}", fontsize=6.5, color="#fff")
        x += w
    ax = fig.add_axes([0.6, 0.12, 0.38, 0.78])
    st = r["steps"]
    last = min(len(st), max(1, int(len(st) * upto / max(1, ntok))))
    xs = [s["step"] for s in st[:last]]
    for k, lab, c, ls in [("absn", "numerics", "#777", ":"), ("absp", "policy lag", "#bbb", "--"), ("absh", "harness lag", "#fff", "-")]:
        ax.plot(xs, [s.get(k) if s.get(k) else np.nan for s in st[:last]], ls, color=c, lw=1.3, label=lab)
    for e in json.load(open(os.path.join(RES, f"A_s{r['config']['seed']}.json")))["edits"]:
        if e["harness"]:
            ax.axvline(e["step"], color="#fff", lw=0.6, alpha=0.4)
    pos = [s[k] for s in st for k in ("absn", "absp", "absh") if s.get(k)]
    ax.set_yscale("log")
    ax.set_ylim(min(pos) * 0.8, max(pos) * 1.25)
    ax.set_xlim(0, len(st))
    ax.grid(color="#1a1a1a")
    ax.set_title("mean |term| on stale tokens (lines: harness edits)", fontsize=8, loc="left", color="#bbb")
    ax.set_xlabel("RL step", color="#888")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("L")
    plt.close(fig)
    return im


def main():
    r = pick()
    ntok = min(len(r["demo"]["tokens"]), 140)
    frames = [frame(r, u, ntok) for u in list(range(1, ntok + 1, 4)) + [ntok] * 12]
    out = os.path.join(ROOT, "docs", "screenshots", "demo.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main()
