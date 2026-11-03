"""Charts and the results table, regenerated from results/*.json only. Black and white."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
SHOTS = os.path.join(ROOT, "docs", "screenshots")
ARMS = {"A": "A discard stale", "B": "B naive reuse", "C": "C re-render, no IS",
        "D": "D harnesslag", "E": "E re-render, one cap"}
STYLE = {"A": ("#ffffff", "-", "o"), "B": ("#8a8a8a", ":", "x"), "C": ("#bdbdbd", "--", "s"),
         "D": ("#ffffff", "-", "D"), "E": ("#6e6e6e", "-.", "^")}
plt.rcParams.update({"figure.facecolor": "#000", "axes.facecolor": "#000", "savefig.facecolor": "#000",
                     "axes.edgecolor": "#555", "axes.labelcolor": "#ddd", "text.color": "#ddd",
                     "xtick.color": "#999", "ytick.color": "#999", "grid.color": "#222", "font.family": "monospace",
                     "font.size": 9, "axes.grid": True, "legend.frameon": False})


def load():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "*_s*.json"))):
        r = json.load(open(p))
        runs[(r["config"]["arm"], r["config"]["seed"])] = r
    return runs


def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), float("nan")
    return float(xs.mean()), float(student_t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))


def summarize(runs):
    out = {"arms": {}, "seeds": sorted({s for _, s in runs})}
    for arm in ARMS:
        rs = [runs[k] for k in sorted(runs) if k[0] == arm]
        fin = [r["final"]["acc"] for r in rs if "final" in r]
        curves = [[e["acc"] for e in r["evals"]] for r in rs]
        stale = [s for r in rs for s in r["steps"] if s.get("stale_items")]
        out["arms"][arm] = {
            "runs": len(rs), "complete": len(fin), "final_acc": fin, "final_mean_ci": ci95(fin) if fin else None,
            "final_n": next((r["final"]["n"] for r in rs if "final" in r), None),
            "curve_x": [e["fresh_rollouts"] for e in rs[0]["evals"]] if rs else [],
            "curve_mean": np.mean([c[:min(map(len, curves))] for c in curves], 0).tolist() if curves else [],
            "clip_share": float(np.mean([s["clip"] for s in stale if s.get("clip") is not None])) if stale else None,
            "dropped_nonrenderable": int(sum(s.get("dropped_nonrenderable", 0) for r in rs for s in r["steps"])),
            "dropped_old_harness": int(sum(s.get("dropped_old_harness", 0) for r in rs for s in r["steps"])),
            "cpu_hours": float(sum(r["time"]["total"] for r in rs) / 3600 * rs[0]["config"]["threads"]) if rs else 0,
            "wall_hours": float(sum(r["time"]["total"] for r in rs) / 3600),
        }
    out["edits"] = {s: [e for e in runs[("A", s)]["edits"]] for (a, s) in runs if a == "A"}
    a, d, b = (out["arms"][k] for k in "ADB")
    if a["curve_mean"] and d["curve_mean"]:
        target = a["curve_mean"][-1]
        hit = next((x for x, y in zip(d["curve_x"], d["curve_mean"]) if y >= target), None)
        out["hypothesis"] = {
            "A_final_curve_acc": target,
            "D_rollouts_to_match_A": hit, "A_rollouts": a["curve_x"][-1],
            "rollout_saving": (1 - hit / a["curve_x"][-1]) if hit else None,
            "D_minus_B_final_points": 100 * (d["final_mean_ci"][0] - b["final_mean_ci"][0])
            if d["final_mean_ci"] and b["final_mean_ci"] else None,
        }
    return out


def edit_steps(runs, seed):
    r = runs.get(("A", seed))
    return [e["step"] for e in r["edits"] if e["harness"]] if r else []


def plot_curves(runs, summ):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for arm, (c, ls, mk) in STYLE.items():
        rs = [runs[k] for k in sorted(runs) if k[0] == arm]
        if not rs:
            continue
        n = min(len(r["evals"]) for r in rs)
        x = [e["fresh_rollouts"] for e in rs[0]["evals"][:n]]
        ys = np.array([[e["acc"] for e in r["evals"][:n]] for r in rs]) * 100
        ax.plot(x, ys.mean(0), ls, color=c, marker=mk, ms=4, lw=1.4 if arm == "D" else 1, label=ARMS[arm])
        if len(rs) > 1:
            ax.fill_between(x, ys.min(0), ys.max(0), color=c, alpha=0.06)
    ax.set_xlabel("fresh rollouts")
    ax.set_ylabel(f"test acc % ({next(iter(runs.values()))['config']['test_n']}-problem subset)")
    ax.set_title("accuracy vs fresh rollouts (mean of seeds, band = min..max)", loc="left")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(SHOTS, "accuracy.png"), dpi=150)


def plot_mismatch(runs, summ, arm="D"):
    seeds = sorted(s for a, s in runs if a == arm)
    if not seeds:
        return
    fig, axes = plt.subplots(len(seeds), 1, figsize=(7, 1.9 * len(seeds) + 0.4), sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], seeds):
        st = runs[(arm, s)]["steps"]
        x = [r["step"] for r in st]
        for key, lab, c, ls in [("absn", "numerics", "#777", ":"), ("absp", "policy lag", "#bbb", "--"),
                                ("absh", "harness lag", "#fff", "-")]:
            ax.plot(x, [r.get(key) if r.get(key) is not None else np.nan for r in st], ls, color=c, lw=1.2, label=lab)
        for e in edit_steps(runs, s):
            ax.axvline(e, color="#fff", lw=0.6, alpha=0.5)
            ax.text(e + 0.3, ax.get_ylim()[1] * 0.85, "edit", fontsize=7, color="#aaa")
        ax.set_yscale("log")
        ax.set_ylabel(f"seed {s}\nmean |term|")
    axes[0, 0].set_title(f"per-token log-ratio terms on stale tokens, arm {arm} (vertical lines: accepted harness edits)",
                         loc="left", fontsize=8)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="lower right")
    axes[-1, 0].set_xlabel("RL step")
    fig.tight_layout()
    fig.savefig(os.path.join(SHOTS, "mismatch.png"), dpi=150)


def plot_final(summ):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, (arm, v) in enumerate(summ["arms"].items()):
        if not v["final_acc"]:
            continue
        m, h = v["final_mean_ci"]
        ax.bar(i, m * 100, color="#fff" if arm == "D" else "#555", width=0.6)
        ax.errorbar(i, m * 100, yerr=0 if np.isnan(h) else h * 100, color="#aaa", capsize=4)
        ax.scatter([i] * len(v["final_acc"]), np.array(v["final_acc"]) * 100, color="#000", edgecolor="#ddd", zorder=3, s=14)
    ax.set_xticks(range(len(ARMS)), [ARMS[a] for a in ARMS], fontsize=7)
    ax.set_ylabel("GSM8K test acc %")
    n = next((v["final_n"] for v in summ["arms"].values() if v["final_n"]), "?")
    ax.set_title(f"final accuracy on the {n}-problem test split (bar = mean, whisker = 95% CI over seeds, dots = seeds)",
                 loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SHOTS, "final.png"), dpi=150)


def table(summ):
    rows = ["| arm | final test acc (mean ± 95% CI) | per seed | stale tokens clipped | dropped non-renderable | CPU-hours |",
            "|---|---|---|---|---|---|"]
    for arm, v in summ["arms"].items():
        if not v["final_acc"]:
            continue
        m, h = v["final_mean_ci"]
        clip = "n/a (no IS)" if arm in "BC" else f"{100 * v['clip_share']:.2f}%"
        rows.append(f"| {ARMS[arm]} | {100 * m:.2f} ± {100 * h:.2f} | "
                    f"{', '.join(f'{100 * x:.2f}' for x in v['final_acc'])} | {clip} | "
                    f"{v['dropped_nonrenderable']} | {v['cpu_hours']:.1f} |")
    return "\n".join(rows)


def plot_method():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    box = dict(boxstyle="square,pad=0.45", fc="#000", ec="#fff", lw=1)
    nodes = [(0.9, 2.6, "raw trace\nactions + tool results\n+ sampler log-probs"),
             (0.9, 0.8, "old harness\nlog mu(old weights, old ctx)"),
             (4.0, 2.6, "trainer, old ctx\nlog pi(old, old)"),
             (4.0, 0.8, "trainer, old ctx\nlog pi(current, old)"),
             (7.1, 1.7, "re-render under\ncurrent harness\nlog pi(current, new)")]
    for x, y, t in nodes:
        ax.text(x, y, t, ha="center", va="center", color="#fff", fontsize=8, bbox=box)
    arr = dict(arrowstyle="->", color="#bbb", lw=1)
    for (a, b), lab in [(((1.9, 0.8), (3.0, 2.5)), "numerics"), (((4.0, 2.15), (4.0, 1.25)), "policy lag"),
                        (((5.05, 0.9), (6.05, 1.55)), "harness lag")]:
        ax.annotate("", b, a, arrowprops=arr)
        ax.text((a[0] + b[0]) / 2 + 0.12, (a[1] + b[1]) / 2, lab, color="#fff", fontsize=8, style="italic")
    ax.text(9.95, 0.2, "sum of the three = total log-ratio, token by token\n"
            "drop traces whose tool results change under the edit\n"
            "D: center numerics, truncate exp(policy) and exp(harness) separately",
            ha="right", va="bottom", color="#999", fontsize=7.5)
    fig.savefig(os.path.join(SHOTS, "method.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = load()
    os.makedirs(SHOTS, exist_ok=True)
    summ = summarize(runs)
    json.dump(summ, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    open(os.path.join(RES, "summary.md"), "w").write(table(summ) + "\n")
    readme = os.path.join(ROOT, "README.md")
    txt = open(readme).read()
    a, b = "<!-- results:start -->", "<!-- results:end -->"
    if a in txt and b in txt:
        open(readme, "w").write(txt.split(a)[0] + a + "\n" + table(summ) + "\n" + b + txt.split(b)[1])
    plot_curves(runs, summ)
    plot_mismatch(runs, summ)
    plot_final(summ)
    plot_method()
    print(table(summ))
    print(json.dumps(summ.get("hypothesis"), indent=1))


if __name__ == "__main__":
    main()
