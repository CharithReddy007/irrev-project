"""Figures and tables required by the brief (Section 8):
  fig_probe_layers   probe accuracy vs layer, with baselines           (RQ1)
  fig_three_way      stated vs probed vs behaviour comparison          (RQ2/RQ3)
  fig_gap            the blindness gap with 95% CIs                    (headline)
  table_ablation     ablation table (LaTeX)                            (Section 7.3)
  fig_steering       steering curve, probe vs random direction         (RQ4, bonus)

    python -m irrev.analysis.figures --models qwen7b llama8b --out results/figures

Static paper figures: one axis per panel, recessive grid, fixed series colours, and every
series also distinguished by line style / marker so identity never relies on colour alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fixed categorical order (validated default palette, light mode): blue, orange, aqua, yellow
C = {"probe": "#2a78d6", "probe_mp": "#eb6834", "control": "#1baf7a", "shuffled": "#eda100",
     "base": "#6b6a64", "text": "#2b2b2a", "grid": "#e4e3dc"}
_BASE_NAMES = {"qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B", "qwen14b": "Qwen2.5-14B"}


class _Names(dict):
    """'qwen7b-4bit' -> 'Qwen2.5-7B (4-bit)'."""
    def get(self, m, default=None):
        base, _, suf = m.rpartition("-")
        if base in _BASE_NAMES:
            return f"{_BASE_NAMES[base]} ({suf.replace('bit', '-bit')})"
        return _BASE_NAMES.get(m, default if default is not None else m)


MODEL_NAMES = _Names()


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["text"], "xtick.color": C["text"],
        "ytick.color": C["text"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    })


def save(fig, out, name, watermark=None):
    if watermark:
        fig.text(0.5, 0.5, watermark, ha="center", va="center", fontsize=28, color="#d0cfc8",
                 alpha=0.5, rotation=20, zorder=0)
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out) / f"{name}.pdf")
    fig.savefig(Path(out) / f"{name}.png")
    plt.close(fig)
    print("wrote", Path(out) / f"{name}.pdf")


# ------------------------------------------------------------------ RQ1
def fig_probe_layers(models, out, task="binary", position="last", wm=None):
    files = [(m, f"results/{m}/probes/{task}_{position}_hmain_all.json") for m in models]
    files = [(m, f) for m, f in files if os.path.exists(f)]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(3.4 * len(files), 2.6), squeeze=False, sharey=True)
    for ax, (m, f) in zip(axes[0], files):
        res = json.load(open(f))
        s = res["summary"]
        L = [r["layer"] for r in s["per_layer"]]
        def band(key, color, ls, marker, label):
            mu = np.array([r.get(f"{key}_mean", np.nan) for r in s["per_layer"]])
            sd = np.array([r.get(f"{key}_std", 0) for r in s["per_layer"]])
            ax.plot(L, mu, color=color, ls=ls, marker=marker, ms=3, label=label)
            ax.fill_between(L, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        band("probe_all", C["probe"], "-", "o", "probe (all test)")
        if "probe_minimal_pairs_mean" in s["per_layer"][0]:
            band("probe_minimal_pairs", C["probe_mp"], "-", "s", "probe (minimal pairs)")
        band("control_all", C["control"], "--", "^", "control task")
        band("shuffled_all", C["shuffled"], ":", "v", "shuffled labels")
        b = s["baselines"]
        for key, ls, lab in [("majority_all_mean", (0, (1, 1)), "majority"),
                             ("lexical_action_minimal_pairs_mean", "-.", "lexical (action), min. pairs"),
                             ("lexical_state_minimal_pairs_mean", (0, (5, 2)), "lexical (state+action), min. pairs")]:
            if key in b:
                ax.axhline(b[key], color=C["base"], ls=ls, lw=1, label=lab)
        ax.set_title(f"{MODEL_NAMES.get(m, m)} ({task})")
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylim(0.3, 1.02)
    axes[0][0].set_ylabel("balanced accuracy (test)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3 if len(files) == 1 else 4, frameon=False)
    save(fig, out, f"fig_probe_layers_{task}_{position}", wm)


# ------------------------------------------------------------------ RQ2/RQ3
def _get(path, *keys, default=None):
    if not os.path.exists(path):
        return default
    d = json.load(open(path))
    for k in keys:
        if d is None or k not in d:
            return default
        d = d[k]
    return d


def fig_three_way(models, out, wm=None):
    """For each model: probe accuracy, stated accuracy (binary items), K, A. Same 0-1 scale."""
    bars = []
    for m in models:
        bars.append((m, [
            ("Probe\n(bal. acc.)", _get(f"results/{m}/rq2_binary.json", "probe_vs_true", "balanced_accuracy")),
            ("Stated\n(bal. acc.)", _get(f"results/{m}/rq2_binary.json", "stated_vs_true", "balanced_accuracy")),
            ("Stated on\ntask pairs (K)", _get(f"results/{m}/rq3_neutral.json", "K_forced_choice", "rate")),
            ("Acts on\ntask pairs (A)", _get(f"results/{m}/rq3_neutral.json", "A_score", "rate")),
        ]))
    bars = [(m, b) for m, b in bars if any(v is not None for _, v in b)]
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(3.8 + 0.6 * len(bars), 2.6))
    labels = [l for l, _ in bars[0][1]]
    x = np.arange(len(labels))
    w = 0.8 / len(bars)
    hatches = ["", "////", "\\\\\\\\", "xx"]
    cols = [C["probe"], C["probe_mp"], C["control"]]
    for k, (m, b) in enumerate(bars):
        vals = [v if v is not None else 0 for _, v in b]
        rects = ax.bar(x + (k - (len(bars) - 1) / 2) * w, vals, w * 0.92, color=cols[k % 3], hatch=hatches[k % 4],
                       edgecolor="white", linewidth=0.5, label=MODEL_NAMES.get(m, m))
        for r, v in zip(rects, b):
            if v[1] is not None:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{v[1]:.2f}", ha="center", va="bottom",
                        fontsize=6, color=C["text"])
    ax.set_xticks(x, labels)
    # chance level of each measure: 0.5 for balanced accuracy, 0.25 for K (correct in both orders)
    for xi, ch in zip(x[:3], [0.5, 0.5, 0.25]):
        ax.hlines(ch, xi - 0.45, xi + 0.45, colors=C["text"], linestyles=":", linewidth=1)
    ax.plot([], [], ":", color=C["text"], linewidth=1, label="chance")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("What the model encodes, says and does")
    save(fig, out, "fig_three_way", wm)


def fig_gap(models, out, wm=None):
    rows = []
    for m in models:
        for tag in ("neutral", "careful"):
            g = _get(f"results/{m}/rq3_{tag}.json", "gap_forced_choice_score")
            if g:
                rows.append((f"{MODEL_NAMES.get(m, m)}\n{tag} prompt", g["value"], g["ci"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 0.5 + 0.45 * len(rows)))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, (lo, hi)) in zip(y, rows):
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=C["probe"], ms=5, capsize=3, lw=1.4)
        ax.text(hi + 0.02, yi, f"{v:+.2f}", va="center", fontsize=7, color=C["text"])
    ax.axvline(0, color=C["base"], lw=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("blindness gap  K - A  (95% CI)")
    ax.set_xlim(-1, 1.15)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_gap", wm)


# ------------------------------------------------------------------ ablation table
ABLATIONS = [
    ("binary_last_hmain_all", "Main (binary, last token, all data)"),
    ("binary_mean_hmain_all", "Token position: mean over action"),
    ("binary_last_hmain_minimal", "Train on minimal pairs only"),
    ("binary_last_hmain_all_sandbox2scienceworld", "Transfer: sandbox $\\rightarrow$ ScienceWorld"),
    ("binary_last_hmain_all_scienceworld2sandbox", "Transfer: ScienceWorld $\\rightarrow$ sandbox"),
    ("binary_last_h1_all", "Horizon 1"),
    ("binary_last_h2_all", "Horizon 2"),
    ("three_last_hmain_all", "Three-class labels"),
]


def table_ablation(models, out):
    lines = ["\\begin{table}[t]", "\\centering\\small", "\\begin{tabular}{l" + "cc" * len(models) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{MODEL_NAMES.get(m, m)}}}" for m in models) + " \\\\",
             "Setting & " + " & ".join("All & Min.\\ pairs" for _ in models) + " \\\\", "\\midrule"]
    any_row = False
    for tag, name in ABLATIONS:
        cells = []
        for m in models:
            f = f"results/{m}/probes/{tag}.json"
            if not os.path.exists(f):
                cells += ["--", "--"]
                continue
            b = json.load(open(f))["summary"]["best"]
            fmt = lambda k: (f"{b[k + '_mean']:.2f}$\\pm${b.get(k + '_std', 0):.2f}" if k + "_mean" in b else "--")
            cells += [fmt("probe_all"), fmt("probe_minimal_pairs")]
            any_row = True
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Probe balanced accuracy at the best layer (mean $\\pm$ std over 3 scenario splits).}",
              "\\label{tab:ablation}", "\\end{table}"]
    if any_row:
        Path(out).mkdir(parents=True, exist_ok=True)
        open(Path(out) / "table_ablation.tex", "w").write("\n".join(lines))
        print("wrote", Path(out) / "table_ablation.tex")


# ------------------------------------------------------------------ RQ4
def fig_steering(models, out, wm=None):
    for m in models:
        for f in glob.glob(f"results/{m}/rq4_L*.json"):
            res = json.load(open(f))
            fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharex=True)
            for direction, color, ls, mk in (("probe", C["probe"], "-", "o"), ("random", C["base"], "--", "s")):
                rows = [r for r in res["rows"] if r["direction"] == direction and r["version"] == "irreversible"]
                if not rows:
                    continue
                # average random seeds per strength
                S = sorted({r["strength"] for r in rows})
                car = [np.mean([r["careful"] for r in rows if r["strength"] == s]) for s in S]
                lo = [np.mean([r["careful_ci"][0] for r in rows if r["strength"] == s]) for s in S]
                hi = [np.mean([r["careful_ci"][1] for r in rows if r["strength"] == s]) for s in S]
                suc = [np.mean([r["success"] for r in rows if r["strength"] == s]) for s in S]
                lab = "probe direction" if direction == "probe" else "random direction (same norm)"
                axes[0].plot(S, car, color=color, ls=ls, marker=mk, ms=4, label=lab)
                axes[0].fill_between(S, lo, hi, color=color, alpha=0.15, lw=0)
                axes[1].plot(S, suc, color=color, ls=ls, marker=mk, ms=4, label=lab)
            axes[0].set_ylabel("careful rate (irreversible version)")
            axes[1].set_ylabel("task success rate")
            for ax in axes:
                ax.set_xlabel("steering strength")
                ax.set_ylim(-0.02, 1.02)
            axes[0].legend(frameon=False, loc="best")
            fig.suptitle(f"{MODEL_NAMES.get(m, m)}: steering at layer {res['layer']}", fontsize=9)
            save(fig, out, f"fig_steering_{m}_L{res['layer']}", wm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen7b", "llama8b"])
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--watermark", default=None, help="e.g. SYNTHETIC for demo data")
    a = ap.parse_args()
    style()
    for task in ("binary", "three"):
        fig_probe_layers(a.models, a.out, task, "last", a.watermark)
    fig_three_way(a.models, a.out, a.watermark)
    fig_gap(a.models, a.out, a.watermark)
    table_ablation(a.models, a.out)
    fig_steering(a.models, a.out, a.watermark)


if __name__ == "__main__":
    main()
