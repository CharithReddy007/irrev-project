"""RQ2: compare true label, stated answer and probe prediction for every pair.

The probe prediction for every pair comes from cross-fitting at the best layer (5-fold
GroupKFold by scenario over the whole dataset), so each pair is predicted by a probe that
never saw its scenario. We also report the comparison on the main test split only.

    python -m irrev.analysis.compare_rq2 --model qwen7b --task three
Outputs results/<model>/rq2_<task>.json and a per-pair CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from irrev.analysis.stats import classification_metrics, bootstrap_ci
from irrev.probes.run import load_data, fit_lr, predict


def crossfit(X, y, groups, layer, n_folds=5, seed=0):
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=n_folds).split(X[:, 0, :1], y, groups):
        sc, m, _ = fit_lr(np.asarray(X[tr, layer], np.float32), y[tr], groups[tr], seed=seed)
        pred[te] = predict(sc, m, np.asarray(X[te, layer], np.float32))[0]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--task", choices=["binary", "three"], default="three")
    ap.add_argument("--layer", type=int, default=None, help="default: best layer from the probe results")
    ap.add_argument("--position", default="last")
    ap.add_argument("--hidden", default=None, help="folder with last.npy/mean.npy (default results/<model>/hidden)")
    a = ap.parse_args()

    recs, X, y, keep, classes = load_data(a.data, a.hidden or f"results/{a.model}/hidden", a.position, None, a.task)
    if a.layer is None:
        res = json.load(open(f"results/{a.model}/probes/{a.task}_{a.position}_hmain_all.json"))
        a.layer = res["summary"]["best_layer"]
    stated = {json.loads(l)["id"]: json.loads(l)["stated"] for l in open(f"results/{a.model}/stated.jsonl")}

    idx = np.where(keep)[0]
    groups = np.array([recs[i]["scenario_id"] for i in idx])
    Xk = X[idx] if not isinstance(X, np.memmap) else np.asarray(X[idx])
    probe = crossfit(Xk, y[idx], groups, a.layer)

    def to_id(lab):
        if a.task == "binary" and lab == "partially_reversible":
            return -2   # stated "partial" on a binary item: counts as wrong, reported separately
        return classes.index(lab) if lab in classes else -1   # -1 = unparsed

    st = np.array([to_id(stated.get(recs[i]["id"], "unparsed")) for i in idx])
    yt = y[idx]
    parsed = st >= 0
    out = {"layer": a.layer, "task": a.task, "n": int(len(idx)),
           "unparsed": int((st == -1).sum()), "stated_partial_on_binary": int((st == -2).sum())}
    # accuracy of the stated answer (unparsed counted as wrong)
    out["stated_vs_true"] = classification_metrics(yt, np.where(parsed, st, -1), labels=list(range(len(classes))))
    out["stated_vs_true"]["accuracy_ci"] = list(bootstrap_ci(
        lambda a_, b_: float(np.mean(a_ == b_)), yt, st, groups=groups)[1:])
    out["probe_vs_true"] = classification_metrics(yt, probe)
    out["kappa_stated_probe"] = float(cohen_kappa_score(st[parsed], probe[parsed]))
    out["kappa_stated_true"] = float(cohen_kappa_score(st[parsed], yt[parsed]))
    out["kappa_probe_true"] = float(cohen_kappa_score(probe, yt))
    out["kappa_stated_probe_ci"] = list(bootstrap_ci(
        lambda s, p: cohen_kappa_score(s, p), st[parsed], probe[parsed], groups=groups[parsed])[1:])
    # the four agreement cells that matter for the story
    both = (st == yt) & (probe == yt)
    out["cells"] = {
        "both_correct": int(both.sum()),
        "probe_correct_stated_wrong": int(((probe == yt) & (st != yt)).sum()),
        "stated_correct_probe_wrong": int(((st == yt) & (probe != yt)).sum()),
        "both_wrong": int(((st != yt) & (probe != yt)).sum()),
    }
    out["confusion_stated (rows=true)"] = confusion_matrix(yt[parsed], st[parsed], labels=range(len(classes))).tolist()
    out["confusion_probe (rows=true)"] = confusion_matrix(yt, probe, labels=range(len(classes))).tolist()
    mp = np.array([recs[i].get("is_minimal_pair", False) for i in idx])
    if mp.any():
        out["minimal_pairs"] = {"stated_acc": float(np.mean(st[mp] == yt[mp])),
                                "probe_acc": float(np.mean(probe[mp] == yt[mp]))}
    Path(f"results/{a.model}").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"results/{a.model}/rq2_{a.task}.json", "w"), indent=1)
    with open(f"results/{a.model}/rq2_{a.task}_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "environment", "verb", "true", "stated", "probe"])
        name = lambda v: classes[v] if v >= 0 else ("partial" if v == -2 else "unparsed")
        for j, i in enumerate(idx):
            w.writerow([recs[i]["id"], recs[i]["environment"], recs[i]["verb"], classes[yt[j]], name(st[j]), classes[probe[j]]])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
