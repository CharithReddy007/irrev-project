"""Step 3 (RQ1): linear probes on every layer, with all compulsory baselines and controls.

For each layer l:
  probe      L2 logistic regression on standardised hidden states (C picked by grouped CV on train)
  shuffled   same probe, labels randomly permuted on train            (should be ~chance)
  control    same probe, Hewitt & Liang control task: every verb gets a fixed random label
             -> selectivity = probe accuracy - control accuracy
Plus once per split:
  majority          majority class of train
  lexical_action    bag-of-words logistic regression on the action text only   (compulsory)
  lexical_state     bag-of-words on state + action text                         (extra, stricter)
Layer 0 (embeddings) is included automatically.

Everything is evaluated on the full test set, the minimal-pair subset of the test set, and
per environment. Repeated over 3 scenario splits (split_0..2), reported as mean and std.

Ablation switches:
  --position last|mean       (1) token position
  --train-subset minimal     (2) train on minimal pairs only
  --train-env/--test-env     (3) cross-environment transfer
  --horizon 1|2|3            (4) labelling horizon
  --task binary|three        (7) binary vs three-class

    python -m irrev.probes.run --model qwen7b --task binary
    python -m irrev.probes.run --model qwen7b --task three --position mean
    python -m irrev.probes.run --model qwen7b --task binary --train-env sandbox --test-env scienceworld
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from irrev.analysis.stats import classification_metrics, bootstrap_ci
from irrev.labeling.bfs import LABELS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)


# ------------------------------------------------------------------ data
def load_data(data_path: str, hidden_dir: str | None, position: str, horizon: int | None, task: str):
    recs = [json.loads(l) for l in open(data_path)]
    X = None
    if hidden_dir:
        ids = json.load(open(Path(hidden_dir) / "ids.json"))
        assert ids == [r["id"] for r in recs], "hidden-state rows do not match the dataset order"
        X = np.load(Path(hidden_dir) / f"{position}.npy", mmap_mode="r")
    labels = [r["labels_by_horizon"].get(str(horizon), r["label"]) if horizon else r["label"] for r in recs]
    keep = np.ones(len(recs), bool)
    if task == "binary":
        keep = np.array([l != "partially_reversible" for l in labels])
        classes = ["reversible", "irreversible"]
    else:
        classes = list(LABELS)
    y = np.array([classes.index(l) if l in classes else -1 for l in labels])
    return recs, X, y, keep, classes


def control_labels(recs, y, classes, seed=0):
    """Hewitt & Liang control task: each verb type gets a random label drawn from the label
    distribution; every occurrence of that verb gets that label."""
    rng = np.random.default_rng(seed)
    valid = y[y >= 0]
    dist = np.bincount(valid, minlength=len(classes)) / len(valid)
    verbs = sorted({r["verb"] for r in recs})
    vmap = {v: int(rng.choice(len(classes), p=dist)) for v in verbs}
    return np.array([vmap[r["verb"]] for r in recs])


# ------------------------------------------------------------------ fitting
def fit_lr(Xtr, ytr, groups, C=None, n_folds=3, seed=0):
    """Standardise + L2 logistic regression. If C is None, choose it by grouped CV."""
    scaler = StandardScaler().fit(Xtr)
    Z = scaler.transform(Xtr)
    if C is None:
        n_groups = len(np.unique(groups))
        best, C = -1, C_GRID[0]
        if n_groups >= n_folds and len(np.unique(ytr)) > 1:
            gkf = GroupKFold(n_splits=n_folds)
            for c in C_GRID:
                scores = []
                for tr, va in gkf.split(Z, ytr, groups):
                    if len(np.unique(ytr[tr])) < 2:
                        continue
                    m = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(Z[tr], ytr[tr])
                    scores.append(classification_metrics(ytr[va], m.predict(Z[va]))["balanced_accuracy"])
                if scores and np.mean(scores) > best:
                    best, C = float(np.mean(scores)), c
    model = LogisticRegression(C=C, max_iter=3000, random_state=seed).fit(Z, ytr)
    return scaler, model, C


def predict(scaler, model, X):
    Z = scaler.transform(X)
    return model.predict(Z), model.predict_proba(Z)


def evaluate(y, pred, proba, recs_idx, recs, classes, n_boot):
    """Metrics on the full test set, minimal pairs, and per environment, with cluster bootstrap CIs."""
    out = {}
    subsets = {"all": np.arange(len(y))}
    mp = np.array([recs[i].get("is_minimal_pair", False) for i in recs_idx])
    if mp.any():
        subsets["minimal_pairs"] = np.where(mp)[0]
    envs = np.array([recs[i]["environment"] for i in recs_idx])
    for e in np.unique(envs):
        subsets[f"env:{e}"] = np.where(envs == e)[0]
    groups_all = np.array([recs[i]["scenario_id"] for i in recs_idx])
    labels = list(range(len(classes)))
    for name, idx in subsets.items():
        if len(idx) == 0:
            continue
        m = classification_metrics(y[idx], pred[idx], proba[idx] if proba is not None else None, labels=labels)
        if n_boot:
            from sklearn.metrics import balanced_accuracy_score
            pt, lo, hi = bootstrap_ci(lambda a, b: balanced_accuracy_score(a, b), y[idx], pred[idx],
                                      groups=groups_all[idx], n_boot=n_boot)
            m["balanced_accuracy_ci"] = [lo, hi]
        out[name] = m
    return out


def run_layer(l, X, y, yc, tr, te, groups, recs, classes, seed, n_boot):
    Xtr = np.asarray(X[tr, l], dtype=np.float32)
    Xte = np.asarray(X[te, l], dtype=np.float32)
    res = {"layer": int(l)}
    sc, m, C = fit_lr(Xtr, y[tr], groups[tr], seed=seed)
    p, pr = predict(sc, m, Xte)
    res["C"] = C
    res["probe"] = evaluate(y[te], p, pr, te, recs, classes, n_boot)
    res["test_pred"] = p.tolist()
    res["test_proba"] = pr.tolist()
    # shuffled labels (same C)
    rng = np.random.default_rng(seed + 1)
    ysh = y[tr].copy()
    rng.shuffle(ysh)
    sc2, m2, _ = fit_lr(Xtr, ysh, groups[tr], C=C, seed=seed)
    p2, pr2 = predict(sc2, m2, Xte)
    res["shuffled"] = evaluate(y[te], p2, pr2, te, recs, classes, 0)
    # control task (same C)
    if len(np.unique(yc[tr])) > 1:
        sc3, m3, _ = fit_lr(Xtr, yc[tr], groups[tr], C=C, seed=seed)
        p3, pr3 = predict(sc3, m3, Xte)
        res["control"] = evaluate(yc[te], p3, pr3, te, recs, classes, 0)
        res["selectivity"] = res["probe"]["all"]["accuracy"] - res["control"]["all"]["accuracy"]
    return res


def baselines(recs, y, tr, te, classes, n_boot):
    out = {}
    maj = np.bincount(y[tr]).argmax()
    pm = np.full(len(te), maj)
    out["majority"] = evaluate(y[te], pm, None, te, recs, classes, 0)
    for name, field in [("lexical_action", lambda r: r["action_text"]),
                        ("lexical_state", lambda r: r["state_text"] + " " + r["action_text"])]:
        vec = CountVectorizer(token_pattern=r"[A-Za-z_]+", lowercase=True, min_df=1)
        Xtr = vec.fit_transform([field(recs[i]) for i in tr])
        Xte = vec.transform([field(recs[i]) for i in te])
        best, bm = -1, None
        for c in C_GRID + (10.0,):
            mdl = LogisticRegression(C=c, max_iter=3000).fit(Xtr, y[tr])
            # choose C on train with a simple grouped split (no test peeking)
            groups = np.array([recs[i]["scenario_id"] for i in tr])
            gkf = GroupKFold(n_splits=3)
            s = []
            for a, b in gkf.split(Xtr, y[tr], groups):
                mm = LogisticRegression(C=c, max_iter=3000).fit(Xtr[a], y[tr][a])
                s.append(classification_metrics(y[tr][b], mm.predict(Xtr[b]))["balanced_accuracy"])
            if np.mean(s) > best:
                best, bm = np.mean(s), mdl
        out[name] = evaluate(y[te], bm.predict(Xte), bm.predict_proba(Xte), te, recs, classes, n_boot)
    return out


def select_rows(recs, keep, split_field, env_train, env_test, train_subset):
    idx = np.where(keep)[0]
    tr = [i for i in idx if recs[i][split_field] == "train"]
    te = [i for i in idx if recs[i][split_field] == "test"]
    if env_train:   # cross-environment: train on all of env_train, test on all of env_test
        tr = [i for i in idx if recs[i]["environment"] == env_train]
        te = [i for i in idx if recs[i]["environment"] == env_test]
    if train_subset == "minimal":
        tr = [i for i in tr if recs[i].get("is_minimal_pair", False)]
    return np.array(tr), np.array(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--task", choices=["binary", "three"], default="binary")
    ap.add_argument("--position", choices=["last", "mean"], default="last")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--train-subset", choices=["all", "minimal"], default="all")
    ap.add_argument("--train-env", default=None)
    ap.add_argument("--test-env", default=None)
    ap.add_argument("--splits", nargs="+", default=["split_0", "split_1", "split_2"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    hidden = a.hidden or f"results/{a.model}/hidden"
    recs, X, y, keep, classes = load_data(a.data, hidden, a.position, a.horizon, a.task)
    yc = control_labels(recs, y, classes)
    groups = np.array([r["scenario_id"] for r in recs])
    layers = a.layers if a.layers is not None else list(range(X.shape[1]))
    tag = a.tag or (f"{a.task}_{a.position}_h{a.horizon or 'main'}_{a.train_subset}"
                    + (f"_{a.train_env}2{a.test_env}" if a.train_env else ""))
    out_dir = Path(f"results/{a.model}/probes")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = a.splits if not a.train_env else ["cross_env"]

    all_runs = []
    for si, split in enumerate(splits):
        t0 = time.time()
        tr, te = select_rows(recs, keep, split if split != "cross_env" else "split", a.train_env, a.test_env,
                             a.train_subset)
        print(f"[{tag}] {split}: train {len(tr)}  test {len(te)}", flush=True)
        base = baselines(recs, y, tr, te, classes, a.n_boot)
        per_layer = Parallel(n_jobs=a.jobs)(
            delayed(run_layer)(l, X, y, yc, tr, te, groups, recs, classes, si, a.n_boot) for l in layers)
        all_runs.append({"split": split, "train_n": int(len(tr)), "test_n": int(len(te)),
                         "test_ids": [recs[i]["id"] for i in te], "baselines": base, "layers": per_layer})
        print(f"  done in {time.time() - t0:.0f}s; best layer "
              f"{max(per_layer, key=lambda r: r['probe']['all']['balanced_accuracy'])['layer']}", flush=True)

    summary = summarise(all_runs)
    json.dump({"config": vars(a), "classes": classes, "runs": all_runs, "summary": summary},
              open(out_dir / f"{tag}.json", "w"))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=1))
    print("saved", out_dir / f"{tag}.json")


def summarise(runs):
    """Mean and std across splits, per layer, for the main numbers."""
    layers = [r["layer"] for r in runs[0]["layers"]]
    per_layer = []
    for li, l in enumerate(layers):
        row = {"layer": l}
        for kind in ["probe", "shuffled", "control"]:
            for subset in ["all", "minimal_pairs"]:
                vals = [r["layers"][li].get(kind, {}).get(subset, {}).get("balanced_accuracy") for r in runs]
                vals = [v for v in vals if v is not None]
                if vals:
                    row[f"{kind}_{subset}_mean"] = float(np.mean(vals))
                    row[f"{kind}_{subset}_std"] = float(np.std(vals))
        sel = [r["layers"][li].get("selectivity") for r in runs if r["layers"][li].get("selectivity") is not None]
        if sel:
            row["selectivity_mean"], row["selectivity_std"] = float(np.mean(sel)), float(np.std(sel))
        auc = [r["layers"][li]["probe"]["all"].get("auroc") for r in runs]
        row["probe_auroc_mean"] = float(np.nanmean([x for x in auc if x is not None])) if any(auc) else None
        per_layer.append(row)
    best = max(per_layer, key=lambda r: r["probe_all_mean"])
    base = {}
    for name in runs[0]["baselines"]:
        for subset in ["all", "minimal_pairs"]:
            vals = [r["baselines"][name].get(subset, {}).get("balanced_accuracy") for r in runs]
            vals = [v for v in vals if v is not None]
            if vals:
                base[f"{name}_{subset}_mean"] = float(np.mean(vals))
                base[f"{name}_{subset}_std"] = float(np.std(vals))
    return {"best_layer": best["layer"], "best": best, "baselines": base, "per_layer": per_layer}


if __name__ == "__main__":
    main()
