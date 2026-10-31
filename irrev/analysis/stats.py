"""Statistics helpers (Section 7.4): metrics, bootstrap CIs, McNemar's test."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, proba=None, labels=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "accuracy": float((y_true == y_pred).mean()),
    }
    if proba is not None and len(np.unique(y_true)) > 1:
        proba = np.asarray(proba)
        try:
            if proba.ndim == 1 or proba.shape[1] == 2:
                p1 = proba if proba.ndim == 1 else proba[:, 1]
                out["auroc"] = float(roc_auc_score(y_true, p1))
            else:
                out["auroc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                                   labels=labels))
        except ValueError:
            out["auroc"] = float("nan")
    return out


def bootstrap_ci(metric_fn, *arrays, groups=None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of metric_fn(*arrays).

    groups: optional cluster ids (e.g. scenario_id). If given, whole clusters are resampled
    (cluster bootstrap), which is the honest choice when minimal pairs share a scenario.
    Returns (point_estimate, low, high).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = metric_fn(*arrays)
    stats = []
    if groups is None:
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                stats.append(metric_fn(*[a[idx] for a in arrays]))
            except ValueError:
                continue
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in pick])
            try:
                stats.append(metric_fn(*[a[idx] for a in arrays]))
            except ValueError:
                continue
    stats = np.asarray([s for s in stats if s == s])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(stats) else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def mcnemar(a, b) -> dict:
    """Paired binary outcomes a[i], b[i] (e.g. careful in irreversible vs reversible version).
    Exact binomial test on discordant pairs; also the chi-square version with continuity correction."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n01 = int((~a & b).sum())   # a=0, b=1
    n10 = int((a & ~b).sum())   # a=1, b=0
    n = n01 + n10
    p_exact = binomtest(n10, n, 0.5).pvalue if n > 0 else 1.0
    stat = (abs(n10 - n01) - 1) ** 2 / n if n > 0 else 0.0
    return {"n_pairs": int(len(a)), "a1_b0": n10, "a0_b1": n01, "p_exact": float(p_exact),
            "chi2_cc": float(stat), "p_chi2": float(chi2.sf(stat, 1)) if n > 0 else 1.0}


def fmt_ci(point, lo, hi, digits=3) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
