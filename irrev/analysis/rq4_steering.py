"""RQ4: caution and task success as a function of steering strength, probe direction vs
random direction of the same norm.

Causal evidence = careful behaviour rises with strength along the probe direction (monotone
trend, Spearman rho > 0 with CI excluding 0) AND the random direction shows no such trend,
AND task success has not collapsed at the strengths where caution rises.

    python -m irrev.analysis.rq4_steering --model qwen7b --layer 16
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np
from scipy.stats import spearmanr

from irrev.analysis.stats import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--prompt", default="neutral")
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(f"results/{a.model}/agent/{a.prompt}_steer_L{a.layer}_*_s*.jsonl")):
        m = re.search(r"_(probe|rand\d+)_s([+-]?[\d.]+)\.jsonl$", path)
        kind, s = m.group(1), float(m.group(2))
        eps = [json.loads(l) for l in open(path)]
        for version in ("irreversible", "reversible"):
            sub = [e for e in eps if e["version"] == version]
            if not sub:
                continue
            car = np.array([e["measures"]["careful"] for e in sub], float)
            suc = np.array([e["success"] for e in sub], float)
            pf = np.array([e["n_parse_fail"] for e in sub], float)
            _, lo, hi = bootstrap_ci(lambda x: float(np.mean(x)), car)
            rows.append({"direction": "probe" if kind == "probe" else "random", "seed": kind, "strength": s,
                         "version": version, "careful": float(car.mean()), "careful_ci": [lo, hi],
                         "success": float(suc.mean()), "parse_fail_per_ep": float(pf.mean()), "n": len(sub)})
    # also include the unsteered neutral run as strength 0 for both curves if present
    trend = {}
    for d in ("probe", "random"):
        for version in ("irreversible", "reversible"):
            sub = [r for r in rows if r["direction"] == d and r["version"] == version]
            if len(sub) >= 3:
                rho, p = spearmanr([r["strength"] for r in sub], [r["careful"] for r in sub])
                trend[f"{d}_{version}"] = {"spearman_rho": float(rho), "p": float(p), "n_strengths": len(sub)}
    out = {"layer": a.layer, "rows": sorted(rows, key=lambda r: (r["direction"], r["version"], r["strength"])),
           "trend": trend}
    json.dump(out, open(f"results/{a.model}/rq4_L{a.layer}.json", "w"), indent=1)
    print(json.dumps(out["trend"], indent=1))
    for r in out["rows"]:
        print(f"{r['direction']:6s} {r['seed']:6s} {r['version']:12s} s={r['strength']:+5.1f} careful={r['careful']:.2f} "
              f"[{r['careful_ci'][0]:.2f},{r['careful_ci'][1]:.2f}] success={r['success']:.2f} parsefail={r['parse_fail_per_ep']:.1f}")


if __name__ == "__main__":
    main()
