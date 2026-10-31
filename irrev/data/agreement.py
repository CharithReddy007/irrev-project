"""Quality control (Section 4.2): two annotators label the same random 100 pairs blind.

Step 1 - make the sheet (automatic labels hidden):
    python -m irrev.data.agreement sample --data data/dataset.jsonl --n 100 --out data/qc_sheet.csv
Step 2 - annotators A and B each fill their own column, without talking to each other,
         and without looking at data/qc_key.json.
Step 3 - compute agreement:
    python -m irrev.data.agreement score --sheet data/qc_sheet_filled.csv --key data/qc_key.json

Accepted spellings in the sheet: r / p / i, or the full label names.
"""
from __future__ import annotations

import argparse
import csv
import json
import random

from sklearn.metrics import cohen_kappa_score, confusion_matrix

ALIASES = {"r": "reversible", "p": "partially_reversible", "i": "irreversible",
           "reversible": "reversible", "partial": "partially_reversible",
           "partially_reversible": "partially_reversible", "irreversible": "irreversible"}
ORDER = ["reversible", "partially_reversible", "irreversible"]


def sample(data: str, n: int, out: str, seed: int = 7) -> None:
    recs = [json.loads(l) for l in open(data)]
    rng = random.Random(seed)
    # stratify by environment so both testbeds are checked
    envs = sorted({r["environment"] for r in recs})
    picked = []
    for e in envs:
        sub = [r for r in recs if r["environment"] == e]
        picked += rng.sample(sub, min(len(sub), round(n * len(sub) / len(recs))))
    rng.shuffle(picked)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qc_id", "environment", "state_text", "action_text", "annotator_A", "annotator_B"])
        for i, r in enumerate(picked):
            w.writerow([i, r["environment"], r["state_text"], r["action_text"], "", ""])
    key = out.replace("sheet.csv", "key.json") if out.endswith("sheet.csv") else out + ".key.json"
    json.dump({str(i): {"id": r["id"], "auto": r["label"]} for i, r in enumerate(picked)}, open(key, "w"), indent=1)
    print(f"sheet: {out} ({len(picked)} rows); hidden key: {key}")


def score(sheet: str, key: str) -> dict:
    k = json.load(open(key))
    A, B, auto = [], [], []
    for row in csv.DictReader(open(sheet)):
        a, b = row["annotator_A"].strip().lower(), row["annotator_B"].strip().lower()
        if not a or not b:
            continue
        A.append(ALIASES[a]); B.append(ALIASES[b]); auto.append(k[row["qc_id"]]["auto"])
    res = {
        "n": len(A),
        "kappa_A_B": cohen_kappa_score(A, B),
        "kappa_A_auto": cohen_kappa_score(A, auto),
        "kappa_B_auto": cohen_kappa_score(B, auto),
        "raw_agreement_A_B": sum(x == y for x, y in zip(A, B)) / len(A),
        "confusion_A_vs_auto (rows=A, cols=auto)": confusion_matrix(A, auto, labels=ORDER).tolist(),
    }
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("sample"); s1.add_argument("--data"); s1.add_argument("--n", type=int, default=100); s1.add_argument("--out")
    s2 = sp.add_parser("score"); s2.add_argument("--sheet"); s2.add_argument("--key")
    a = ap.parse_args()
    sample(a.data, a.n, a.out) if a.cmd == "sample" else score(a.sheet, a.key)
