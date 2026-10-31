"""Merge environment files into one dataset, validate it, and assign scenario-level splits.

Splits
------
* test set = 20% of *scenarios* (never individual actions), fixed by --test-seed. Members of
  a minimal-pair group always share a scenario, so they always land on the same side.
* for the 3 repeated probe runs (Section 7.4) we also store three alternative splits
  split_0, split_1, split_2, each a different 80/20 scenario split.
* Stratification: scenarios are split within each environment separately so both testbeds
  appear in train and test.

    python -m irrev.data.build --inputs data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl \
        --out data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

REQUIRED = ["id", "environment", "state_text", "action_text", "label", "horizon", "source"]
VALID_LABELS = {"reversible", "partially_reversible", "irreversible"}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def validate(recs: list[dict]) -> None:
    ids = set()
    for r in recs:
        for k in REQUIRED:
            assert k in r, f"{r.get('id')} missing {k}"
        assert r["label"] in VALID_LABELS, (r["id"], r["label"])
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        ids.add(r["id"])


def scenario_split(recs: list[dict], seed: int, test_frac: float = 0.2) -> dict[str, str]:
    rng = random.Random(seed)
    by_env = defaultdict(set)
    for r in recs:
        by_env[r["environment"]].add(r["scenario_id"])
    assign = {}
    for env, scens in by_env.items():
        scens = sorted(scens)
        rng.shuffle(scens)
        n_test = max(1, round(len(scens) * test_frac))
        for i, s in enumerate(scens):
            assign[s] = "test" if i < n_test else "train"
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/dataset.jsonl")
    ap.add_argument("--test-seed", type=int, default=12345)
    a = ap.parse_args()
    recs = [r for p in a.inputs for r in load_jsonl(p)]
    validate(recs)
    main_split = scenario_split(recs, a.test_seed)
    alt = [scenario_split(recs, a.test_seed + 1 + k) for k in range(3)]
    for r in recs:
        r["split"] = main_split[r["scenario_id"]]
        for k in range(3):
            r[f"split_{k}"] = alt[k][r["scenario_id"]]
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} records -> {a.out}")
    for env in sorted({r["environment"] for r in recs}):
        sub = [r for r in recs if r["environment"] == env]
        print(env, len(sub), dict(Counter(r["label"] for r in sub)),
              "test:", sum(r["split"] == "test" for r in sub),
              "minimal pairs:", f"{sum(r.get('is_minimal_pair', False) for r in sub) / len(sub):.1%}")
    mp = sum(r.get("is_minimal_pair", False) for r in recs) / len(recs)
    print(f"overall minimal-pair share {mp:.1%} (brief requires >= 30%)")
    if len(recs) < 2000:
        print("WARNING: fewer than 2,000 pairs; the brief asks for at least 2,000 across both testbeds.")


if __name__ == "__main__":
    main()
