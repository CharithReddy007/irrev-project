"""Step 6 (RQ4): the steering direction = mean(irreversible) - mean(reversible) hidden state
at a layer, computed on the TRAIN split only.

    python -m irrev.agent.direction --model qwen7b --layer 16

Also writes the norm of the direction relative to the average hidden-state norm, so the
strengths can be reported in interpretable units (strength 1 = one diff-of-means vector).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None, help="probe layer (>=1); default = best binary layer")
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--position", default="last")
    ap.add_argument("--env", default="sandbox", help="compute the direction on this environment's train rows")
    ap.add_argument("--hidden", default=None, help="folder with last.npy/mean.npy (default results/<model>/hidden)")
    a = ap.parse_args()
    if a.layer is None:
        res = json.load(open(f"results/{a.model}/probes/binary_{a.position}_hmain_all.json"))
        a.layer = max(1, res["summary"]["best_layer"])
    recs = [json.loads(l) for l in open(a.data)]
    X = np.load(f"{a.hidden or f'results/{a.model}/hidden'}/{a.position}.npy", mmap_mode="r")
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and (a.env == "all" or r["environment"] == a.env)]
    irr = [i for i in tr if recs[i]["label"] == "irreversible"]
    rev = [i for i in tr if recs[i]["label"] == "reversible"]
    H_irr = np.asarray(X[irr, a.layer], np.float32)
    H_rev = np.asarray(X[rev, a.layer], np.float32)
    direction = H_irr.mean(0) - H_rev.mean(0)
    mean_norm = float(np.linalg.norm(np.asarray(X[tr, a.layer], np.float32), axis=1).mean())
    out = Path(f"results/{a.model}/steering")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"direction_L{a.layer}.npz", direction=direction, layer=a.layer)
    info = {"layer": a.layer, "n_irr": len(irr), "n_rev": len(rev), "direction_norm": float(np.linalg.norm(direction)),
            "mean_hidden_norm": mean_norm, "ratio": float(np.linalg.norm(direction)) / mean_norm}
    json.dump(info, open(out / f"direction_L{a.layer}.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
