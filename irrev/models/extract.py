"""Step 2: run each (state, action) pair through the model once and save hidden states.

Saves, per model:
    results/<model>/hidden/last.npy   float16 [N, L+1, d]  hidden state of the last action token
    results/<model>/hidden/mean.npy   float16 [N, L+1, d]  mean over action tokens (ablation 1)
    results/<model>/hidden/ids.json   record ids in row order
    results/<model>/hidden/meta.json  model name, n_layers, d, prompt lengths, versions
Resumable: rows already written are skipped (tracked in progress.json).

    python -m irrev.models.extract --model qwen7b --data data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from irrev.models import lm as LMmod
from irrev.models.prompts import probe_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.data)]
    if a.limit:
        recs = recs[: a.limit]
    out = Path(a.out or f"results/{a.model}/hidden")
    out.mkdir(parents=True, exist_ok=True)

    LMmod.set_seed(0)
    lm = LMmod.load(a.model)
    L1, d = lm.n_layers + 1, lm.model.config.hidden_size
    N = len(recs)
    mode = "r+" if (out / "last.npy").exists() else "w+"
    last = np.lib.format.open_memmap(out / "last.npy", mode=mode, dtype=np.float16, shape=(N, L1, d))
    mean = np.lib.format.open_memmap(out / "mean.npy", mode=mode, dtype=np.float16, shape=(N, L1, d))
    prog_path = out / "progress.json"
    done = set(json.load(open(prog_path))) if prog_path.exists() else set()
    lengths = {}
    t0 = time.time()
    for i, r in enumerate(recs):
        if i in done:
            continue
        msgs = probe_messages(r)
        h_last, h_mean, n_tok = LMmod.extract_hidden(lm, msgs, r["action_text"])
        last[i], mean[i] = h_last.astype(np.float16), h_mean.astype(np.float16)
        lengths[r["id"]] = n_tok
        done.add(i)
        if (i + 1) % 50 == 0 or i == N - 1:
            last.flush(); mean.flush()
            json.dump(sorted(done), open(prog_path, "w"))
            rate = (time.time() - t0) / max(1, len(lengths))
            print(f"{i + 1}/{N}  {rate:.2f}s/pair", flush=True)
    json.dump([r["id"] for r in recs], open(out / "ids.json", "w"))
    import torch, transformers
    json.dump({"model": LMmod.MODELS.get(LMmod.parse_model_name(a.model)[0], a.model), "precision": getattr(lm, "precision", None), "n_layers": lm.n_layers, "hidden_size": d,
               "N": N, "torch": torch.__version__, "transformers": transformers.__version__,
               "max_prompt_tokens": max(lengths.values()) if lengths else None},
              open(out / "meta.json", "w"), indent=1)
    print("saved to", out)


if __name__ == "__main__":
    main()
