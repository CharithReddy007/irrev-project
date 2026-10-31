"""Run the agent on all tasks with a real model (RQ3), optionally with steering (RQ4).

    # RQ3, neutral prompt
    python -m irrev.agent.run --model qwen7b
    # ablation 5, explicit careful instruction
    python -m irrev.agent.run --model qwen7b --careful
    # RQ4 steering sweep at the best layer (direction from irrev.agent.direction)
    python -m irrev.agent.run --model qwen7b --steer-layer 16 --strengths -8 -4 -2 0 2 4 8
    python -m irrev.agent.run --model qwen7b --steer-layer 16 --strengths -8 -4 -2 0 2 4 8 --random-direction 0

Each configuration writes results/<model>/agent/<tag>.jsonl (one line per episode, with every
prompt and output) and is resumable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from irrev.agent.caution import caution_measures
from irrev.agent.loop import run_batched


def main():
    from irrev.models import lm as LMmod
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="data/agent_tasks.jsonl")
    ap.add_argument("--careful", action="store_true")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--steer-layer", type=int, default=None)
    ap.add_argument("--strengths", type=float, nargs="*", default=[0.0])
    ap.add_argument("--random-direction", type=int, default=None, help="seed of a random control direction")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16, help="episodes per batched model call (lower if OOM)")
    ap.add_argument("--versions", nargs="+", default=["reversible", "irreversible"],
                    help="e.g. --versions irreversible to save time in steering sweeps")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks)][: a.limit]
    tasks = [t for t in tasks if t["version"] in a.versions]
    LMmod.set_seed(0)
    lm = LMmod.load(a.model)

    vec = None
    if a.steer_layer is not None:
        d = np.load(f"results/{a.model}/steering/direction_L{a.steer_layer}.npz")
        vec = d["direction"]
        if a.random_direction is not None:
            rng = np.random.default_rng(a.random_direction)
            r = rng.normal(size=vec.shape)
            vec = r / np.linalg.norm(r) * np.linalg.norm(vec)      # same norm (Section 7.2)

    out_dir = Path(f"results/{a.model}/agent")
    out_dir.mkdir(parents=True, exist_ok=True)
    for strength in a.strengths:
        tag = "careful" if a.careful else "neutral"
        if a.steer_layer is not None:
            kind = f"rand{a.random_direction}" if a.random_direction is not None else "probe"
            tag += f"_steer_L{a.steer_layer}_{kind}_s{strength:+g}"
        path = out_dir / f"{tag}.jsonl"
        done = {json.loads(l)["task_id"] for l in open(path)} if path.exists() else set()

        todo = [t for t in tasks if t["task_id"] not in done]
        print(f"[{tag}] {len(done)} episodes already done, {len(todo)} to run", flush=True)
        by_id = {t["task_id"]: t for t in todo}

        def batch_policy(batch):
            with LMmod.steer(lm, a.steer_layer or 1, vec, strength):
                return LMmod.generate_batch(lm, batch, max_new_tokens=a.max_new_tokens)

        with open(path, "a") as f:
            def save(ep):
                t = by_id[ep["task_id"]]
                ep["measures"] = caution_measures(ep, t)
                ep["config"] = {"model": a.model, "tag": tag, "strength": strength, "layer": a.steer_layer,
                                "random_direction": a.random_direction, "batch_size": a.batch_size,
                                "precision": getattr(lm, "precision", None)}
                f.write(json.dumps(ep) + "\n")
                f.flush()
                m = ep["measures"]
                print(f"[{tag}] {t['task_id']} success={ep['success']} careful={m['careful']} "
                      f"irrev={m['took_irreversible']} parse_fail={ep['n_parse_fail']}", flush=True)

            run_batched(todo, batch_policy, batch_size=a.batch_size, careful_prompt=a.careful,
                        max_steps=a.max_steps, on_done=save)


if __name__ == "__main__":
    main()
