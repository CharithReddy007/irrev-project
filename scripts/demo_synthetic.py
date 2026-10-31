"""End-to-end pipeline test on SYNTHETIC data (no GPU, no model).

It fakes the three things a model would produce - hidden states, stated answers and agent
behaviour - with known planted effects, then runs every analysis and figure script on them.
Use it to check your installation and to see what each output looks like.

    python scripts/demo_synthetic.py
All outputs go to results/demo/ and results/figures_demo/ and are watermarked SYNTHETIC.
NEVER report anything from these folders.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irrev.agent import policies                      # noqa: E402
from irrev.agent.caution import caution_measures      # noqa: E402
from irrev.agent.loop import run_episode              # noqa: E402

M = "demo"
R = Path(f"results/{M}")
DATA = str(R / "demo_dataset.jsonl")


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def fake_hidden(recs, rng, L=9, d=64):
    lab = {"reversible": -1, "partially_reversible": 0, "irreversible": 1}
    y = np.array([lab[r["label"]] for r in recs], float)
    verbs = sorted({r["verb"] for r in recs})
    vi = np.array([verbs.index(r["verb"]) for r in recs])
    X = rng.normal(size=(len(recs), L, d)).astype(np.float32)
    strength = np.sin(np.linspace(0, np.pi, L)) * 1.8          # peaks in the middle layers
    for l in range(L):
        X[:, l, 0] += strength[l] * y
        X[:, l, 1:1 + min(5, len(verbs))] += np.eye(len(verbs))[vi][:, :5] * 2.0   # verb identity
    out = R / "hidden"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "last.npy", X.astype(np.float16))
    np.save(out / "mean.npy", (X + rng.normal(scale=0.5, size=X.shape)).astype(np.float16))
    json.dump([r["id"] for r in recs], open(out / "ids.json", "w"))


def fake_stated(recs, rng):
    with open(R / "stated.jsonl", "w") as f:
        for r in recs:
            ok = rng.random() < (0.85 if r["environment"] == "sandbox" else 0.7)
            s = r["label"] if ok else rng.choice(["reversible", "irreversible", "partially_reversible", "unparsed"])
            f.write(json.dumps({"id": r["id"], "stated": s, "true": r["label"]}) + "\n")


def mixed_policy(task, p_careful):
    pol = policies.careful(task) if random.random() < p_careful else policies.reckless(task)
    return pol


def fake_agent(tasks, tag, p_irr, p_rev, seed):
    random.seed(seed)
    (R / "agent").mkdir(parents=True, exist_ok=True)
    with open(R / "agent" / f"{tag}.jsonl", "w") as f:
        for t in tasks:
            p = p_irr if t["version"] == "irreversible" else p_rev
            ep = run_episode(t, mixed_policy(t, p))
            ep["measures"] = caution_measures(ep, t)
            ep["config"] = {"tag": tag}
            f.write(json.dumps(ep) + "\n")


def fake_knowledge(tasks, rng):
    pids = sorted({t["pair_id"] for t in tasks})
    with open(R / "knowledge.jsonl", "w") as f:
        for p in pids:
            k = rng.random() < 0.8
            f.write(json.dumps({"pair_id": p, "known_forced_choice": k, "known_stated": k and rng.random() < .9}) + "\n")


def main():
    rng = np.random.default_rng(0)
    # self-contained: builds its own small synthetic dataset + task list under results/demo/,
    # so it runs on a fresh checkout and never touches the real data/ folder
    R.mkdir(parents=True, exist_ok=True)
    from irrev.data.gen_sandbox import generate, balance_classes
    from irrev.data.build import scenario_split
    from irrev.agent.tasks import make_tasks
    recs = balance_classes(generate(300, seed=7), random.Random(7))
    main_split = scenario_split(recs, 1)
    alt = [scenario_split(recs, 2 + k) for k in range(3)]
    for r in recs:
        r["split"] = main_split[r["scenario_id"]]
        for k in range(3):
            r[f"split_{k}"] = alt[k][r["scenario_id"]]
    with open(DATA, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tasks = make_tasks(13, seed=0)
    R.mkdir(parents=True, exist_ok=True)
    fake_hidden(recs, rng)
    fake_stated(recs, rng)
    # planted effect: the agent is only slightly more careful when the action is irreversible
    fake_agent(tasks, "neutral", p_irr=0.30, p_rev=0.20, seed=1)
    fake_agent(tasks, "careful", p_irr=0.65, p_rev=0.45, seed=2)
    for s in (-4, -2, 0, 2, 4):
        fake_agent(tasks, f"neutral_steer_L4_probe_s{s:+g}", p_irr=min(.95, max(.02, .3 + .12 * s)), p_rev=.2, seed=10 + s)
        fake_agent(tasks, f"neutral_steer_L4_rand0_s{s:+g}", p_irr=.3, p_rev=.2, seed=30 + s)
    fake_knowledge(tasks, rng)

    common = ["--model", M, "--n-boot", "200", "--jobs", "2", "--data", DATA]
    sh("irrev.probes.run", *common, "--task", "binary")
    sh("irrev.probes.run", *common, "--task", "three")
    sh("irrev.probes.run", *common, "--task", "binary", "--position", "mean")
    sh("irrev.probes.run", *common, "--task", "binary", "--train-subset", "minimal")
    sh("irrev.probes.run", *common, "--task", "binary", "--horizon", "1")
    sh("irrev.analysis.compare_rq2", "--model", M, "--task", "binary", "--data", DATA)
    sh("irrev.analysis.rq3_gap", "--model", M, "--tag", "neutral")
    sh("irrev.analysis.rq3_gap", "--model", M, "--tag", "careful")
    sh("irrev.analysis.rq4_steering", "--model", M, "--layer", "4")
    sh("irrev.analysis.figures", "--models", M, "--out", "results/figures_demo", "--watermark", "SYNTHETIC")
    print("\nDemo finished. Outputs in results/demo and results/figures_demo (SYNTHETIC - do not report).")


if __name__ == "__main__":
    main()
