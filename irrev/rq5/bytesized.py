"""RQ5: is the model's uncertainty about the next state related to reversibility?

Data: ByteSized32-State-Prediction (github.com/cognitiveailab/GPT-simulator, data/test.zip ->
test.jsonl). No licence file is shown in the repository: use it for research only and do not
redistribute any part of it without asking the authors.

Format (checked on test.jsonl, 2,954 transitions, 31 games). Each line has
  current_state  {observation, look, inventory, taskDesc, lastAction, objects, max_UUID}
  action_state   same keys; action_state["lastAction"] is the action taken; this is the
                 state right after the action (before the world tick)
  tick_state     state after the world tick
  game, state_id, current_score_state, next_score_state

Labelling. There is no simulator API to search over, so we label transitions in two layers:
  1. no change in look+inventory+objects            -> reversible (nothing happened)
  2. otherwise by verb, using INVERSE_VERBS / IRREVERSIBLE_VERBS below, refined with the
     object diff (objects created or removed -> irreversible)
  3. anything else -> "unknown", which goes to the hand-labelling sheet
Then two annotators hand-label a sample of 100 as in Section 4.2. Report this honestly as a
weaker labelling method than the BFS used for the main testbeds.

Uncertainty. For each transition we ask the model to predict the next look + inventory text
and record (a) the mean token NLL of its own greedy prediction (its uncertainty) and (b) the
mean NLL of the true next state under teacher forcing (its surprise). AUROC of each score for
separating irreversible from reversible transitions is the RQ5 metric.

    python -m irrev.rq5.bytesized label --data GPT-simulator/data/test.jsonl --out data/bytesized_labeled.jsonl
    python -m irrev.rq5.bytesized score --model qwen7b --data data/bytesized_labeled.jsonl --n 600
    python -m irrev.rq5.bytesized analyse --model qwen7b
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

INVERSE_VERBS = {"open", "close", "take", "put", "move", "turn", "connect", "disconnect", "activate",
                 "deactivate", "rotate", "set", "change", "sit", "stand", "go", "drop", "pick", "wear", "remove"}
IRREVERSIBLE_VERBS = {"eat", "cook", "dice", "chop", "slice", "pour", "dig", "hammer", "feed", "burn",
                      "break", "drink", "mix", "plant", "water", "wash", "focus", "answer", "use", "sweep"}
NOOP_VERBS = {"look", "inventory", "examine", "read", "check"}


def _objs(state):
    def walk(o, acc):
        acc[o.get("uuid")] = (o.get("name"), json.dumps(o.get("properties", {}), sort_keys=True))
        for c in o.get("contains", []) or []:
            if isinstance(c, dict):
                walk(c, acc)
        return acc
    acc = {}
    for o in state.get("objects", []):
        walk(o, acc)
    return acc


def rule_label(rec) -> tuple[str, str]:
    cur, nxt = rec["current_state"], rec["action_state"]
    action = nxt.get("lastAction", "")
    verb = action.split(" ")[0].lower() if action else ""
    same_text = cur["look"] == nxt["look"] and cur["inventory"] == nxt["inventory"]
    o0, o1 = _objs(cur), _objs(nxt)
    if same_text and o0 == o1:
        return "reversible", "no_change"
    if set(o0) - set(o1) or set(o1) - set(o0):
        return "irreversible", "objects_created_or_removed"
    if verb in NOOP_VERBS:
        return "reversible", "noop_verb"
    if verb in IRREVERSIBLE_VERBS:
        return "irreversible", "verb_rule"
    if verb in INVERSE_VERBS:
        return "reversible", "verb_rule"
    return "unknown", "no_rule"


def cmd_label(a):
    recs = [json.loads(l) for l in open(a.data)]
    out = []
    for r in recs:
        lab, why = rule_label(r)
        out.append({"id": f"bs_{r['game']}_{r['state_id']}", "game": r["game"],
                    "action": r["action_state"]["lastAction"], "label": lab, "label_rule": why,
                    "current": {k: r["current_state"][k] for k in ("observation", "look", "inventory", "taskDesc")},
                    "next": {k: r["action_state"][k] for k in ("observation", "look", "inventory")}})
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("labels:", Counter(o["label"] for o in out))
    print("rules:", Counter(o["label_rule"] for o in out))


PROMPT = ("You are simulating a text-based game. Given the current state and an action, predict the "
          "state right after the action.\n\nTask: {task}\n\nCurrent observation:\n{look}\n{inv}\n\n"
          "Action: {action}\n\nWrite the next observation (the room description) and the inventory, in the "
          "same format as above, and nothing else.")


def messages(rec):
    c = rec["current"]
    return [{"role": "user", "content": PROMPT.format(task=c["taskDesc"], look=c["look"].strip(),
                                                      inv=c["inventory"].strip(), action=rec["action"])}]


def cmd_score(a):
    from irrev.models import lm as LMmod
    recs = [json.loads(l) for l in open(a.data)]
    recs = [r for r in recs if r["label"] in ("reversible", "irreversible")]
    if not a.keep_no_change:
        # no-change transitions (failed or look-only actions) are trivially predictable and
        # would inflate any uncertainty difference; exclude them from the main analysis
        recs = [r for r in recs if r["label_rule"] != "no_change"]
    rng = random.Random(0)
    # balanced sample by label
    by = {l: [r for r in recs if r["label"] == l] for l in ("reversible", "irreversible")}
    k = min(a.n // 2, *(len(v) for v in by.values()))
    sample = rng.sample(by["reversible"], k) + rng.sample(by["irreversible"], k)
    LMmod.set_seed(0)
    lm = LMmod.load(a.model)
    out = Path(f"results/{a.model}/rq5_scores.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, r in enumerate(sample):
            m = messages(r)
            pred, nll_own, n_tok = LMmod.generate(lm, m, max_new_tokens=a.max_new_tokens, return_scores=True)
            gold = r["next"]["look"].strip() + "\n" + r["next"]["inventory"].strip()
            nll_gold = LMmod.continuation_nll(lm, m, gold)
            f.write(json.dumps({"id": r["id"], "game": r["game"], "label": r["label"], "action": r["action"],
                                "pred": pred, "nll_own": nll_own, "nll_gold": nll_gold, "n_tok": n_tok}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(sample)}", flush=True)
    print("saved", out)


def cmd_analyse(a):
    from sklearn.metrics import roc_auc_score
    from irrev.analysis.stats import bootstrap_ci
    rows = [json.loads(l) for l in open(f"results/{a.model}/rq5_scores.jsonl")]
    y = np.array([r["label"] == "irreversible" for r in rows], int)
    games = np.array([r["game"] for r in rows])
    res = {"n": len(rows), "n_irr": int(y.sum())}
    for key in ("nll_own", "nll_gold"):
        s = np.array([r[key] for r in rows], float)
        ok = ~np.isnan(s)
        p, lo, hi = bootstrap_ci(lambda yy, ss: roc_auc_score(yy, ss), y[ok], s[ok], groups=games[ok])
        res[key] = {"auroc_irr_vs_rev": p, "ci": [lo, hi],
                    "mean_irr": float(s[ok & (y == 1)].mean()), "mean_rev": float(s[ok & (y == 0)].mean())}
    json.dump(res, open(f"results/{a.model}/rq5.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("label"); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("score"); p.add_argument("--model", required=True); p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=600); p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--keep-no-change", action="store_true")
    p = sp.add_parser("analyse"); p.add_argument("--model", required=True)
    a = ap.parse_args()
    {"label": cmd_label, "score": cmd_score, "analyse": cmd_analyse}[a.cmd](a)
