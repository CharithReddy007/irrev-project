"""Presentation demo: run the agent on one task pair (reversible and irreversible version)
and print the episode as a readable transcript. Record the terminal for the talk.

    python scripts/demo_agent.py --model qwen7b --pair cleanup_03
    python scripts/demo_agent.py --scripted careful --pair cleanup_03     # no GPU, for rehearsal
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irrev.agent import policies  # noqa: E402
from irrev.agent.caution import caution_measures  # noqa: E402
from irrev.agent.loop import run_episode  # noqa: E402


def show(ep, task):
    m = caution_measures(ep, task)
    print("=" * 78)
    print(f"{task['task_id']}  [{task['version'].upper()} version]")
    print("Task:", task["instruction"])
    print("Settings:", task["settings"], "| backups:", list(task["backups"]))
    print("-" * 78)
    for s in ep["steps"]:
        tag = f"  <{s['label']}>" if s.get("label") else ""
        call = f"{s.get('tool')}({json.dumps(s.get('args', {}))[:70]})" if s.get("tool") else f"UNPARSED: {s['raw'][:60]!r}"
        print(f"step {s['step']:2d}: {call}{tag}")
        print(textwrap.indent(textwrap.shorten(s["obs"], 100), "         -> "))
    print("-" * 78)
    print(f"success={ep['success']}  careful={m['careful']}  took_irreversible={m['took_irreversible']}  "
          f"lost_important_file={ep['protected_lost']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="cleanup_00")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scripted", choices=["careful", "reckless"], default=None)
    ap.add_argument("--tasks", default="data/agent_tasks.jsonl")
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks) if json.loads(l)["pair_id"] == a.pair]
    if a.model:
        from irrev.models import lm as L
        lm = L.load(a.model)
        make = lambda t: (lambda msgs: L.generate(lm, msgs, max_new_tokens=200))
    else:
        make = getattr(policies, a.scripted or "reckless")
    for t in sorted(tasks, key=lambda t: t["version"], reverse=True):   # reversible first
        show(run_episode(t, make(t)), t)


if __name__ == "__main__":
    main()
