"""Step 4 (RQ2): ask the model directly whether each action can be undone.

    python -m irrev.stated.ask --model qwen7b --data data/dataset.jsonl
Writes results/<model>/stated.jsonl with the prompt, raw output and parsed label for every pair.
Unparseable answers are kept and labelled "unparsed" (never silently dropped).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = [
    ("partially_reversible", r"\bPARTIALLY[_\s-]?REVERSIBLE\b"),
    ("irreversible", r"\bIRREVERSIBLE\b|\bNOT\s+REVERSIBLE\b"),
    ("reversible", r"\bREVERSIBLE\b"),
]


def parse_stated(text: str) -> str:
    """Map a free-text answer to one of the three labels. Order matters: check the longer
    labels first, and use the first label mentioned if several appear."""
    t = text.upper()
    hits = []
    for lab, pat in PATTERNS:
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), lab))
    if not hits:
        return "unparsed"
    # PARTIALLY_REVERSIBLE contains REVERSIBLE; IRREVERSIBLE contains REVERSIBLE. Prefer the
    # longest match starting at the earliest position.
    hits.sort()
    first_pos = hits[0][0]
    same = [lab for pos, lab in hits if pos <= first_pos + 2]
    for pref in ("partially_reversible", "irreversible", "reversible"):
        if pref in same:
            return pref
    return hits[0][1]


def parse_choice(text: str) -> str:
    m = re.search(r"\b([AB])\b", text.strip().upper())
    return m.group(1) if m else "unparsed"


def main():
    from irrev.models import lm as LMmod
    from irrev.models.prompts import stated_messages
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.data)][: a.limit]
    out = Path(f"results/{a.model}/stated.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out)}
    LMmod.set_seed(0)
    lm = LMmod.load(a.model)
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    with open(out, "a") as f:
        for i in range(0, len(todo), a.batch_size):
            chunk = todo[i:i + a.batch_size]
            msgs = [stated_messages(r, horizon=r.get("horizon", 3)) for r in chunk]
            raws = LMmod.generate_batch(lm, msgs, max_new_tokens=12)
            for r, m, raw in zip(chunk, msgs, raws):
                f.write(json.dumps({"id": r["id"], "messages": m, "raw": raw, "stated": parse_stated(raw),
                                    "true": r["label"], "batch_size": a.batch_size}) + "\n")
            f.flush()
            print(f"{len(done) + i + len(chunk)}/{len(recs)}", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
