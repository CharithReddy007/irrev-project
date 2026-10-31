"""Turn ToolEmu's 144 curated cases into a hand-labelling sheet (CSV).

We do NOT run the ToolEmu emulator (it calls GPT-4). We only reuse the case descriptions:
for each case, the team writes one or more (state, action) pairs that fit our sandbox and
labels them. The sheet pre-fills a suggested sandbox tool where the case text makes it
obvious; humans confirm or change it.

    git clone https://github.com/ryoungj/ToolEmu   (Apache-2.0)
    python -m irrev.data.toolemu_sheet --cases ToolEmu/assets/all_cases.json --out data/toolemu_sheet.csv

After labelling, convert the filled sheet to JSONL with --import.
"""
from __future__ import annotations

import argparse
import csv
import json
import re

KEYWORDS = [
    (r"\bdelet|remov|wipe|erase|clean ?up\b", "delete_file"),
    (r"\bemail|send .*message|notify\b", "send_email"),
    (r"\btransfer|pay|deposit|withdraw|purchase|buy\b", "transfer_money"),
    (r"\bedit|update|modify|change|overwrite\b", "edit_file"),
]


def suggest(text: str) -> str:
    t = text.lower()
    for pat, tool in KEYWORDS:
        if re.search(pat, t):
            return tool
    return ""


def export(cases_path: str, out: str) -> None:
    cases = json.load(open(cases_path))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_name", "toolkits", "user_instruction", "risky_actions", "suggested_tool",
                    "sandbox_state_text", "sandbox_action_text", "label_annotator1", "label_annotator2", "notes"])
        for c in cases:
            instr = c["User Instruction"]
            risky = " | ".join(c.get("Potential Risky Actions", []))
            w.writerow([c["name"], ",".join(c["Toolkits"]), instr, risky, suggest(instr + " " + risky),
                        "", "", "", "", ""])
    print(f"wrote {len(cases)} rows to {out}")


def import_sheet(sheet: str, out: str) -> None:
    n = 0
    with open(sheet) as f, open(out, "w") as g:
        for row in csv.DictReader(f):
            if not row["sandbox_action_text"] or not row["label_annotator1"]:
                continue
            label = row["label_annotator1"].strip()
            g.write(json.dumps({
                "id": f"toolemu_{row['case_name']}", "environment": "sandbox",
                "state_text": row["sandbox_state_text"], "action_text": row["sandbox_action_text"],
                "label": label, "horizon": 3, "source": f"toolemu:{row['case_name']}",
                "scenario_id": f"toolemu_{row['case_name']}", "group_id": None, "is_minimal_pair": False,
                "verb": json.loads(row["sandbox_action_text"]).get("tool", ""),
                "labels_by_horizon": {"3": label}, "task_text": row["user_instruction"],
                "label_annotator2": row["label_annotator2"].strip(),
            }) + "\n")
            n += 1
    print(f"imported {n} labelled rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--import-sheet")
    a = ap.parse_args()
    if a.import_sheet:
        import_sheet(a.import_sheet, a.out)
    else:
        export(a.cases, a.out)
