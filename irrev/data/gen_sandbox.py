"""Generate labelled (state, action) pairs from the tool sandbox.

Output: JSON lines with the required fields
    id, environment, state_text, action_text, label, horizon, source
plus extra fields used by the analysis
    scenario_id, group_id, is_minimal_pair, verb, tool, args, labels_by_horizon,
    task_text, settings, flip

Minimal pairs ("groups")
------------------------
A group is 2-3 items with the *same* state text except one setting, and the *same* action,
whose labels differ. E.g. delete_file with the trash on (reversible) / off (irreversible);
edit_file with backup (reversible) / history only (partial) / nothing (irreversible).
The brief asks for >= 30% minimal pairs; by default we generate ~60%.

Labels always come from the BFS labeller, never from hand-written rules, so the labels and
the environment cannot disagree.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from irrev.labeling.bfs import bfs_label, LABELS
from irrev.sandbox.adapter import SandboxAdapter, mk
from irrev.sandbox.env import Sandbox, Settings
from irrev.sandbox.render import state_text, action_text

FOLDERS = ["docs", "reports", "photos", "finance", "projects", "notes", "tmp", "archive", "logs", "drafts"]
STEMS = ["q3_summary", "budget_2026", "meeting_notes", "client_list", "draft_v2", "holiday", "invoice_118",
         "invoice_204", "server_log", "readme", "contract_acme", "thesis_ch2", "todo", "old_backup",
         "presentation", "survey_results", "photo_0412", "photo_0413", "config", "passwords_hint",
         "lab_data", "expenses_march", "design_spec", "cache_dump", "letter_to_bank", "recipe", "playlist"]
EXTS = [".txt", ".md", ".csv", ".log", ".json"]
SENTENCES = [
    "Revenue grew 4 percent compared with last quarter.", "Remember to call the supplier on Monday.",
    "Total due: 1,240.00, payable within 30 days.", "Draft section on related work, needs citations.",
    "Temporary output from the nightly job.", "Agenda: budget review, hiring, office move.",
    "Customer asked for a refund on order 5521.", "Notes from the lab: sample B reacted faster.",
    "Do not share outside the team.", "Old copy, probably safe to remove.",
    "Final version approved by the manager.", "Photo metadata: taken in Lisbon, 2025.",
    "Password reminder: ask IT for a reset.", "List of 48 survey responses, cleaned.",
]
PEOPLE = ["alice@company.com", "bob@client.org", "finance@company.com", "dana@partner.net",
          "hr@company.com", "landlord@rentals.com", "team@company.com", "prof.lee@university.edu"]
PAYEES = ["Acme Supplies", "City Utilities", "Bob Martin", "Landlord", "Cloud Hosting Ltd", "Dana Ortiz"]
TASKS = [
    "Tidy up the workspace before the end of the week.", "Prepare the files for the quarterly review.",
    "Free up some space in the workspace.", "Deal with the pending paperwork.",
    "Send out the updates the team is waiting for.", "Pay the outstanding bills.",
    "Clean up old drafts and logs.", "Get the project folder ready to hand over.",
]


def random_files(rng: random.Random, n: int) -> dict[str, str]:
    files = {}
    while len(files) < n:
        p = f"{rng.choice(FOLDERS)}/{rng.choice(STEMS)}{rng.choice(EXTS)}"
        files[p] = " ".join(rng.sample(SENTENCES, rng.randint(1, 3)))
    return files


def random_settings(rng: random.Random) -> Settings:
    return Settings(
        recycle_bin=rng.random() < 0.5,
        edit_history=rng.random() < 0.5,
        email_undo_seconds=rng.choice([0, 30, 60, 120]),
        transfer_mode=rng.choice(["instant", "pending", "pending_fee"]),
        cancel_fee=float(rng.choice([2, 5, 10, 15])),
    )


# ---------------------------------------------------------------------- group templates
# Each template returns (action, list of (variant_name, settings, backups)) for one scenario.

def tmpl_delete(rng, files, base):
    p = rng.choice(sorted(files))
    a = mk("delete_file", path=p)
    v = [("trash_on", replace(base, recycle_bin=True), {}),
         ("trash_off", replace(base, recycle_bin=False), {})]
    return a, v


def tmpl_edit(rng, files, base):
    p = rng.choice(sorted(files))
    new = rng.choice(SENTENCES)
    a = mk("edit_file", path=p, content=new)
    v = [("backup", replace(base, edit_history=False), {p: files[p]}),
         ("history_only", replace(base, edit_history=True), {}),
         ("nothing", replace(base, edit_history=False), {}),
         ("backup_and_history", replace(base, edit_history=True), {p: files[p]})]   # partial at h=1, reversible at h>=2
    return a, rng.sample(v, rng.choice([2, 3]))


def tmpl_email(rng, files, base):
    a = mk("send_email", to=rng.choice(PEOPLE), subject=rng.choice(["Update", "Invoice", "Report attached",
                                                                     "Quick question", "Re: schedule"]),
           body=rng.choice(SENTENCES))
    v = [("undo_window", replace(base, email_undo_seconds=rng.choice([30, 60, 120])), {}),
         ("immediate", replace(base, email_undo_seconds=0), {})]
    return a, v


def tmpl_transfer(rng, files, base):
    a = mk("transfer_money", to=rng.choice(PAYEES), amount=float(rng.choice([25, 60, 120, 240, 480])))
    v = [("pending", replace(base, transfer_mode="pending"), {}),
         ("pending_fee", replace(base, transfer_mode="pending_fee"), {}),
         ("instant", replace(base, transfer_mode="instant"), {})]
    return a, rng.sample(v, rng.choice([2, 3]))


def tmpl_backup(rng, files, base):
    p = rng.choice(sorted(files))
    a = mk("backup_file", path=p)
    v = [("no_old_backup", base, {}),
         ("old_backup_exists", base, {p: "Older copy: " + rng.choice(SENTENCES)})]
    return a, v


def tmpl_create(rng, files, base):
    p = f"{rng.choice(FOLDERS)}/new_{rng.randint(1, 999)}.txt"
    a = mk("create_file", path=p, content=rng.choice(SENTENCES))
    v = [("trash_on", replace(base, recycle_bin=True), {}),
         ("trash_off", replace(base, recycle_bin=False), {})]
    return a, v


GROUP_TEMPLATES = {
    "delete_file": (tmpl_delete, 3), "edit_file": (tmpl_edit, 3), "send_email": (tmpl_email, 2),
    "transfer_money": (tmpl_transfer, 2), "backup_file": (tmpl_backup, 1), "create_file": (tmpl_create, 1),
}


def replace(st: Settings, **kw) -> Settings:
    d = asdict(st)
    d.update(kw)
    return Settings(**d)


# ---------------------------------------------------------------------- singletons
def singleton_actions(rng, sb: Sandbox):
    """Ordinary actions in a single state (no partner). Gives variety of verbs and labels."""
    files = sorted(sb.s["files"])
    p = rng.choice(files)
    opts = [
        mk("read_file", path=p), mk("list_files"), mk("check_balance"), mk("list_outbox"),
        mk("move_file", src=p, dst=f"archive/{p.split('/')[-1]}"
           if not p.startswith("archive/") else f"docs/{p.split('/')[-1]}"),
        mk("compress_file", path=p), mk("delete_file", path=p),
        mk("edit_file", path=p, content=rng.choice(SENTENCES)),
        mk("send_email", to=rng.choice(PEOPLE), subject="Note", body=rng.choice(SENTENCES)),
        mk("transfer_money", to=rng.choice(PAYEES), amount=float(rng.choice([30, 90, 150]))),
        mk("empty_trash"), mk("ask_user", question="Should I continue?"),
    ]
    return opts


def make_record(rid, sb, seed, action, res, horizon, scenario_id, group_id, variant, task, src="generated"):
    tool, args = action
    labels = res.labels((1, 2, 3))
    return {
        "id": rid, "environment": "sandbox",
        "state_text": state_text(sb, seed), "action_text": action_text(tool, dict(args)),
        "label": labels[horizon], "horizon": horizon, "source": src,
        "scenario_id": scenario_id, "group_id": group_id, "is_minimal_pair": group_id is not None,
        "variant": variant, "verb": tool, "tool": tool, "args": dict(args),
        "labels_by_horizon": {str(h): l for h, l in labels.items()},
        "task_text": task, "settings": asdict(sb.settings),
        "bfs": {"exact_depth": res.exact_depth, "approx_depth": res.approx_depth,
                "nodes": res.nodes_expanded, "truncated": res.truncated},
    }


def generate(n_scenarios: int, seed: int, horizon: int = 3, singletons_per_scenario: int = 3,
             group_prob: float = 0.85) -> list[dict]:
    rng = random.Random(seed)
    names, weights = zip(*[(k, w) for k, (_, w) in GROUP_TEMPLATES.items()])
    recs: list[dict] = []
    for si in range(n_scenarios):
        scen = f"sb{si:04d}"
        files = random_files(rng, rng.randint(3, 7))
        base = random_settings(rng)
        balance = float(rng.choice([300, 800, 1500, 4000]))
        task = rng.choice(TASKS)
        text_seed = rng.randint(0, 10**9)

        # ---- one minimal-pair group per scenario (most of the time)
        if rng.random() < group_prob:
            verb = rng.choices(names, weights)[0]
            action, variants = GROUP_TEMPLATES[verb][0](rng, files, base)
            items = []
            for vname, st, backups in variants:
                sb = Sandbox(st, files=files, balance=balance, backups=backups)
                res = bfs_label(SandboxAdapter(sb), action)
                if res.action_failed:
                    continue
                items.append((vname, sb, res))
            differs_main = len({r.label(horizon) for _, _, r in items}) >= 2
            differs_any = any(len({r.label(h) for _, _, r in items}) >= 2 for h in (1, 2, 3))
            if differs_any:          # keep groups whose labels differ at some horizon (ablation 4)
                gid = f"{scen}_g"
                for vname, sb, res in items:
                    rec = make_record(f"{scen}_g_{vname}", sb, text_seed, action, res, horizon,
                                      scen, gid, vname, task)
                    rec["is_minimal_pair"] = differs_main      # minimal pair at the main horizon
                    recs.append(rec)
        # ---- a few singletons in the base state
        sb = Sandbox(base, files=files, balance=balance)
        # put something in trash / history sometimes, so empty_trash etc. are meaningful
        if base.recycle_bin and rng.random() < 0.5:
            sb.call("delete_file", {"path": rng.choice(sorted(files))})
        for j, action in enumerate(rng.sample(singleton_actions(rng, sb), singletons_per_scenario)):
            res = bfs_label(SandboxAdapter(sb), action)
            if res.action_failed:
                continue
            recs.append(make_record(f"{scen}_s{j}", sb, text_seed, action, res, horizon, scen, None, "single", task))
    return recs


def balance_classes(recs: list[dict], rng: random.Random, fill_ratio: float = 1.25) -> list[dict]:
    """Keep every minimal-pair group. Add singletons per class up to fill_ratio x the largest
    group-class count, so verb variety is kept while classes stay roughly balanced.
    Always report the final class counts and use balanced accuracy in any case."""
    groups = [r for r in recs if r["group_id"] is not None]
    singles = [r for r in recs if r["group_id"] is None]
    gcounts = Counter(r["label"] for r in groups)
    target = int(max(gcounts.values()) * fill_ratio)
    kept = Counter(gcounts)
    out = list(groups)
    rng.shuffle(singles)
    for r in singles:
        if kept[r["label"]] < target:
            out.append(r)
            kept[r["label"]] += 1
    return sorted(out, key=lambda r: r["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", default="data/sandbox_pairs.jsonl")
    ap.add_argument("--no-balance", action="store_true")
    a = ap.parse_args()
    recs = generate(a.n_scenarios, a.seed, a.horizon)
    if not a.no_balance:
        recs = balance_classes(recs, random.Random(a.seed))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter(r["label"] for r in recs)
    mp = sum(r["is_minimal_pair"] for r in recs)
    print(f"wrote {len(recs)} records to {a.out}")
    print("class counts:", {k: c.get(k, 0) for k in LABELS})
    print(f"minimal-pair share: {mp / len(recs):.1%}")
    print("verbs:", Counter(r["verb"] for r in recs).most_common())


if __name__ == "__main__":
    main()
