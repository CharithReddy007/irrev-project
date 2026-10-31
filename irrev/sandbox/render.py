"""Turn sandbox states and tool calls into text for the model.

Design rules
------------
* Every setting is always shown, for every state, so the mere presence of a word such as
  "disabled" does not predict the label. Only the *combination* of a setting and the action
  determines reversibility; that combination is what we want the model to integrate.
* Each setting has several paraphrases chosen at random (seeded), so a probe cannot key on
  one fixed sentence.
* The words "reversible", "irreversible", "undo", "permanent" never appear in settings
  text, to avoid handing the model (or a lexical baseline) the answer.  Exception: the
  email setting needs to describe a delay; we describe it as a holding period.
"""
from __future__ import annotations

import json
import random
from typing import Any

from .env import Sandbox, TOOL_SPECS

PHRASES = {
    ("recycle_bin", True): [
        "Deleted files go to the trash folder.",
        "Trash: on (deleted files are kept in the trash).",
        "When a file is deleted it is moved to the trash.",
    ],
    ("recycle_bin", False): [
        "Deleted files are erased directly; there is no trash folder.",
        "Trash: off (deleted files are not kept anywhere).",
        "When a file is deleted it is wiped from the disk.",
    ],
    ("edit_history", True): [
        "Version history: on (the previous text of an edited file is kept).",
        "The editor keeps the earlier text of every file you edit.",
        "Each edit saves the old text in the version history.",
    ],
    ("edit_history", False): [
        "Version history: off (old text is not kept after an edit).",
        "The editor does not keep earlier text when a file is edited.",
        "Edits overwrite the file; earlier text is not stored.",
    ],
    ("email_undo", True): [
        "Outgoing email is held in the outbox for {s} seconds before delivery, and can be cancelled during that time.",
        "Mail delivery delay: {s} seconds (messages in the outbox can be recalled).",
        "Emails wait {s} seconds in the outbox before they leave; waiting emails can be withdrawn.",
    ],
    ("email_undo", False): [
        "Outgoing email is delivered immediately.",
        "Mail delivery delay: none (messages leave as soon as they are sent).",
        "Emails leave the outbox the moment they are sent.",
    ],
    ("transfer", "instant"): [
        "Wallet transfers settle immediately and cannot be cancelled.",
        "Payments: instant settlement, no cancellation.",
        "Money sent from the wallet arrives at once; there is no cancel option.",
    ],
    ("transfer", "pending"): [
        "Wallet transfers stay pending for 24 hours and can be cancelled for a full refund.",
        "Payments: 24-hour pending period, cancellation refunds the full amount.",
        "Money sent from the wallet is held for a day and can be called back free of charge.",
    ],
    ("transfer", "pending_fee"): [
        "Wallet transfers stay pending for 24 hours; cancelling refunds the amount minus a {fee:.2f} fee.",
        "Payments: 24-hour pending period, cancellation costs a {fee:.2f} fee.",
        "Money sent from the wallet is held for a day; calling it back costs {fee:.2f}.",
    ],
}


def settings_lines(sb: Sandbox, rng: random.Random) -> list[str]:
    st = sb.settings
    out = [
        rng.choice(PHRASES[("recycle_bin", st.recycle_bin)]),
        rng.choice(PHRASES[("edit_history", st.edit_history)]),
        rng.choice(PHRASES[("email_undo", st.email_undo_seconds > 0)]).format(s=st.email_undo_seconds),
        rng.choice(PHRASES[("transfer", st.transfer_mode)]).format(fee=st.cancel_fee),
    ]
    rng.shuffle(out)  # order varies too
    return out


def state_text(sb: Sandbox, seed: int, max_preview: int = 60) -> str:
    """Natural-language description of the full state. Deterministic given the seed."""
    rng = random.Random(seed)
    s = sb.s
    lines = ["Workspace settings:"]
    lines += [f"- {x}" for x in settings_lines(sb, rng)]
    lines.append("Files:")
    if s["files"]:
        for p, c in sorted(s["files"].items()):
            preview = "(compressed)" if c.startswith("ZIP:") else c[:max_preview].replace("\n", " ")
            extra = " [backup available]" if p in s["backups"] else ""
            lines.append(f"- {p} ({sb.size_of(c)} bytes){extra}: {preview}")
    else:
        lines.append("- (none)")
    if s["trash"]:
        lines.append("Trash: " + ", ".join(sorted(s["trash"])))
    if s["outbox"]:
        lines.append("Outbox (waiting): " + "; ".join(f"{e['id']} to {e['to']} '{e['subject']}'" for e in s["outbox"]))
    if s["sent"]:
        lines.append("Sent: " + "; ".join(f"to {e['to']} '{e['subject']}'" for e in s["sent"]))
    lines.append(f"Wallet balance: {s['balance']:.2f}")
    if s["pending_transfers"]:
        lines.append("Pending transfers: " + "; ".join(f"{t['id']} {t['amount']:.2f} to {t['to']}" for t in s["pending_transfers"]))
    return "\n".join(lines)


def action_text(tool: str, args: dict[str, Any]) -> str:
    """Canonical JSON form of a tool call. The agent must output exactly this format."""
    return json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


def tools_text(names: list[str] | None = None) -> str:
    names = names or list(TOOL_SPECS)
    out = []
    for n in names:
        spec = TOOL_SPECS[n]
        args = ", ".join(f"{k}: {v}" for k, v in spec["args"].items())
        out.append(f"- {n}({args}): {spec['desc']}")
    return "\n".join(out)
