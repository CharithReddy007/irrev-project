"""Connects the Sandbox to the generic BFS labeller."""
from __future__ import annotations

from typing import Any

from .env import Sandbox, OBSERVATION_TOOLS, META_TOOLS

Action = tuple[str, tuple[tuple[str, Any], ...]]   # hashable (tool, sorted args)


def mk(tool: str, **args) -> Action:
    return (tool, tuple(sorted(args.items())))


class SandboxAdapter:
    """prune=True restricts file-level inverse candidates to paths named in the taken action
    (plus everything in trash/backups/history/outbox/pending, which are the only places an
    action can leave recoverable traces). tests/test_labeling.py checks that pruned and full
    searches give identical labels on small random states."""

    def __init__(self, sb: Sandbox, prune: bool = True):
        self.sb = sb
        self.prune = prune

    def snapshot(self):
        return self.sb.save()

    def restore(self, snap):
        self.sb.restore(snap)

    def exact_key(self):
        return self.sb.exact_key()

    def approx_key(self):
        return self.sb.approx_key()

    def apply(self, action: Action) -> bool:
        tool, args = action
        before = len(self.sb.log)
        self.sb.call(tool, dict(args))
        ok = not self.sb.log[before]["error"]
        self.sb.log = self.sb.log[:before]           # labelling must not pollute logs
        return ok

    def candidates(self, taken: Action) -> list[Action]:
        """Possible inverse steps, built only from the current state and the taken action."""
        s = self.sb.s
        _, targs = taken
        targs = dict(targs)
        known_paths = {v for k, v in targs.items() if k in ("path", "src", "dst")}
        out: list[Action] = []
        for p in sorted(s["trash"]):
            out.append(mk("restore_from_trash", path=p))
        for p in sorted(s["backups"]):
            out.append(mk("restore_backup", path=p))
            out.append(mk("delete_backup", path=p))
        for p, h in sorted(s["history"].items()):
            if h and p in s["files"]:
                out.append(mk("restore_previous_version", path=p))
        for e in s["outbox"]:
            out.append(mk("undo_send", email_id=e["id"]))
        for t in s["pending_transfers"]:
            out.append(mk("cancel_transfer", transfer_id=t["id"]))
        file_pool = sorted(s["files"])
        if self.prune:
            touched = set(known_paths) | {q + ".zip" for q in known_paths} | {q[:-4] for q in known_paths if q.endswith(".zip")}
            file_pool = [p for p in file_pool if p in touched]
        for p in file_pool:
            out.append(mk("delete_file", path=p))
            if p.endswith(".zip"):
                out.append(mk("decompress_file", path=p))
            else:
                out.append(mk("compress_file", path=p))
            out.append(mk("backup_file", path=p))
            for q in sorted(known_paths):          # move back to a path named in the action
                if q != p and q not in s["files"]:
                    out.append(mk("move_file", src=p, dst=q))
        if s["trash"]:
            out.append(mk("empty_trash"))
        # dedupe, keep order
        seen, uniq = set(), []
        for a in out:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        return uniq


def is_noop_tool(tool: str) -> bool:
    return tool in OBSERVATION_TOOLS or tool in META_TOOLS
