"""The agent loop: the model outputs one JSON tool call per step, up to 15 steps.

A policy is any function messages -> text. The HF model, the steered model and the scripted
test policies all fit this interface, so the loop and the logging are identical for all.

Parse handling (Section 9 of the brief: count parse failures separately, never drop them):
  strict     the whole output is one JSON object                 -> ok
  lenient    a JSON object can be extracted from the text        -> ok, counted as format_violation
  failed     nothing parseable                                   -> parse_failure, the model is told
                                                                    and the step still counts
"""
from __future__ import annotations

import json
import re
from typing import Callable

from irrev.agent.tasks import build_sandbox, check_success, protected_lost
from irrev.labeling.bfs import bfs_label
from irrev.models.prompts import agent_system, agent_user
from irrev.sandbox.adapter import SandboxAdapter, mk, is_noop_tool
from irrev.sandbox.render import state_text

Policy = Callable[[list[dict]], str]
PARSE_ERROR_MSG = 'Error: could not read a tool call. Reply with exactly one JSON object like {"tool": "list_files", "args": {}}.'


def parse_tool_call(text: str):
    """Return (call_dict or None, status) where status in {'strict', 'lenient', 'failed'}."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("args", {})
            return obj, "strict"
    except json.JSONDecodeError:
        pass
    # lenient: first balanced {...} that parses and has a "tool" key
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        for j in range(start, len(text)):
            depth += text[j] == "{"
            depth -= text[j] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[start:j + 1])
                    if isinstance(obj, dict) and "tool" in obj:
                        obj.setdefault("args", {})
                        return obj, "lenient"
                except json.JSONDecodeError:
                    pass
                break
    return None, "failed"


class Episode:
    """One agent episode as a small state machine, so many episodes can share one batched
    model call per step (run_batched). run_episode() drives a single one with the same code."""

    def __init__(self, task: dict, careful_prompt: bool = False, max_steps: int = 15, label_calls: bool = True):
        self.task, self.careful_prompt, self.max_steps, self.label_calls = task, careful_prompt, max_steps, label_calls
        self.sb = build_sandbox(task)
        self.msgs = [{"role": "system", "content": agent_system(careful_prompt)},
                     {"role": "user", "content": agent_user(task["instruction"], state_text(self.sb, task["text_seed"]))}]
        self.steps: list[dict] = []
        self.n_parse_fail = self.n_format = 0
        self.done = False

    def feed(self, out: str) -> None:
        """Apply one model output (one step)."""
        sb, step = self.sb, len(self.steps)
        self.msgs.append({"role": "assistant", "content": out})
        call, status = parse_tool_call(out)
        rec = {"step": step, "raw": out, "parse": status}
        if call is None:
            self.n_parse_fail += 1
            obs = PARSE_ERROR_MSG
        else:
            self.n_format += status == "lenient"
            tool, args = str(call.get("tool")), call.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            rec.update(tool=tool, args=args)
            # label the call in the state where it is taken (exact, from the BFS labeller)
            if self.label_calls and not is_noop_tool(tool):
                try:
                    res = bfs_label(SandboxAdapter(sb.clone()), mk(tool, **args))
                    rec["label"] = None if res.action_failed else res.label(3)
                except TypeError:                # unhashable / odd args
                    rec["label"] = None
            obs = sb.call(tool, args)
            rec["error"] = sb.log[-1]["error"]
        rec["obs"] = obs
        self.steps.append(rec)
        if sb.finished or len(self.steps) >= self.max_steps:
            self.done = True
        else:
            self.msgs.append({"role": "user", "content": f"Observation: {obs}"})

    def result(self) -> dict:
        sb, task = self.sb, self.task
        if not sb.finished:
            sb.flush_outbox()       # the episode ends; emails still waiting are delivered
        return {
            "task_id": task["task_id"], "pair_id": task["pair_id"], "family": task["family"], "version": task["version"],
            "careful_prompt": self.careful_prompt, "steps": self.steps, "messages": self.msgs,
            "finished": sb.finished, "success": check_success(task, sb), "protected_lost": protected_lost(task, sb),
            "n_parse_fail": self.n_parse_fail, "n_format_violation": self.n_format, "n_steps": len(self.steps),
        }


def run_episode(task: dict, policy: Policy, careful_prompt: bool = False, max_steps: int = 15,
                label_calls: bool = True) -> dict:
    ep = Episode(task, careful_prompt, max_steps, label_calls)
    while not ep.done:
        ep.feed(policy(ep.msgs))
    return ep.result()


def run_batched(tasks: list[dict], batch_policy: Callable[[list[list[dict]]], list[str]], batch_size: int = 16,
                careful_prompt: bool = False, max_steps: int = 15, on_done: Callable[[dict], None] | None = None):
    """Run many episodes with one batched model call per step.

    Episodes are processed in chunks of batch_size; inside a chunk, finished episodes drop out
    and the rest continue. on_done(result) is called as each episode finishes, so results can be
    written to disk immediately (important on Colab, where sessions disconnect).
    """
    results = []
    for i in range(0, len(tasks), batch_size):
        eps = [Episode(t, careful_prompt, max_steps) for t in tasks[i:i + batch_size]]
        while True:
            active = [e for e in eps if not e.done]
            if not active:
                break
            outs = batch_policy([e.msgs for e in active])
            for e, o in zip(active, outs):
                e.feed(o)
                if e.done:
                    r = e.result()
                    results.append(r)
                    if on_done:
                        on_done(r)
    return results
