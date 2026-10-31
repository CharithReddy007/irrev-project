"""Label (state, action) pairs in ScienceWorld with the same BFS labeller as the sandbox.

ScienceWorld's Python API (checked against the allenai/ScienceWorld source) has no
save/restore, so a "snapshot" is the action history, and restoring means:
    env.load(task, variation, simplifications); env.reset(); replay the history.
The simulator is deterministic after reset (their test_observation_is_deterministic and
test_object_tree_is_uuid_ordered tests check this), so replay reproduces the state exactly.

State fingerprints come from env.getObjectTree(), the full JSON object tree:
  exact_key  - the whole tree; numbers (temperatures) compared within EXACT_TOL = 2 C,
               because the heat model has ~0.02 C run-to-run noise and objects drift ~0.7 C
               when moved for a couple of steps (both measured on Colab)
  approx_key - the same with numbers within APPROX_TOL = 10 C and life/growth properties
               dropped, i.e. "the same objects in the same places with the same states,
               up to slow physical drift"

Time passes on every step in ScienceWorld (things cool, plants grow), so even doing
nothing can change the exact state. We therefore also record `volatile`: whether the
state changes when we only take "look around" steps. Report results with and without
volatile states (it is a limitation worth one sentence in the paper).

Cost: every BFS node costs one reset + replay. Keep histories short (sample states early
in trajectories), restrict candidates to actions that mention an object from the labelled
action, and run horizon 3 on a sample (the brief allows this). Use --workers to run several
JVMs in parallel; each ScienceWorldEnv starts its own JVM (about 0.5-1 GB RAM each).

Usage (on a machine with Java >= 1.8 and `pip install scienceworld`):
    python -m irrev.data.scienceworld_label --out data/sciworld_pairs.jsonl \
        --tasks boil melt freeze change-the-state-of-matter-of mix chemistry-mix \
        --variations 5 --states-per-episode 6 --actions-per-state 4 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from irrev.labeling.bfs import bfs_label

# Colab / Docker fix: inside containers the JVM prints a "Cgroup memory controller" warning on
# stdout before anything else, and ScienceWorld's py4j launcher reads that first line expecting
# a port number -> "ValueError: invalid literal for int()". Disabling the JVM's container check
# and its log output avoids it. Must be set before the first ScienceWorldEnv is created; worker
# processes inherit it.
JVM_FLAGS = "-XX:-UseContainerSupport -Xlog:disable"
if JVM_FLAGS not in os.environ.get("JAVA_TOOL_OPTIONS", ""):
    os.environ["JAVA_TOOL_OPTIONS"] = (os.environ.get("JAVA_TOOL_OPTIONS", "") + " " + JVM_FLAGS).strip()

CODE_VERSION = "2026-10-21a"   # bump when labelling logic changes; stale progress files are then ignored
SIMPLIFICATIONS = "teleportAction,openDoors,noElectricalAction"   # shorter histories, cleaner labels
EXCLUDE_PREFIXES = ("focus on", "wait", "look", "inventory", "examine", "read", "task", "reset", "answer")
# 'focus on' ends or penalises the task, 'wait' only passes time, the rest do not change state.


# State fingerprints with a numeric tolerance.
# ScienceWorld's heat simulation is not bit-for-bit deterministic: replaying the same actions
# gives temperatures that differ in the 3rd decimal (e.g. 10.0317 vs 10.0344 C), checked on
# Colab with the real simulator. Hashing raw floats would make nothing ever "restored exactly".
# So a key keeps the tree STRUCTURE exactly (objects, ids, containers, strings, booleans) and
# compares every NUMBER with a tolerance.
# Defaults chosen from the Colab measurements: replay noise ~0.02 C; moving an object somewhere
# else for 2 steps and back changes its temperature by ~0.7 C. The model never sees
# temperatures (only room descriptions), so sub-perceptual drift should not decide the label.
# Heating/cooling an object changes it by much more than 2 C. Both are CLI options and are saved
# in every record, so a sensitivity check with other values is easy.
EXACT_TOL = 2.0      # "restored exactly", up to thermal drift
APPROX_TOL = 10.0    # "restored only approximately" (e.g. an object left noticeably warmer)
DROP_FOR_APPROX = ("proplife",)   # plant/animal growth counters are ignored for the approx key
_TOL = {"exact": EXACT_TOL, "approx": APPROX_TOL}   # set per job from the CLI
NOOP_COUNT = [0]     # actions skipped because they changed nothing (reported per episode)


def _flatten(obj: Any, drop: tuple[str, ...], nums: list[float]) -> Any:
    """Return the structure with numbers replaced by '#', appending the numbers to nums
    in a deterministic order (dict keys sorted)."""
    if isinstance(obj, dict):
        return {k: _flatten(obj[k], drop, nums) for k in sorted(obj) if k.lower() not in drop}
    if isinstance(obj, list):
        return [_flatten(x, drop, nums) for x in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
        return "#"
    return obj


class TreeKey:
    """Hashable state key: equal if the structure is identical and all numbers are within tol.
    The hash only uses the structure, so equal keys always share a hash."""

    __slots__ = ("struct", "nums", "tol", "_h")

    def __init__(self, tree: Any, tol: float, drop: tuple[str, ...] = ()):
        nums: list[float] = []
        self.struct = json.dumps(_flatten(tree, drop, nums), sort_keys=True)
        self.nums, self.tol = tuple(nums), tol
        self._h = hash(self.struct)

    def __hash__(self):
        return self._h

    def __eq__(self, other):
        return (isinstance(other, TreeKey) and self.struct == other.struct and len(self.nums) == len(other.nums)
                and all(abs(a - b) <= self.tol for a, b in zip(self.nums, other.nums)))

    def max_diff(self, other) -> float:
        if self.struct != other.struct:
            return float("inf")
        return max((abs(a - b) for a, b in zip(self.nums, other.nums)), default=0.0)


def _h(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _mentions(action: str, referents: list[str]) -> bool:
    return any(r and r in action for r in referents)


# ---------------------------------------------------------------------------- object metadata
# ScienceWorld reports, for every valid action, the ids of the objects involved (direct object
# first) via get_valid_action_object_combinations_with_templates(), and the object tree says
# whether each object is movable (propMoveable.isMovable) or a container
# (propContainer.isContainer, a "true"/"false" string). We use these instead of matching names.
MULTI_WORD_VERBS = ("pick up", "put down", "look at", "look in", "go to", "teleport to", "focus on",
                    "turn on", "turn off")
UNDO_VERBS = ("pick up", "put down", "open", "close", "activate", "deactivate", "move", "go to",
              "teleport to", "connect", "disconnect", "turn on", "turn off")
OBJECT_MOVING_VERBS = ("move", "pick up", "pour", "dunk", "put down")
# Large fixtures: ScienceWorld lets you "pour counter into inventory", but no agent would.
FIXTURE_WORDS = {"counter", "cupboard", "table", "chair", "sofa", "couch", "bed", "bathtub", "toilet",
                 "sink", "stove", "oven", "fridge", "freezer", "door", "window", "floor", "wall", "drain",
                 "closet", "shelf", "desk", "bookcase", "painting", "workbench", "dishwasher", "washing"}
# Pouring or dunking into these makes no sense (they cannot hold liquid)
NON_LIQUID_TARGETS = {"chair", "sofa", "couch", "bed", "table", "counter", "desk", "shelf", "bookcase",
                      "painting", "workbench", "door", "window", "floor", "wall"}
# Which verbs can undo which (used to try the most likely undo first)
INVERSE = {"put down": ("pick up",), "pick up": ("move", "put down"), "open": ("close",), "close": ("open",),
           "activate": ("deactivate",), "deactivate": ("activate",), "turn on": ("turn off",),
           "turn off": ("turn on",), "connect": ("disconnect",), "disconnect": ("connect",),
           "move": ("move", "pick up"), "go to": ("go to", "teleport to"), "teleport to": ("teleport to", "go to")}
NEEDS_CONTAINER_TARGET = ("move", "pour", "dunk")


def verb_of(action: str) -> str:
    for v in MULTI_WORD_VERBS:
        if action.startswith(v + " ") or action == v:
            return v
    return action.split(" ")[0]


def object_index(tree: dict) -> dict[int, dict]:
    """uuid -> {name, type, movable, container, room} for every object in the tree."""
    idx: dict[int, dict] = {}

    def walk(o, depth, parent=None):
        try:
            uid = int(o.get("uuid"))
        except (TypeError, ValueError):
            uid = None
        mov = (o.get("propMoveable") or {}).get("isMovable")
        cont = (o.get("propContainer") or {}).get("isContainer")
        if uid is not None:
            idx[uid] = {"name": o.get("name"), "type": o.get("type"),
                        "movable": bool(mov) if mov is not None else False,
                        "container": str(cont).lower() == "true",
                        "room": depth == 1,            # children of the universe are locations
                        "parent": parent}
        for child in (o.get("contents") or {}).values():
            if isinstance(child, dict):
                walk(child, depth + 1, uid)

    walk(tree, 0)
    return idx


def is_sensible(entry: dict, idx: dict[int, dict]) -> bool:
    """Filter out valid-but-meaningless actions such as 'pour cupboard into bathroom' or
    'move counter to metal pot'. Unknown ids are let through (no information to judge)."""
    act, ids = entry["action"], entry.get("obj_ids") or []
    v = verb_of(act)
    if act.startswith(EXCLUDE_PREFIXES):
        return False
    objs = [idx.get(i) for i in ids]
    if v in ("go to", "teleport to"):
        return True
    if any(o and (o["room"] or o["name"] in ("agent", "air")) for o in objs):
        return False
    if v in OBJECT_MOVING_VERBS and objs and objs[0] and (
            not objs[0]["movable"] or set(str(objs[0]["name"]).split()) & FIXTURE_WORDS):
        return False
    if v in NEEDS_CONTAINER_TARGET and len(objs) > 1 and objs[1] and not objs[1]["container"]:
        return False
    if v in ("pour", "dunk") and len(objs) > 1 and objs[1] and set(str(objs[1]["name"]).split()) & NON_LIQUID_TARGETS:
        return False
    if v == "pour" and objs and objs[0] and not objs[0]["container"]:
        return False                                   # you can only pour FROM a container
    return True


class SciWorldAdapter:
    """Snapshot = tuple of actions since reset."""

    def __init__(self, env, task: str, variation: int, simplifications: str = SIMPLIFICATIONS,
                 max_branch: int = 20):
        self.env, self.task, self.var, self.simp = env, task, variation, simplifications
        self.history: tuple[str, ...] = ()
        self.max_branch = max_branch
        self._tree_cache: dict[tuple[str, ...], Any] = {}
        self.replays = 0

    # ---------------------------------------------------------------- replay machinery
    # Speed: ScienceWorldEnv.step() also computes look, inventory and ALL valid actions after
    # every step (~50 ms on Colab). Replays only need the step itself, so we call the Java
    # server directly when it is available (env.server.step / env.server.reset), and fall back
    # to the normal API otherwise (e.g. the fake world in the tests).
    def _raw_step(self, action: str) -> str:
        srv = getattr(self.env, "server", None)
        return srv.step(action) if srv is not None else self.env.step(action)[0]

    def _raw_reset(self) -> None:
        srv = getattr(self.env, "server", None)
        if srv is not None:
            srv.reset()
            srv.step("look around")      # exactly what ScienceWorldEnv.reset() does
        else:
            self.env.reset()

    def _reload(self, history: tuple[str, ...]) -> None:
        if not getattr(self, "_loaded", False):
            self.env.load(self.task, self.var, self.simp)
            self._loaded = True
        self._raw_reset()                # reset is deterministic (ScienceWorld's own tests check this)
        for a in history:
            self._raw_step(a)
        self.history = history
        self.replays += 1

    def snapshot(self):
        return self.history

    def restore(self, snap):
        if snap != self.history:
            self._reload(snap)

    def _tree(self):
        if self.history not in self._tree_cache:
            self._tree_cache[self.history] = self.env.getObjectTree()
        return self._tree_cache[self.history]

    def exact_key(self):
        return TreeKey(self._tree(), _TOL["exact"])

    def approx_key(self):
        return TreeKey(self._tree(), _TOL["approx"], DROP_FOR_APPROX)

    def apply(self, action: str) -> bool:
        obs = self._raw_step(action)
        self.history = self.history + (action,)
        if obs.strip().startswith("Ambiguous"):
            # ScienceWorld asks which object was meant; pick the first option. The choice
            # is part of the history, so replay stays exact.
            obs = self._raw_step("0")
            self.history = self.history + ("0",)
        failed = ("No known action" in obs or "Unknown action" in obs or "not sure" in obs.lower())
        return not failed

    def prepare(self, action: str) -> None:
        """Call in the state where `action` is about to be labelled: remembers which object ids
        the action involves (direct object first), so undo candidates can be found by id."""
        self.target_ids, self.s0_parent, self.s0_room = [], None, None
        idx = object_index(self._tree())
        agent = next((o for o in idx.values() if o["name"] == "agent"), None)
        if agent and agent.get("parent") in idx:
            self.s0_room = idx[agent["parent"]]["name"]      # where the agent stands before acting
        templ = getattr(self.env, "get_valid_action_object_combinations_with_templates", None)
        if templ is not None:
            for e in templ():
                if e["action"] == action:
                    self.target_ids = list(e.get("obj_ids") or [])
                    break
            if self.target_ids:
                self.s0_parent = idx.get(self.target_ids[0], {}).get("parent")

    def _goes_back(self, action: str) -> bool:
        """True for 'go to <room>' / 'teleport to <room>' that leads back to the starting room."""
        room = getattr(self, "s0_room", None)
        return bool(room) and action.rstrip().endswith(room)

    def reference_keys(self, n_max: int):
        """Keys of the states reached from the current state by 1..n_max idle steps
        ('look around'), so undo sequences can be compared with where the world would be
        anyway (time passes on every ScienceWorld step). Index n = after n steps."""
        snap = self.history
        self.restore(snap)
        refs = [None]
        for _ in range(n_max):
            self._raw_step("look around")
            self.history = self.history + ("look around",)
            refs.append((self.exact_key(), self.approx_key()))
        self.restore(snap)
        return refs

    def candidates(self, taken: str) -> list[str]:
        """Undo candidates, most promising first, capped at max_branch:
          0. undo-type verbs (pick up, put down, open, close, move, (de)activate...) whose DIRECT
             object is the taken action's direct object
          1. any other action whose direct object is one of the taken action's objects
          2. actions that involve one of those objects in another role
        For 'go to'/'teleport to' all movement actions are rank 0.
        Falls back to name matching when object ids are unavailable."""
        is_move = taken.startswith(("go to", "teleport to"))
        templ = getattr(self.env, "get_valid_action_object_combinations_with_templates", None)
        targets = getattr(self, "target_ids", [])
        ranked: list[tuple[int, str]] = []
        if templ is not None and (targets or is_move):
            primary = targets[0] if targets else None
            inv = INVERSE.get(verb_of(taken), ())
            parent = getattr(self, "s0_parent", None)
            for e in templ():
                a, ids = e["action"], list(e.get("obj_ids") or [])
                if a.startswith(EXCLUDE_PREFIXES):
                    continue
                v = verb_of(a)
                if is_move and a.startswith(("go to", "teleport to")):
                    ranked.append((0 if self._goes_back(a) else 1, a))
                elif ids and ids[0] == primary and v in inv:
                    # the specific inverse; for 'move X to Y' prefer moving X back where it was
                    back = v != "move" or (len(ids) > 1 and ids[1] == parent)
                    ranked.append((0 if back else 1, a))
                elif ids and ids[0] == primary and v in UNDO_VERBS:
                    ranked.append((2, a))
                elif ids and ids[0] in targets:
                    ranked.append((3, a))
                elif any(i in targets for i in ids):
                    ranked.append((4, a))
        else:   # fallback: name matching (fake world in the tests, or very old API)
            valid = self.env.get_valid_action_object_combinations()
            objs = sorted(set(self.env.get_possible_objects()), key=len, reverse=True)
            referents = [o for o in objs if o and re.search(r"\b" + re.escape(o) + r"\b", taken)]
            for a in valid:
                if a.startswith(EXCLUDE_PREFIXES):
                    continue
                if is_move and a.startswith(("go to", "teleport to")):
                    ranked.append((0 if self._goes_back(a) else 1, a))
                elif _mentions(a, referents):
                    ranked.append((0 if verb_of(a) in UNDO_VERBS else 1, a))
        seen, out = set(), []
        for _, a in sorted(ranked):
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out[: self.max_branch]

    def volatile(self, n_ticks: int = 3) -> bool:
        snap = self.history
        k0 = self.exact_key()
        for _ in range(n_ticks):
            self._raw_step("look around")
            self.history = self.history + ("look around",)
        k1 = self.exact_key()
        self.restore(snap)
        return k0 != k1


def state_text_from_env(env) -> str:
    look = env.look()
    inv = env.inventory()
    return f"Task: {env.get_task_description()}\nObservation:\n{look.strip()}\nInventory:\n{inv.strip()}"


def sample_actions(env, rng: random.Random, k: int, gold_next: str | None = None) -> list[str]:
    """Pick k sensible state-changing actions with diverse verbs. The next gold-path action is
    always included when it changes state, so the dataset also contains actions an agent
    solving the task would really take."""
    templ = getattr(env, "get_valid_action_object_combinations_with_templates", None)
    if templ is not None:
        idx = object_index(env.getObjectTree())
        pool = [e["action"] for e in templ() if is_sensible(e, idx)]
    else:
        pool = [a for a in env.get_valid_action_object_combinations() if not a.startswith(EXCLUDE_PREFIXES)]
    out = []
    if gold_next and gold_next in pool:
        out.append(gold_next)
    by_verb: dict[str, list[str]] = {}
    for a in pool:
        if a not in out:
            by_verb.setdefault(verb_of(a), []).append(a)
    verbs = sorted(by_verb)
    rng.shuffle(verbs)
    for v in verbs:
        if len(out) >= k:
            break
        out.append(rng.choice(sorted(by_verb[v])))
    return out[:k]


def label_episode(job: dict) -> list[dict]:
    from scienceworld import ScienceWorldEnv   # imported here so the module loads without Java

    rng = random.Random(job["seed"])
    NOOP_COUNT[0] = 0
    _TOL["exact"] = job.get("exact_tol", EXACT_TOL)
    _TOL["approx"] = job.get("approx_tol", APPROX_TOL)
    env = ScienceWorldEnv("", envStepLimit=10_000)
    task, var = job["task"], job["variation"]
    ad = SciWorldAdapter(env, task, var, max_branch=job["max_branch"])
    env.load(task, var, SIMPLIFICATIONS, generateGoldPath=True)
    gold = env.get_gold_action_sequence()
    ad._reload(())
    # walk along the gold path (mixed with a few random steps) and stop at sampled states
    trajectory: list[str] = []
    stops = sorted(rng.sample(range(0, min(len(gold), job["max_history"])),
                              k=min(job["states_per_episode"], max(1, min(len(gold), job["max_history"])))))
    recs = []
    for i, a in enumerate(gold[: job["max_history"]]):
        if i in stops:
            ad.restore(tuple(trajectory))
            st_text = state_text_from_env(env)
            vol = ad.volatile()
            gold_next = next((g for g in gold[i:] if not g.startswith(EXCLUDE_PREFIXES)), None)
            kept_here = 0
            for act in sample_actions(env, rng, 3 * job["actions_per_state"], gold_next=gold_next):
                if kept_here >= job["actions_per_state"]:
                    break
                t0 = time.time()
                ad.restore(tuple(trajectory))
                ad.prepare(act)                         # remember the action's object ids
                res = bfs_label(ad, act, max_depth=job["horizon"])
                if res.action_failed:
                    continue
                if res.exact_depth == 0 and job.get("drop_noops", True):
                    # the action changed nothing (it silently failed, e.g. closing a door that the
                    # openDoors simplification keeps open, or only observed). Not a real decision.
                    NOOP_COUNT[0] += 1
                    continue
                kept_here += 1
                labels = res.labels(tuple(range(1, job["horizon"] + 1)))
                scen = f"sw_{task}_{var}"
                recs.append({
                    "id": f"{scen}_t{i}_{_h(act)[:10]}", "environment": "scienceworld",
                    "state_text": st_text, "action_text": act,
                    "label": labels[job["horizon"]], "horizon": job["horizon"], "source": f"scienceworld:{task}",
                    "scenario_id": scen, "group_id": None, "is_minimal_pair": False,
                    "verb": verb_of(act), "tool": verb_of(act), "args": {},
                    "is_gold_action": act == gold_next, "target_ids": list(getattr(ad, "target_ids", [])),
                    "undo_witness": list(res.witness), "tolerances": dict(_TOL),
                    "labels_by_horizon": {str(h): l for h, l in labels.items()},
                    "task_text": env.get_task_description(), "volatile": vol,
                    "history": list(trajectory),
                    "bfs": {"exact_depth": res.exact_depth, "approx_depth": res.approx_depth,
                            "nodes": res.nodes_expanded, "truncated": res.truncated,
                            "seconds": round(time.time() - t0, 2)},
                })
                ad.restore(tuple(trajectory))
        trajectory.append(a)
    env.close()
    for r in recs:
        r["episode_noops_skipped"] = NOOP_COUNT[0]
    return recs


def add_minimal_pairs(recs: list[dict]) -> list[dict]:
    """ScienceWorld minimal pairs: the same action text in the same scenario but different
    states with different labels (e.g. 'move water to sink' before/after heating). We tag
    them with a shared group_id so the minimal-pair analyses can use them."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        by_key.setdefault((r["scenario_id"], r["action_text"]), []).append(r)
    for (scen, act), rs in by_key.items():
        if len({r["label"] for r in rs}) >= 2:
            gid = f"{scen}_{_h(act)[:10]}_g"
            for r in rs:
                r["group_id"], r["is_minimal_pair"] = gid, True
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["boil", "melt", "freeze", "change-the-state-of-matter-of",
                                                   "chemistry-mix", "chemistry-mix-paint-secondary-color",
                                                   "use-thermometer", "measure-melting-point-known-substance",
                                                   "grow-plant", "find-living-thing"])
    ap.add_argument("--variations", type=int, default=5)
    ap.add_argument("--states-per-episode", type=int, default=4)
    ap.add_argument("--actions-per-state", type=int, default=3)
    ap.add_argument("--max-history", type=int, default=15)
    ap.add_argument("--max-branch", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=2, help="2 for the main ScienceWorld labels; 3 on a sample")
    ap.add_argument("--exact-tol", type=float, default=EXACT_TOL)
    ap.add_argument("--approx-tol", type=float, default=APPROX_TOL)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sciworld_pairs.jsonl")
    a = ap.parse_args()

    jobs = [dict(task=t, variation=v, seed=a.seed * 1000 + i * 31 + v, states_per_episode=a.states_per_episode,
                 actions_per_state=a.actions_per_state, max_history=a.max_history, max_branch=a.max_branch,
                 horizon=a.horizon, exact_tol=a.exact_tol, approx_tol=a.approx_tol)
            for i, t in enumerate(a.tasks) for v in range(a.variations)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    # resume: the .partial file starts with a header holding the exact settings (and a code
    # version). Finished episodes follow (records + a "_done" marker each). If the header does
    # not match the current run, the old file is set aside and the run starts fresh, so results
    # from older code or other settings can never mix in.
    part = Path(a.out + ".partial")
    settings = {k: v for k, v in jobs[0].items() if k not in ("task", "variation", "seed")} if jobs else {}
    config = {"code_version": CODE_VERSION, **settings, "seed": a.seed}
    recs: list[dict] = []
    done: set[str] = set()
    if part.exists():
        lines = [json.loads(l) for l in open(part) if l.strip()]
        if lines and lines[0].get("_config") == config:
            for r in lines[1:]:
                if "_done" in r:
                    done.add(r["_done"])
                else:
                    recs.append(r)
            done_scen = {"sw_" + d.replace("|", "_") for d in done}
            recs = list({r["id"]: r for r in recs if r["scenario_id"] in done_scen}.values())   # dedupe by id
        else:
            old = part.with_suffix(part.suffix + ".old")
            part.rename(old)
            print(f"settings or code changed since {part.name} was written: moved it to {old.name}, starting fresh")
    if not part.exists():
        with open(part, "w") as f:
            f.write(json.dumps({"_config": config}) + "\n")
    todo = [j for j in jobs if f"{j['task']}|{j['variation']}" not in done]
    print(f"{len(done)} episodes already done ({len(recs)} records), {len(todo)} to run", flush=True)

    with Pool(a.workers) as pool, open(part, "a") as fpart:
        for i, (job, rs) in enumerate(zip(todo, pool.imap(label_episode, todo))):
            recs.extend(rs)
            for r in rs:
                fpart.write(json.dumps(r) + "\n")
            fpart.write(json.dumps({"_done": f"{job['task']}|{job['variation']}"}) + "\n")
            fpart.flush()
            print(f"[{len(done) + i + 1}/{len(jobs)}] {job['task']} v{job['variation']}: +{len(rs)} "
                  f"(total {len(recs)})", flush=True)
    recs = add_minimal_pairs(recs)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("class counts:", Counter(r["label"] for r in recs))
    print("volatile share:", sum(r["volatile"] for r in recs) / max(1, len(recs)))
    print("minimal-pair share:", sum(r["is_minimal_pair"] for r in recs) / max(1, len(recs)))
    print("gold-action share:", sum(r.get("is_gold_action", False) for r in recs) / max(1, len(recs)))
    noops = {r["scenario_id"]: r.get("episode_noops_skipped", 0) for r in recs}
    print("no-op actions skipped (changed nothing):", sum(noops.values()))
    print("verbs:", Counter(r["verb"] for r in recs).most_common(10))

if __name__ == "__main__":
    main()
