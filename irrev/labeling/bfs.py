"""Generic reversibility labeller (Section 4.2 of the brief).

Given a state s0 and an action a:
  1. remember exact_key(s0) and approx_key(s0)
  2. apply a -> s1
  3. breadth-first search over action sequences of length <= H from s1
  4. if some node has exact_key == exact_key(s0)   -> REVERSIBLE (at the depth found)
     elif some node has approx_key == approx_key(s0) -> PARTIALLY REVERSIBLE
     else                                            -> IRREVERSIBLE

The search records the *shallowest* depth at which each key is reached, so one search with
H=3 gives the labels for H=1, 2 and 3 at the same time (needed for ablation 4).

An environment plugs in by implementing the small Adapter protocol below. The candidate
inverse actions are generated only from (a) the current state and (b) the arguments of the
action that was just taken - i.e. information the agent itself would have. This rule
matters: otherwise "edit_file" would always look reversible, because the search could
simply type the old text back in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol

REV, PART, IRREV = "reversible", "partially_reversible", "irreversible"
LABELS = (REV, PART, IRREV)
LABEL2ID = {REV: 0, PART: 1, IRREV: 2}


class Adapter(Protocol):
    def snapshot(self) -> Any: ...
    def restore(self, snap: Any) -> None: ...
    def exact_key(self) -> Hashable: ...
    def approx_key(self) -> Hashable: ...
    def apply(self, action: Any) -> bool: ...                     # False if the action failed
    def candidates(self, taken: Any) -> list[Any]: ...           # inverse candidates from current state


@dataclass
class BFSResult:
    exact_depth: int | None           # shallowest depth where the exact state reappears
    approx_depth: int | None          # shallowest depth where the approximate state reappears
    nodes_expanded: int
    truncated: bool                   # candidate list was capped somewhere
    action_failed: bool = False
    witness: list[Any] = field(default_factory=list)   # an inverse sequence, for inspection

    def label(self, horizon: int) -> str:
        if self.exact_depth is not None and self.exact_depth <= horizon:
            return REV
        if self.approx_depth is not None and self.approx_depth <= horizon:
            return PART
        return IRREV

    def labels(self, horizons=(1, 2, 3)) -> dict[int, str]:
        return {h: self.label(h) for h in horizons}


def bfs_label(env: Adapter, action: Any, max_depth: int = 3, max_branch: int | None = None,
              max_nodes: int = 20000) -> BFSResult:
    """Label one (state, action) pair. Leaves env restored to the original state."""
    s0 = env.snapshot()
    k_exact, k_approx = env.exact_key(), env.approx_key()
    # Environments where time passes on every step (ScienceWorld) can also give "time-matched"
    # targets: refs[n] = keys of the state reached from s0 by n idle steps. A node reached by the
    # action plus d undo steps (n = d + 1 steps in total) also counts as restored if it matches
    # refs[n], i.e. "the world as it would be now if the action had never been taken".
    refs = env.reference_keys(max_depth + 1) if hasattr(env, "reference_keys") else None

    def is_exact(n_steps, key):
        return key == k_exact or (refs is not None and key == refs[n_steps][0])

    def is_approx(n_steps, key):
        return key == k_approx or (refs is not None and key == refs[n_steps][1])

    if not env.apply(action):
        env.restore(s0)
        return BFSResult(None, None, 0, False, action_failed=True)

    exact_depth = 0 if is_exact(1, env.exact_key()) else None
    approx_depth = 0 if is_approx(1, env.approx_key()) else None
    witness: list[Any] = []
    s1 = env.snapshot()
    seen = {env.exact_key()}
    frontier: list[tuple[Any, list[Any]]] = [(s1, [])]
    nodes, truncated = 0, False

    for depth in range(1, max_depth + 1):
        if exact_depth is not None:
            break                      # cannot do better than an exact match
        nxt = []
        for snap, path in frontier:
            env.restore(snap)
            cands = env.candidates(action)
            if max_branch is not None and len(cands) > max_branch:
                cands, truncated = cands[:max_branch], True
            for c in cands:
                env.restore(snap)
                if not env.apply(c):
                    continue
                nodes += 1
                ke = env.exact_key()
                if exact_depth is None and is_exact(depth + 1, ke):
                    exact_depth, witness = depth, path + [c]
                if approx_depth is None and is_approx(depth + 1, env.approx_key()):
                    approx_depth = depth
                    if not witness:
                        witness = path + [c]
                if ke not in seen:
                    seen.add(ke)
                    nxt.append((env.snapshot(), path + [c]))
                if nodes >= max_nodes:
                    truncated = True
                    break
            if nodes >= max_nodes or exact_depth is not None:
                break
        frontier = nxt
        if not frontier or nodes >= max_nodes:
            break

    env.restore(s0)
    return BFSResult(exact_depth, approx_depth, nodes, truncated, witness=witness)
