# Data card: Reversibility-labelled (state, action) pairs

*Fill every [bracket] from your final files. Numbers below in plain text are from the current
sandbox generator (seed 0, 700 scenarios) and will change once ScienceWorld is added.*

## Summary
Labelled (state, action) pairs for studying whether LLM agents represent the reversibility of
actions. Each pair is labelled reversible, partially reversible or irreversible by a bounded
breadth-first search in the environment.

## Files and format
`dataset.jsonl`, one JSON object per line. Required fields:

| Field | Meaning |
|---|---|
| `id` | unique id |
| `environment` | `sandbox` or `scienceworld` |
| `state_text` | the state as shown to the model |
| `action_text` | the candidate action (JSON tool call in the sandbox, a command in ScienceWorld) |
| `label` | `reversible` / `partially_reversible` / `irreversible` at the main horizon |
| `horizon` | main horizon (3) |
| `source` | `generated`, `scienceworld:<task>` or `toolemu:<case>` |

Extra fields: `scenario_id`, `group_id` (minimal-pair group), `is_minimal_pair`, `verb`,
`labels_by_horizon` (1, 2, 3), `task_text`, `settings`, `bfs` (search depth, nodes, truncation),
`volatile` (ScienceWorld), `split`, `split_0..2`.

## How it was built
- **Sandbox**: random workspaces (3-7 files, random settings), one minimal-pair group plus up to
  three ordinary actions per scenario. Labels from the BFS labeller with relevance pruning
  (verified equal to the full search on random small states, `tests/test_labeling.py`).
- **ScienceWorld**: states sampled along gold trajectories of [tasks] x [variations], with easy
  simplifications (teleport, open doors, no electrical actions). Labels by reload-and-replay BFS
  over valid actions mentioning an object of the labelled action (branching cap [20]).
- **ToolEmu-derived** (if used): [n] pairs written by hand from the 144 curated case descriptions.

## Composition
| | Reversible | Partial | Irreversible | Total | Minimal-pair share |
|---|---|---|---|---|---|
| Sandbox (current) | 858 | 305 | 790 | 1,953 | 60.5% |
| ScienceWorld | [ ] | [ ] | [ ] | [ ] | [ ] |
| **Total** | [ ] | [ ] | [ ] | [ ] | [ ] |

Labels by horizon (sandbox, current): h=1: 704 / 406 / 843; h=2 and h=3: 858 / 305 / 790.

Classes are not perfectly balanced (the partial class is smaller); all results use balanced accuracy.

## Splits
20% of scenarios per environment form the test set (seed 12345); three more scenario-level splits
(`split_0..2`) are used for the repeated probe runs. Members of a minimal-pair group always share a
split.

## Quality control
Two annotators labelled the same 100 random pairs blind to the automatic label.
Cohen's kappa A-B: [ ]; A-auto: [ ]; B-auto: [ ]. Main disagreement types: [ ].

## Known limitations
- Synthetic, short sandbox states; settings are stated explicitly in text, so the task partly tests
  reading comprehension of settings combined with action semantics.
- Bounded horizon (<= 3) and restricted inverse-candidate sets; some "irreversible" labels may be
  reversible with longer plans.
- ScienceWorld time passes on every step; [x]% of states are volatile.
- Setting phrases ("cannot be cancelled") give a lexical cue in the state text; a bag-of-words
  over state + action reaches ~0.68 balanced accuracy on sandbox minimal pairs.
- English only.

## Licence and intended use
Research on agent safety and interpretability. Sandbox data: [your licence, e.g. CC BY 4.0].
ScienceWorld-derived data follows ScienceWorld's Apache-2.0 licence. Do not include
ByteSized32-derived data in the release.
