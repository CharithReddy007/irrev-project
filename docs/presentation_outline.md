# 15-minute presentation: outline and speaker notes

Two speakers per half works well; switch at slide 7. Rehearse with a timer twice.
Demo is required: one reversible and one irreversible version of the same task.

| # | Slide | Time | Speaker notes |
|---|---|---|---|
| 1 | **Title**: "Do LLM agents know what they cannot undo?" | 0:20 | Names, one-line question. |
| 2 | **Motivation**: agents now delete, send, pay. Opening a file vs deleting without a trash. | 1:00 | Concrete example. A careful human slows down before an irreversible step. |
| 3 | **Two questions**: does the model represent reversibility? Does it use it? Define *irreversibility blindness*. | 1:00 | "Says deleting is permanent, deletes anyway." |
| 4 | **Gap in the literature**: RL reversibility, ToolEmu/SafeAgentBench (risk in general), Tang et al. (risk knowledge gap, not reversibility). | 0:50 | Be precise about what is new. |
| 5 | **Sandbox + labels**: settings table; BFS labelling picture (state -> action -> search back within 3 steps). | 1:20 | Exact by construction. Partial = restored only approximately or at a cost. |
| 6 | **Minimal pairs**: same `delete_file`, trash on vs off; two state texts side by side, one line differs. | 1:00 | Why this kills the "the word delete is scary" shortcut. |
| 7 | **Three measurements**: probe / ask / act on the same items. Gap = K - A. | 0:50 | One diagram. |
| 8 | **RQ1 result**: probe accuracy vs layer figure with baselines. | 1:30 | Point at the minimal-pair line and the lexical baselines. Selectivity. |
| 9 | **RQ2 result**: stated accuracy, kappa with probe, one example where they disagree. | 1:00 | |
| 10 | **Demo** (recorded, ~1:30): `scripts/demo_agent.py --model qwen7b --pair <pair>` reversible, then irreversible. | 1:40 | Pick a pair where behaviour is typical, not cherry-picked; say so. |
| 11 | **RQ3 result**: gap figure with CI; per-family table. | 1:30 | The headline number. Careful-prompt ablation. |
| 12 | **Bonus**: steering curve (probe vs random) and/or RQ5 AUROC. | 1:00 | Only if done with controls. |
| 13 | **Limitations** (honest): small sandbox, 2 families, horizon 3, time in ScienceWorld, linear probes. | 0:40 | Reviewers reward this. |
| 14 | **Takeaway**: one sentence finding + what it means for agent design. Released: dataset, code, logs. | 0:20 | |

**Demo recording tips.** Record with `asciinema` or a screen recorder; run the model beforehand and
replay from the saved log if the GPU is busy. Show the settings line that differs before running.
Keep a backup video file locally.

**Likely questions to prepare:**
- "Isn't the probe just reading the settings sentence?" -> lexical state+action baseline, minimal pairs, selectivity.
- "Why linear probes?" -> Alain & Bengio; Hewitt & Liang; deep probes can learn the task themselves.
- "Is the agent just bad at tool use?" -> success rates, parse-failure counts, careful-prompt ablation.
- "Does steering prove causation?" -> random-direction control, dose-response, success alongside caution.
