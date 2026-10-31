# Irreversibility Blindness in LLM Agents

Do open-weight LLM agents **represent** whether an action can be undone, and do they **use** that
when acting? This repository is the full pipeline for ADNN Project Option 1: a tool sandbox with
exact reversibility labels, a ScienceWorld labeller, linear probes with all controls, direct
questions, paired agent tasks, the *irreversibility blindness gap*, activation steering (RQ4) and
next-state uncertainty (RQ5).

New to the project? Read `docs/00_START_HERE.md` first.

**Running on the supercomputer (SLURM): follow `docs/01_CLUSTER_RUNBOOK.md` step by step.** It has every task in order, the exact `srun` command, and the time to request.

## Git + Jupyter (how the team runs it)

1. **Laptop, once:** unzip, `git init`, commit, push to a **private** GitHub repo, add teammates as collaborators.
2. **Cluster login node, once per person:** `git clone <repo> ~/irrev-project`, then in a terminal:
   `bash scripts/cluster/01_check_cluster.sh`, fill `scripts/cluster/cluster.env`, commit and push it,
   `bash scripts/cluster/02_setup_env.sh` (also registers the Jupyter kernel "irrev (project env)"),
   `huggingface-cli login`, `bash scripts/cluster/03_download.sh qwen7b` (and `llama8b`, `bytesized`).
3. **Jupyter:** open `run_on_cluster.ipynb` with the kernel "irrev (project env)" and run the cells in order.
   Each cell submits a SLURM job, so closing the browser does not stop anything.
4. **Code changes:** edit on the laptop, `git push`; on the cluster `git pull`. Never commit the HF token.

## Status of the code

| Part | Tested here | Needs your GPU machine |
|---|---|---|
| Sandbox, BFS labeller, dataset generator, splits, QC sheets | yes (unit tests) | - |
| ScienceWorld labeller | yes, on a fake simulator with the same API | run on real ScienceWorld (Java) |
| Probes, baselines, controls, ablations, statistics, figures | yes, on synthetic hidden states | real hidden states |
| Agent loop, tasks, caution metrics, gap, McNemar | yes, with scripted policies | real model runs |
| Model code (hidden states, generation, steering hooks) | **no** (no GPU here) | run `scripts/00_smoke_test.py` first |
| RQ5 labelling | yes, on the real ByteSized32 test set | scoring needs the model |

## Setup (GPU machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock.txt         # commit this: exact versions (brief, Section 5)
huggingface-cli login                      # Llama-3.1 is gated: accept the licence first
java -version                              # ScienceWorld needs Java >= 1.8
python -m tests.run_all                    # unit tests, CPU only, ~10 s
python scripts/demo_synthetic.py           # whole pipeline on fake data, ~2 min (outputs are SYNTHETIC)
python scripts/00_smoke_test.py --model qwen7b
```

## Pipeline

`bash scripts/cluster/job.sh <stage> <model>` runs each stage (inside `srun`, see the runbook); every step is resumable.

| Stage | What it does | RQ |
|---|---|---|
| `data` | sandbox pairs, ScienceWorld pairs, merged dataset with scenario splits, agent tasks, QC sheet | 4.2-4.3 |
| `smoke` | checks model loading, action-token span, hidden-state shapes, steering hook, determinism | - |
| `hidden` | one forward pass per pair, saves every layer (last token and mean over action) | Step 2 |
| `probes` | probes per layer + majority, lexical, shuffled, control task, layer 0; ablations 1,2,3,4,7 | RQ1 |
| `stated` | asks the model directly; compares true vs stated vs probe (kappa) | RQ2 |
| `agent` | 130 episodes (65 pairs), neutral and careful prompt (ablation 5) | RQ3 |
| `knowledge` | forced choice on each task pair, both orders (K) | gap |
| `analysis` | behaviour metrics, McNemar, gap = K - A with bootstrap CIs | RQ3 |
| `steer` | diff-of-means direction, 7 strengths, probe vs 3 random directions | RQ4 |
| `rq5` | ByteSized32 labelling, NLL scores, AUROC | RQ5 |
| `figures` | all figures and the ablation table for the report | Sec. 8 |

## Which command makes which figure or table

| Report item | File | Command |
|---|---|---|
| Probe accuracy vs layer with baselines | `results/figures/fig_probe_layers_binary_last.pdf` | `probes`, then `figures` |
| Stated vs probed vs behaviour | `results/figures/fig_three_way.pdf` | `stated`, `agent`, `knowledge`, `analysis`, then `figures` |
| Blindness gap with CIs | `results/figures/fig_gap.pdf` | `analysis`, then `figures` |
| Ablation table | `results/figures/table_ablation.tex` | `probes`, then `figures` |
| Steering curve | `results/figures/fig_steering_<model>_L<l>.pdf` | `steer`, then `figures` |
| QC agreement (kappa) | printed + JSON | `python -m irrev.data.agreement score ...` |

## Repository layout

```
irrev/sandbox/      env.py (the world and 23 tools), render.py (state text), adapter.py (labeller hookup)
irrev/labeling/     bfs.py - generic reversibility labeller, horizons 1-3 from one search
irrev/data/         gen_sandbox.py, scienceworld_label.py, toolemu_sheet.py, build.py, agreement.py
irrev/models/       lm.py (load, generate, hidden states, steering hook), prompts.py, extract.py
irrev/probes/       run.py - probes + all baselines/controls/ablations
irrev/stated/       ask.py - RQ2 questions and answer parsing
irrev/agent/        tasks.py (65 task pairs), loop.py, caution.py, run.py, knowledge.py, direction.py, policies.py
irrev/rq5/          bytesized.py
irrev/analysis/     stats.py, compare_rq2.py, rq3_gap.py, rq4_steering.py, figures.py
scripts/            00_smoke_test.py, cluster/ (job.sh, submit.sh, setup), demo_synthetic.py, demo_agent.py
docs/               start guide, team plan, pre-registration, data card, slides outline, report draft
tests/              unit tests + a fake ScienceWorld
```

## Reproducibility rules (from the brief)

* Greedy decoding everywhere, fixed seeds (`PYTHONHASHSEED=0` is set by the pipeline script).
* Every prompt and output is saved (`results/<model>/...jsonl`).
* Never load two models at once; the 14B model with batch size 1.
* All irreversible actions happen in the simulated sandbox only. No real files, email or accounts. No paid APIs.

## Data and licences

* Generated sandbox data: yours to release.
* ScienceWorld: Apache-2.0 (LICENSE file in the repository, checked October 2026).
* ToolEmu test cases: Apache-2.0. Only the case descriptions are used; the emulator is never run.
* ByteSized32-State-Prediction: no licence file; research use only, do not redistribute.
* Llama-3.1: Llama 3.1 Community License and Acceptable Use Policy.
* `docs/report/acl.sty` and `acl_natbib.bst` are from github.com/acl-org/acl-style-files.
Re-check all licences before releasing anything.
