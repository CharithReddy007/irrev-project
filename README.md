# Irreversibility Blindness in LLM Agents

Do open-weight LLM agents **represent** whether an action can be undone, and do they **use**
that knowledge when they act? This repository holds the complete pipeline, the labelled
dataset and every result file for ADNN Project Option 1. Models: **Qwen2.5-7B-Instruct** and
**Llama-3.1-8B-Instruct**, greedy decoding throughout.

We measure the same thing three ways:

| | Question | How |
|---|---|---|
| **RQ1** | Is reversibility *encoded* in the hidden states? | Linear probes on every layer, with baselines and controls |
| **RQ2** | Does the model *say* it when asked? | Direct question per (state, action) pair, compared with the probe |
| **RQ3** | Does the agent *act* on it? | 65 paired agent tasks (reversible vs irreversible version); Gap = K − A |

RQ4 (activation steering) and RQ5 (next-state uncertainty) are implemented but were **not run**.

## Main results

All numbers come from the files in `results/` (runs of 31 Oct 2026, one A100 40 GB).

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| RQ1: probe, binary, balanced accuracy (best layer) | 0.866 (layer 23 of 28) | 0.905 (layer 14 of 32) |
| RQ1: probe vs best lexical baseline, minimal pairs | 0.856 vs 0.674 | 0.920 vs 0.674 |
| RQ1: probe, three classes (chance 0.33) | 0.819 | 0.885 |
| RQ2: default answer when asked | "partially reversible" (89%) | "reversible" (95%) |
| RQ2: kappa, stated answer vs probe | 0.04 | 0.07 |
| RQ3: careful in irreversible vs reversible version | 0.02 vs 0.02 | 0.48 vs 0.49 |
| RQ3: A (more careful when irreversible) | 0.000 | 0.092 |
| RQ3: K (forced choice correct in both orders; chance 0.25) | 0.246 | 0.308 |
| RQ3: **Gap = K − A**, 95% CI (neutral prompt) | **0.246** [0.14, 0.35] | **0.215** [0.08, 0.37] |

**Reading.** Reversibility is linearly decodable from the hidden states of both models, and
this holds on minimal pairs, where the action text is identical and only one setting differs.
The information does not reach the answers (each model falls back to one default answer) or
the behaviour (caution does not depend on reversibility, A ≈ 0). K is close to its chance
level, so the positive gap mainly reflects A ≈ 0; the probes are the evidence that the
information is present.

## Repository layout

```
irrev/sandbox/      env.py (simulated workspace, 23 tools), render.py (state as text), adapter.py
irrev/labeling/     bfs.py: reversibility labeller (search for an undo sequence, horizons 1-3)
irrev/data/         gen_sandbox.py, scienceworld_label.py, build.py (merge + splits), agreement.py (QC)
irrev/models/       lm.py (load, generate, hidden states, steering hook), prompts.py, extract.py
irrev/probes/       run.py: probes per layer + baselines, controls and ablations (RQ1)
irrev/stated/       ask.py: direct questions and answer parsing (RQ2)
irrev/agent/        tasks.py (65 task pairs), loop.py, caution.py, run.py, knowledge.py (RQ3)
irrev/analysis/     stats.py, compare_rq2.py, rq3_gap.py, figures.py
irrev/rq5/, irrev/agent/direction.py, irrev/analysis/rq4_steering.py   bonus code, not run
scripts/            00_smoke_test.py, demo_agent.py, demo_synthetic.py, cluster/ (setup + job scripts)
tests/              unit tests, including a fake ScienceWorld with the same API
run_on_cluster.ipynb  the notebook that produced every result, with its outputs
data/               dataset_v1_FROZEN.jsonl (2,553 pairs), agent_tasks.jsonl, QC sheet
results/<model>/    probes/*.json, stated.jsonl, rq2_*.json, agent/*.jsonl, knowledge.jsonl, rq3_*.json
results/figures/    report figures (PDF + PNG) and table_ablation.tex
docs/               reproduce.md, data_card.md, preregistration.md, contribution_log.md, report/
```

## Reproducing the results

`docs/reproduce.md` lists every step on a SLURM cluster with the exact commands and the
measured run times. In short:

```bash
bash scripts/cluster/02_setup_env.sh                 # once: Python venv + Jupyter kernel
huggingface-cli login                                # Llama-3.1 is gated: accept its licence first
bash scripts/cluster/03_download.sh qwen7b           # and llama8b
# inside an srun session with one A100 (or open run_on_cluster.ipynb and run the cells in order):
bash scripts/cluster/job.sh tests
bash scripts/cluster/job.sh data
bash scripts/cluster/job.sh sciworld
bash scripts/cluster/job.sh build
for s in smoke hidden probes stated rq2 agent agent_careful knowledge analysis; do
  bash scripts/cluster/job.sh $s qwen7b                # then the same with llama8b
done
bash scripts/cluster/job.sh figures
```

Without a GPU, `python -m tests.run_all` and `python scripts/demo_synthetic.py` check the
installation (the demo runs the whole analysis on synthetic data; its outputs say SYNTHETIC).

| Stage | What it does |
|---|---|
| `data` | 1,953 sandbox pairs (700 scenarios, seed 0) and the 65 agent task pairs |
| `sciworld` | 600 ScienceWorld pairs (10 tasks × 5 variations), horizon 2 |
| `build` | merges both, scenario-level 80/20 splits, freezes `data/dataset_v1_FROZEN.jsonl` |
| `smoke` | checks model loading, action-token span, hidden-state shapes, steering hook, batching, parsing |
| `hidden` | one forward pass per pair; saves every layer (last token and mean over the action) |
| `probes` | 8 probe configurations: main, three-class, mean pooling, minimal-pair training, cross-environment, horizons 1 and 2 |
| `stated`, `rq2` | direct questions; stated vs true vs cross-fitted probe (kappa) |
| `agent`, `agent_careful` | 130 episodes each (65 pairs), neutral and "be careful" system prompt |
| `knowledge`, `analysis` | forced choice per pair in both orders (K); A, McNemar, Gap with bootstrap CIs |
| `figures` | the four report figures and the ablation table |

| Report item | File |
|---|---|
| Probe accuracy vs layer, with baselines | `results/figures/fig_probe_layers_binary_last.pdf`, `..._three_last.pdf` |
| Encodes / says / does | `results/figures/fig_three_way.pdf` |
| Blindness gap with CIs | `results/figures/fig_gap.pdf` |
| Ablation table | `results/figures/table_ablation.tex` |
| Demo transcript | `results/demo_dedupe_01.txt` (`scripts/demo_agent.py --model qwen7b --pair dedupe_01`) |

## Pre-registration and deviations

The analysis plan was committed before any agent run (`docs/preregistration.md`, commit
`6ab8750`). Changes made afterwards are listed in its Deviations table: the control task is
reported but not used for conclusions, the tool-call parser also accepts the format Llama
uses, and the three-way figure shows balanced accuracy.

## Reproducibility notes

* Greedy decoding, fixed seeds, `PYTHONHASHSEED=0`; bfloat16 on A100 (`IRREV_DTYPE`).
* Exact package versions: `requirements.lock.txt`. transformers must be < 5 with torch 2.5.
* Every prompt and model output is saved in `results/<model>/`.
* Agent runs generate many episodes in one batch. In bfloat16, batched and single-episode
  generation can diverge after a few steps, so `scripts/demo_agent.py` (single episode) can
  differ from the logged episode. All reported numbers use the logged batched runs.
* CPU stages run with one BLAS thread per process (`common.sh`); on a busy shared node,
  multi-threaded BLAS made `rq2` more than ten times slower.
* All "irreversible" actions happen inside the simulated sandbox. No real files, email or money.

## Data and licences

* Sandbox data: generated by this code.
* ScienceWorld: Apache-2.0.
* ByteSized32-State-Prediction (RQ5, not run): no licence file; research use only, not redistributed.
* Llama-3.1: Llama 3.1 Community License and Acceptable Use Policy.
* `docs/report/acl.sty` and `acl_natbib.bst`: github.com/acl-org/acl-style-files.
