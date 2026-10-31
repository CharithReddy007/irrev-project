# Cluster runbook: the whole project, task by task

Do the tasks **in this order**. Each task says where it runs, the exact command, how long it
should take and how much time to request. **Do not continue past a ⛔ checkpoint until it passes.**

Time estimates are for one A100 (bf16) GPU. On a V100, double the GPU times and set
`IRREV_DTYPE=float16` in `cluster.env`. Waiting in the queue is not included.


> **Our DGX (dgx-login01):** we run everything from Jupyter inside an `srun` session, using
> `run_on_cluster.ipynb`. Its cells are the same tasks as the tables below. `cluster.env` is
> already filled in for this cluster (venv instead of conda; `gpu_student` / `cpu_student`;
> `gpu:a100:1`). The 5 GB and 10 GB MIG slices are too small for the models.

---

## 0. Once per terminal session (login node)

Use `tmux` so a dropped connection does not kill an interactive `srun`:

```bash
tmux new -s irrev            # later: tmux attach -t irrev
cd $HOME/irrev-project        # or wherever you unzipped it
source scripts/cluster/cluster.env
# short names for the two resource requests (copy these lines exactly)
GPU="srun -p $GPU_PARTITION --gres=$GPU_GRES ${ACCOUNT:+-A $ACCOUNT} $EXTRA_SBATCH"
CPU="srun -p $CPU_PARTITION ${ACCOUNT:+-A $ACCOUNT} $EXTRA_SBATCH"
```

Every model task is `bash scripts/cluster/job.sh <task> <model>`, and `<model>` is `qwen7b` or
`llama8b`. The job script loads modules, activates the environment and switches the compute node
to offline mode, so you never do that yourself. Logs are printed to the terminal.

> Prefer batch jobs? `bash scripts/cluster/submit.sh <task> <model>` submits the same task with
> the same resources. Logs go to `logs/`. You can do either; the results are identical.

---

## Phase A: set up (login node, needs internet). About 1 hour

| # | Task | Command | Takes |
|---|---|---|---|
| A1 | Check the cluster | `bash scripts/cluster/01_check_cluster.sh` | 1 min |
| A2 | Fill in `scripts/cluster/cluster.env` from A1's output (the lines marked FILL) | editor | 5 min |
| A3 | Create the environment | `bash scripts/cluster/02_setup_env.sh` | 20–30 min |
| A4 | Log in to Hugging Face (token is kept outside the code) | `conda activate $ENV_DIR && huggingface-cli login` | 1 min |
| A5 | Download Qwen | `bash scripts/cluster/03_download.sh qwen7b` | 5–15 min |
| A6 | Download Llama | `bash scripts/cluster/03_download.sh llama8b` | 5–15 min |
| A7 | Download ByteSized32 (RQ5) | `bash scripts/cluster/03_download.sh bytesized` | 1 min |

⛔ **Checkpoint A:** A3 prints `scienceworld import OK` and a Java 17 version. A5 and A6 print
`downloaded to ...`.

---

## Phase B: data (CPU). About 2 hours of wall time

| # | Task | Command | Takes | Request |
|---|---|---|---|---|
| B1 | Unit tests + synthetic demo | `$CPU -c 4 --mem=16G -t 00:30:00 bash scripts/cluster/job.sh tests` | 5 min | 30 min |
| B2 | Sandbox pairs + agent tasks | `$CPU -c 2 --mem=8G -t 00:15:00 bash scripts/cluster/job.sh data` | 1 min | 15 min |
| B3 | ScienceWorld pilot | `$CPU -c 4 --mem=16G -t 00:30:00 bash scripts/cluster/job.sh sciworld_pilot` | 3–5 min | 30 min |
| B4 | ScienceWorld full labelling | `$CPU -c 16 --mem=64G -t 03:00:00 bash scripts/cluster/job.sh sciworld` | 30–90 min | 3 h |
| B5 | Look at the ScienceWorld labels | `$CPU -c 2 --mem=8G -t 00:15:00 bash scripts/cluster/job.sh sciworld_check` | 1 min | 15 min |
| B6 | Build + freeze dataset, QC sheet | `$CPU -c 2 --mem=8G -t 00:15:00 bash scripts/cluster/job.sh build` | 1 min | 15 min |
| B7 | *(optional)* horizon-3 sample, can run in the background | `bash scripts/cluster/submit.sh sciworld_h3` | 2–6 h | 12 h |

Expected numbers:
- B1: every test prints `ok` and the demo ends with `Demo finished`.
- B2: `wrote 1953 records`, class counts `858 / 305 / 790`, minimal-pair share `60.5%`, `130 tasks (65 pairs)`.
- B3: a few records without a Python error.
- B4: if it is killed, run the same command again; it resumes from the `.partial` file.

⛔ **Checkpoint B (B5).** Read the 8 sample lines per label and check:
- `go to ...` is mostly reversible;
- `eat`, `pour` into the sink or drain, and `burn` are irreversible;
- `open` / `close` / `pick up` are reversible;
- B6 says `overall minimal-pair share >= 30%` and the total is ≥ 2,000 records.

If something looks wrong, send me the B5 output **before** going on.
`data/dataset_v1_FROZEN.jsonl` is then the dataset for every later step.

Meanwhile, two people fill in `data/qc_sheet.csv` independently (column `label`), save it as
`data/qc_sheet_filled.csv` and run `$CPU -c 1 -t 00:10:00 bash scripts/cluster/job.sh qc_score`.
That gives you Cohen's kappa for the report.

---

## Phase C: GPU work for Qwen. About 4–6 GPU hours

| # | Task | Command | Takes | Request |
|---|---|---|---|---|
| C1 | **Smoke test** | `$GPU -c 8 --mem=64G -t 00:30:00 bash scripts/cluster/job.sh smoke qwen7b` | 5–10 min | 30 min |
| C2 | Hidden states (RQ1) | `$GPU -c 8 --mem=64G -t 01:00:00 bash scripts/cluster/job.sh hidden qwen7b` | 10–20 min | 1 h |
| C3 | Probes, 8 configurations (CPU) | `$CPU -c 32 --mem=96G -t 04:00:00 bash scripts/cluster/job.sh probes qwen7b` | 30–90 min | 4 h |
| C4 | Stated answers (RQ2) | `$GPU -c 8 --mem=64G -t 01:00:00 bash scripts/cluster/job.sh stated qwen7b` | 10–20 min | 1 h |
| C5 | RQ2 comparison (CPU) | `$CPU -c 4 --mem=16G -t 00:30:00 bash scripts/cluster/job.sh rq2 qwen7b` | 2 min | 30 min |
| C6 | Agent, neutral prompt (RQ3) | `$GPU -c 8 --mem=64G -t 02:00:00 bash scripts/cluster/job.sh agent qwen7b` | 20–50 min | 2 h |
| C7 | Agent, careful prompt | `$GPU -c 8 --mem=64G -t 02:00:00 bash scripts/cluster/job.sh agent_careful qwen7b` | 20–50 min | 2 h |
| C8 | Knowledge K, forced choice | `$GPU -c 8 --mem=64G -t 00:30:00 bash scripts/cluster/job.sh knowledge qwen7b` | 5 min | 30 min |
| C9 | Gap = K − A, McNemar (CPU) | `$CPU -c 4 --mem=16G -t 00:30:00 bash scripts/cluster/job.sh analysis qwen7b` | 1 min | 30 min |

⛔ **Checkpoint C1:** the smoke test must end with `ALL CHECKS PASSED`. If anything fails, stop
and send me the whole output. Nothing after C1 is valid until it passes.

⛔ **Before C6:** commit `docs/preregistration.md` (git commit or a dated email to the team).
The agent results must not influence what you said you would measure.

What to check:
- C3 writes `results/qwen7b/probes/*.json` (the best layer is in `summary.best_layer`). The probe should clearly beat the lexical baselines, and the
  shuffled-label and control-task scores should sit near chance.
- C6 and C7 each write 130 episodes.

C2, C4, C6, C7 and C8 each load the model once. If your cluster prefers one long interactive
session, you can run them back to back in a single allocation:

```bash
$GPU -c 8 --mem=64G -t 04:00:00 --pty bash      # opens a shell on the GPU node
for t in hidden stated agent agent_careful knowledge; do bash scripts/cluster/job.sh $t qwen7b || break; done
exit
```

(Run the smoke test separately first, and the probes on CPU afterwards.)

---

## Phase D: the same for Llama

Repeat C1 to C9 with `llama8b` in place of `qwen7b`. The time is about the same.

---

## Phase E: bonus, steering (RQ4) and RQ5

| # | Task | Command | Takes | Request |
|---|---|---|---|---|
| E1 | Steering direction (CPU) | `$CPU -c 4 --mem=32G -t 00:30:00 bash scripts/cluster/job.sh steerdir qwen7b` | 2 min | 30 min |
| E2a | 28 steering runs as an array (preferred) | `bash scripts/cluster/submit.sh steer qwen7b` | 20–40 min each | 2 h each |
| E2b | …or one at a time with srun, ID 0–27 | `$GPU -c 8 --mem=64G -t 02:00:00 bash scripts/cluster/job.sh steer qwen7b 0` | 20–40 min | 2 h |
| E2c | …or all 28 in one long job | `$GPU -c 8 --mem=64G -t 24:00:00 bash scripts/cluster/job.sh steer_all qwen7b` | 10–18 h | 24 h |
| E3 | Steering analysis (CPU) | `$CPU -c 4 --mem=16G -t 00:30:00 bash scripts/cluster/job.sh steer_analysis qwen7b` | 1 min | 30 min |
| E4 | RQ5 ByteSized32 | `$GPU -c 8 --mem=64G -t 02:00:00 bash scripts/cluster/job.sh rq5 qwen7b` | 30–60 min | 2 h |

Choose **one** of E2a, E2b or E2c.
- The array finishes fastest if your cluster runs jobs in parallel.
- With E2b, you can split the IDs across teammates (e.g. person 1 runs 0–6, person 2 runs 7–13, ...).
- Every run resumes where it stopped, so a killed run can simply be started again.

Steering on Qwen alone is enough for the bonus. Run it on Llama too only if there is time left.

---

## Phase F: figures (CPU)

| # | Task | Command | Takes | Request |
|---|---|---|---|---|
| F1 | All figures + ablation table | `$CPU -c 2 --mem=8G -t 00:30:00 bash scripts/cluster/job.sh figures` | 1 min | 30 min |

Figures go to `results/figures/`. Copy them to your laptop with
`scp -r <user>@<cluster>:~/irrev-project/results/figures .`

---

## Total budget

| | Compute | Wall time if run back to back |
|---|---|---|
| Setup (login node) | – | ~1 h |
| Data (CPU) | ~3 CPU-h × 16 cores | ~2 h |
| Qwen core (C1–C9) | ~2–3 GPU-h + ~2 h CPU | ~4–5 h |
| Llama core (D) | ~2–3 GPU-h + ~2 h CPU | ~4–5 h |
| **Required parts total** | **~5–6 GPU-h** | **~1.5 days including queues** |
| Steering, Qwen | 10–18 GPU-h | 1 night (array: a few hours) |
| RQ5, both models | ~2 GPU-h | ~2 h |

A sensible plan:
- **Day 1:** phases A and B.
- **Day 2:** C, then D.
- **Night 2:** steering array.
- **Day 3:** E3, E4, F, then start writing.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | request a GPU with more memory (≥ 24 GB); for agent runs, add a smaller batch, e.g. `python -m irrev.agent.run --model qwen7b --batch-size 4` inside an srun |
| `OSError: ... offline mode` / model not found | the download (A5/A6) did not finish, or `HF_HOME` differs between login and compute node; re-run A5 |
| Job killed at the time limit | run the same command again; every stage resumes |
| `Java gateway` / `py4j` error in ScienceWorld | check `java -version` inside the environment (it must be 17); send me the log |
| Llama 401 / gated error | accept the licence on the model page with the same HF account, run A4 again, then A6 |

Send me the **full** output of any failing command; do not change code yourself first.
