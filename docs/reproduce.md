# Reproducing the results on a SLURM cluster

This is how the results in `results/` were produced (31 Oct 2026, university DGX, one A100
40 GB, 16 CPU cores). Times are the measured ones; queue waiting is not included.
`scripts/cluster/cluster.env` holds the cluster settings. Edit it first on another cluster
(`bash scripts/cluster/01_check_cluster.sh` prints what to fill in).

## 1. Get a session

Start `tmux` on the login node first, so a dropped connection does not end the job.

```bash
cd ~
tmux new -s irrev                       # later: tmux attach -t irrev
# setup and data (Phases A-B):
srun -N 1 --ntasks-per-node=16 --gres=gpu:a100_1g.5gb:1 --mem=64000 --time=04:00:00 --partition=gpu_student --pty /bin/bash
# models (Phases C-F): a 7B/8B model needs ~16 GB, so the 5 GB / 10 GB slices are too small
srun -N 1 --ntasks-per-node=16 --gres=gpu:a100:1 --mem=96000 --time=12:00:00 --partition=gpu_student --pty /bin/bash
```

Run `cd ~` before `srun` and before Jupyter; starting either inside a folder that is later
deleted or re-cloned breaks it. To work in the browser, start Jupyter on the compute node and
open the URL it prints:

```bash
jupyter-notebook --notebook-dir=$HOME --port=1263 --ip=$(hostname -i)
```

`run_on_cluster.ipynb` runs every stage below from cells, with `run("<stage>", MODEL)`.
In a terminal, the same stage is `bash scripts/cluster/job.sh <stage> <model>`.

## 2. Setup (once, needs internet)

| Step | Command | Time |
|---|---|---|
| Environment (venv, PyTorch cu121, transformers < 5, ScienceWorld, Jupyter kernel) | `bash scripts/cluster/02_setup_env.sh` | ~25 min |
| Hugging Face login (token from huggingface.co/settings/tokens, type "Read") | `huggingface-cli login` | 1 min |
| Model weights (~30 GB in `~/hf_cache`) | `bash scripts/cluster/03_download.sh qwen7b`, then `llama8b` | ~15 min |

Check: the setup script ends with `scienceworld import OK`, and `torch.cuda.is_available()` is True.
The compute nodes of this cluster have internet; if yours do not, run these on the login node.

## 3. Data (CPU)

| Stage | Time | Expected output |
|---|---|---|
| `tests` | 2.6 min | all tests `PASS`, then `Demo finished` |
| `data` | 0.1 min | 1953 records, 858 / 305 / 790, minimal-pair share 60.5%, 130 tasks (65 pairs) |
| `sciworld` | 15 min (16 workers) | 600 records, 384 / 33 / 183; resumes from `.partial` if interrupted |
| `sciworld_check` | <1 min | 8 sample records per label, for reading by eye |
| `build` | <1 min | 2553 records, minimal-pair share 46.7%, QC sheet of 100 rows |

## 4. Models (one A100, per model: `qwen7b`, then `llama8b`)

| Stage | Time (Qwen / Llama) | Notes |
|---|---|---|
| `smoke` | 0.2 min | must end with `ALL CHECKS PASSED`; line [5] shows how the first tool call parsed |
| `hidden` | 2.7 / 2.9 min | |
| `probes` | 55 min / ~1 h | CPU-bound, 16 workers |
| `stated` | 2 min | |
| `rq2` | 3.7 min | |
| `agent`, `agent_careful` | 2.3 + 3.4 min | |
| `knowledge`, `analysis` | 0.5 + <1 min | `analysis` prints the HEADLINE gap |

Then `figures` (<1 min). Per model, the GPU work takes well under one hour.

## Problems we hit, and the fixes now in the code

| Symptom | Cause | Fix |
|---|---|---|
| `Disabling PyTorch because PyTorch >= 2.6 is required` | pip installed transformers 5.x | pinned `transformers<5` in setup and requirements |
| `rq2` running for an hour at 1600% CPU | multi-threaded BLAS on a crowded shared node | `common.sh` sets one BLAS thread per process |
| Llama: every agent step `parse failed` | Llama writes `{"tool_name": {args}}` | the parser accepts that shape and counts it as a format violation |
| Jupyter kernel: "No such file or directory" | Jupyter started inside a folder that was then deleted | start Jupyter from `~` |
| `git push` rejects the password | GitHub needs a personal access token | use a token with Contents: read and write |
