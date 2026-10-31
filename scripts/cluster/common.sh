#!/usr/bin/env bash
# Sourced by every job script: loads modules, activates the environment, sets offline mode.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"

if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
  module load $MODULES
fi

# activate the project environment: a Python venv (bin/activate) or a conda env
if [ -f "$ENV_DIR/bin/activate" ] && [ ! -d "$ENV_DIR/conda-meta" ]; then
  source "$ENV_DIR/bin/activate"
elif [ -d "$ENV_DIR" ]; then
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  [ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
fi

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
# Compute nodes often have no internet: use only the pre-downloaded cache.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export MPLBACKEND=Agg
cd "$PROJECT_DIR"
mkdir -p logs
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || true
