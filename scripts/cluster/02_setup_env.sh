#!/usr/bin/env bash
# Step 2. Run ONCE (needs internet):  bash scripts/cluster/02_setup_env.sh
# Creates a Python virtual environment with PyTorch, transformers, scikit-learn and ScienceWorld,
# and registers it as a Jupyter kernel. Uses the system Python + Java (no conda needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
PY=${PYTHON_BIN:-python3}
mkdir -p "$(dirname "$ENV_DIR")" "$HF_HOME"

if [ ! -f "$ENV_DIR/bin/activate" ]; then
  if ! $PY -m venv "$ENV_DIR" 2>/dev/null; then
    echo "python venv module missing -> using virtualenv instead"
    rm -rf "$ENV_DIR"
    $PY -m pip install --user -q virtualenv
    $PY -m virtualenv "$ENV_DIR"
  fi
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/$TORCH_CUDA
python -m pip install "transformers>=4.45" "accelerate>=0.33" "huggingface_hub" numpy scipy \
    "scikit-learn>=1.4" joblib matplotlib pandas pytest scienceworld ipykernel
# register the environment as a Jupyter kernel called "irrev (project env)"
python -m ipykernel install --user --name irrev --display-name "irrev (project env)"

cd "$PROJECT_DIR"
python -m pip freeze > requirements.lock.txt
echo
python - <<'PY'
import torch, transformers, sklearn
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| GPU visible:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| sklearn", sklearn.__version__)
import scienceworld; print("scienceworld import OK")
PY
java -version 2>&1 | head -1
echo "Environment ready at $ENV_DIR. Exact versions saved to requirements.lock.txt (commit it)."
