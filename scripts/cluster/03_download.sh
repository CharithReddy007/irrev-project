#!/usr/bin/env bash
# Step 3. Needs internet (login node, or a compute node if it has internet). Downloads model weights into $HF_HOME and the
# ByteSized32 data, so compute jobs can run offline.
#   bash scripts/cluster/03_download.sh qwen7b            # start with this one
#   bash scripts/cluster/03_download.sh llama8b           # after Meta approves your access request
#   bash scripts/cluster/03_download.sh qwen14b           # optional
#   bash scripts/cluster/03_download.sh bytesized         # RQ5 data
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
source "$ENV_DIR/bin/activate"
export HF_HUB_OFFLINE=0

dl () {  # repo id
  python - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
print("downloaded to", p)
PY
}

case "${1:?which: qwen7b | llama8b | qwen14b | bytesized}" in
  qwen7b)  dl Qwen/Qwen2.5-7B-Instruct ;;
  qwen14b) dl Qwen/Qwen2.5-14B-Instruct ;;
  llama8b)
    echo "Llama is gated. If this fails: accept the licence on the model page, then run 'huggingface-cli login'."
    dl meta-llama/Llama-3.1-8B-Instruct ;;
  bytesized)
    cd "$SCRATCH_DIR"
    [ -d GPT-simulator ] || git clone --depth 1 https://github.com/cognitiveailab/GPT-simulator
    cd GPT-simulator/data && unzip -o -q test.zip && ls -la test.jsonl
    echo "Research use only; do not redistribute (no licence file in the repo)." ;;
esac
du -sh "$HF_HOME" 2>/dev/null || true
