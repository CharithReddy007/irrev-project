#!/usr/bin/env bash
# The body of every batch job. Submit with submit.sh, not directly.
#   batch:       bash scripts/cluster/submit.sh <stage> <model>
#   interactive: srun <resources> bash scripts/cluster/job.sh <stage> <model>   (see docs/reproduce.md)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
STAGE=$1; M=${2:-qwen7b}
NCPU=${SLURM_CPUS_PER_TASK:-4}
STRENGTHS=(-4 -2 -1 0 1 2 4)

best_layer () {
  python -c "import json;print(max(1,json.load(open('results/$M/probes/binary_last_hmain_all.json'))['summary']['best_layer']))"
}

case $STAGE in
  tests)      python -m tests.run_all && python scripts/demo_synthetic.py ;;
  smoke)      python scripts/00_smoke_test.py --model $M ;;
  data)
    python -m irrev.data.gen_sandbox --n-scenarios 700 --seed 0 --out data/sandbox_pairs.jsonl
    python -m irrev.agent.tasks --out data/agent_tasks.jsonl ;;
  sciworld_pilot)
    python -m irrev.data.scienceworld_label --tasks boil --variations 1 --states-per-episode 2 \
        --actions-per-state 3 --workers 1 --out data/sciworld_pilot.jsonl ;;
  sciworld)
    python -m irrev.data.scienceworld_label --out data/sciworld_pairs.jsonl --workers $NCPU ;;
  sciworld_check)   # read labels by eye before freezing
    python - <<'PY'
import json, random, collections
recs = [json.loads(l) for l in open('data/sciworld_pairs.jsonl')]
by = collections.defaultdict(list)
for r in recs: by[r['label']].append(r)
random.seed(1)
for lab, rs in by.items():
    print(f"\n===== {lab} ({len(rs)})")
    for r in random.sample(rs, min(8, len(rs))):
        print(f"  [{r['source'].split(':')[1][:22]:22s}] {r['action_text'][:45]:45s} undo={r['undo_witness']}")
moves = [r for r in recs if r['action_text'].startswith(('go to', 'teleport to'))]
print("\nmovement labels:", dict(collections.Counter(r['label'] for r in moves)))
print("verbs:", collections.Counter(r['verb'] for r in recs).most_common(12))
PY
    ;;
  sciworld_h3)      # optional horizon-3 sample for the horizon ablation (not merged into the dataset)
    python -m irrev.data.scienceworld_label --tasks boil melt --variations 2 --states-per-episode 2 \
        --horizon 3 --workers $NCPU --seed 1 --out data/sciworld_h3_sample.jsonl ;;
  build)
    python -m irrev.data.build --inputs data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl --out data/dataset.jsonl
    python -m irrev.data.agreement sample --data data/dataset.jsonl --n 100 --out data/qc_sheet.csv
    cp data/dataset.jsonl data/dataset_v1_FROZEN.jsonl ;;
  qc_score)         # after both annotators filled data/qc_sheet_filled.csv
    python -m irrev.data.agreement score --sheet data/qc_sheet_filled.csv --key data/qc_key.json ;;
  hidden)     python -m irrev.models.extract --model $M ;;
  probes)
    J="--jobs $NCPU"
    python -m irrev.probes.run --model $M --task binary $J
    python -m irrev.probes.run --model $M --task three $J
    python -m irrev.probes.run --model $M --task binary --position mean $J
    python -m irrev.probes.run --model $M --task binary --train-subset minimal $J
    python -m irrev.probes.run --model $M --task binary --train-env sandbox --test-env scienceworld $J
    python -m irrev.probes.run --model $M --task binary --train-env scienceworld --test-env sandbox $J
    python -m irrev.probes.run --model $M --task binary --horizon 1 $J
    python -m irrev.probes.run --model $M --task binary --horizon 2 $J ;;
  stated)
    python -m irrev.stated.ask --model $M ;;
  rq2)        # CPU, after hidden + probes + stated
    python -m irrev.analysis.compare_rq2 --model $M --task binary
    python -m irrev.analysis.compare_rq2 --model $M --task three ;;
  agent)          python -m irrev.agent.run --model $M ;;
  agent_careful)  python -m irrev.agent.run --model $M --careful ;;
  knowledge)      python -m irrev.agent.knowledge --model $M ;;
  analysis)
    python -m irrev.analysis.rq3_gap --model $M --tag neutral
    python -m irrev.analysis.rq3_gap --model $M --tag careful ;;
  steerdir)   python -m irrev.agent.direction --model $M --layer $(best_layer) ;;
  steer)      # array job: task id 0..27 -> (direction, strength)
    # batch array job: SLURM_ARRAY_TASK_ID; interactive srun: pass the id as 3rd argument (0..27)
    ID=${SLURM_ARRAY_TASK_ID:-${3:?give a steering job id 0..27, or run as an array job}}
    D=$((ID / ${#STRENGTHS[@]})); S=${STRENGTHS[$((ID % ${#STRENGTHS[@]}))]}
    L=$(best_layer)
    if [ $D -eq 0 ]; then
      python -m irrev.agent.run --model $M --steer-layer $L --strengths $S
    else
      python -m irrev.agent.run --model $M --steer-layer $L --strengths $S --random-direction $((D - 1))
    fi ;;
  steer_all)  # all 28 steering runs one after another (for one long srun instead of an array)
    L=$(best_layer)
    python -m irrev.agent.run --model $M --steer-layer $L --strengths ${STRENGTHS[@]}
    for RD in 0 1 2; do
      python -m irrev.agent.run --model $M --steer-layer $L --strengths ${STRENGTHS[@]} --random-direction $RD
    done ;;
  steer_analysis) python -m irrev.analysis.rq4_steering --model $M --layer $(best_layer) ;;
  rq5)
    BS=${SCRATCH_DIR}/GPT-simulator/data/test.jsonl
    python -m irrev.rq5.bytesized label --data $BS --out data/bytesized_labeled.jsonl
    python -m irrev.rq5.bytesized score --model $M --data data/bytesized_labeled.jsonl --n 330
    python -m irrev.rq5.bytesized analyse --model $M ;;
  figures)    python -m irrev.analysis.figures --models qwen7b llama8b ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
echo "[$(date)] stage $STAGE finished"
