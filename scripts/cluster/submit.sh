#!/usr/bin/env bash
# Submit one stage as a SLURM job with suitable resources.
#   bash scripts/cluster/submit.sh <stage> [model]
#   AFTER=<jobid> bash scripts/cluster/submit.sh <stage> [model]    # start only after job <jobid> succeeds
# Logs: logs/<stage>-<model>-<jobid>.out      Check jobs: squeue -u $USER
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
STAGE=${1:?stage}; M=${2:-qwen7b}

# stage -> kind(gpu/cpu) time cpus mem [array]
case $STAGE in
  smoke)            R="gpu 00:30:00 8 64G" ;;
  hidden)           R="gpu 03:00:00 8 64G" ;;
  stated)           R="gpu 03:00:00 8 64G" ;;
  agent|agent_careful) R="gpu 12:00:00 8 64G" ;;
  knowledge)        R="gpu 02:00:00 8 64G" ;;
  steer)            R="gpu 12:00:00 8 64G 0-27" ;;
  rq5)              R="gpu 06:00:00 8 64G" ;;
  steer_all)        R="gpu 24:00:00 8 64G" ;;
  tests|data|build|steerdir|rq2|analysis|steer_analysis|figures|sciworld_check|qc_score) R="cpu 02:00:00 8 32G" ;;
  sciworld_h3)      R="cpu 12:00:00 16 64G" ;;
  sciworld_pilot)   R="cpu 02:00:00 4 16G" ;;
  sciworld)         R="cpu 24:00:00 16 64G" ;;
  probes)           R="cpu 12:00:00 32 96G" ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
read KIND TIME CPUS MEM ARRAY <<< "$R"
[ "$M" = "qwen14b" ] && MEM=96G

ARGS=(-J "$STAGE-$M" -t "$TIME" -c "$CPUS" --mem "$MEM" -o "$PROJECT_DIR/logs/%x-%j.out")
if [ "$KIND" = gpu ]; then ARGS+=(-p "$GPU_PARTITION" --gres "$GPU_GRES"); else ARGS+=(-p "$CPU_PARTITION"); fi
[ -n "${ACCOUNT:-}" ] && ARGS+=(-A "$ACCOUNT")
[ -n "${ARRAY:-}" ] && ARGS+=(--array "$ARRAY" -o "$PROJECT_DIR/logs/%x-%A_%a.out")
[ -n "${AFTER:-}" ] && ARGS+=(--dependency "afterok:$AFTER")
[ -n "${EXTRA_SBATCH:-}" ] && ARGS+=($EXTRA_SBATCH)

mkdir -p "$PROJECT_DIR/logs"
JID=$(sbatch --parsable "${ARGS[@]}" "$HERE/job.sh" "$STAGE" "$M")
echo "submitted $STAGE ($M) as job $JID  ->  log: logs/$STAGE-$M-$JID.out"
