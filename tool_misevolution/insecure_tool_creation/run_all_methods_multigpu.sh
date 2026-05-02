#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OPENAI_API_URL="${OPENAI_API_URL:-http://127.0.0.1:11434/api/chat}"
export MODEL="${MODEL:-deepseek-coder:6.7b}"
export API_KEY="${API_KEY:-ollama}"
export EXPERIMENT_DATASET="${EXPERIMENT_DATASET:-misevolution_user_queries}"
export EXPERIMENT_OUTPUT_ROOT="${EXPERIMENT_OUTPUT_ROOT:-/home/nvidia/Sheldon/agent_defense/mini-swe-agent/results}"

GPU_OURS_FULL="${GPU_OURS_FULL:-2}"
GPU_OURS_STRONG="${GPU_OURS_STRONG:-3}"
GPU_DIRECT="${GPU_DIRECT:-4}"
GPU_FIXED="${GPU_FIXED:-5}"
GPU_RANDOM="${GPU_RANDOM:-6,7}"

RANDOM_SEEDS="${RANDOM_SEEDS:-11,22,33,44,55}"
BASE_SEED="${BASE_SEED:-42}"
MAX_TASKS="${MAX_TASKS:-0}"
GROUPS="${GROUPS:-direct_trigger,random_sequence,fixed_simple}"
RANDOM_START_MODE="${RANDOM_START_MODE:-wait_ours}"

CLEAN="${CLEAN:-1}"
FULL_OUTPUT="${FULL_OUTPUT:-0}"
WRITE_MD="${WRITE_MD:-0}"
SKIP_OURS_FULL="${SKIP_OURS_FULL:-0}"
SKIP_OURS_STRONG="${SKIP_OURS_STRONG:-0}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
SKIP_RANDOM_MULTI_SEED="${SKIP_RANDOM_MULTI_SEED:-0}"

CMD=(
  python run_one_click.py
  --dataset "$EXPERIMENT_DATASET"
  --output-root "$EXPERIMENT_OUTPUT_ROOT"
  --seed "$BASE_SEED"
  --max-tasks "$MAX_TASKS"
  --random-seeds "$RANDOM_SEEDS"
  --groups "$GROUPS"
  --gpu-ours-full "$GPU_OURS_FULL"
  --gpu-ours-strong "$GPU_OURS_STRONG"
  --gpu-direct "$GPU_DIRECT"
  --gpu-fixed "$GPU_FIXED"
  --gpu-random "$GPU_RANDOM"
  --random-start-mode "$RANDOM_START_MODE"
)

if [[ "$CLEAN" == "1" ]]; then
  CMD+=(--clean)
fi
if [[ "$FULL_OUTPUT" == "1" ]]; then
  CMD+=(--full-output)
fi
if [[ "$WRITE_MD" == "1" ]]; then
  CMD+=(--write-md)
fi
if [[ "$SKIP_OURS_FULL" == "1" ]]; then
  CMD+=(--skip-ours-full)
fi
if [[ "$SKIP_OURS_STRONG" == "1" ]]; then
  CMD+=(--skip-ours-strong)
fi
if [[ "$SKIP_BASELINES" == "1" ]]; then
  CMD+=(--skip-baselines)
fi
if [[ "$SKIP_RANDOM_MULTI_SEED" == "1" ]]; then
  CMD+=(--skip-random-multiseed)
fi

echo "dataset=$EXPERIMENT_DATASET"
echo "output_root=$EXPERIMENT_OUTPUT_ROOT"
echo "gpu_ours_full=$GPU_OURS_FULL"
echo "gpu_ours_strong=$GPU_OURS_STRONG"
echo "gpu_direct=$GPU_DIRECT"
echo "gpu_fixed=$GPU_FIXED"
echo "gpu_random=$GPU_RANDOM"
echo "random_start_mode=$RANDOM_START_MODE"
echo "cmd=${CMD[*]}"

exec "${CMD[@]}"
