#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/Dassl.pytorch:$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"
PYTHON="${PYTHON:-python}"

declare -a DATASETS_CLI=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) shift; while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS_CLI+=("$1"); shift; done ;;
    *) shift ;;
  esac
done
DATASETS=(btmri chmnist covid ctkidney dermamnist kneexray kvasir lungcolon retina octmnist)
if (( ${#DATASETS_CLI[@]} > 0 )); then DATASETS=("${DATASETS_CLI[@]}"); fi

TRAINER="OGKD"
SHOTS=${SHOTS:-16}
LOAD_EPOCH=${LOAD_EPOCH:-50}
MAX_WORKERS=${MAX_WORKERS:-1}   # seeds run sequentially by default; raise (e.g. 3) only on a large GPU
read -r -a SEEDS <<< "${SEEDS:-1 2 3}"

SUB_BASE="base"; SUB_NOVEL="new"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset_raw}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_ROOT:-$REPO_ROOT/output}/base2novel/${STAMP}"
RESULTS_DIR="${RESULTS_ROOT:-$REPO_ROOT/results}/base2novel/${STAMP}"
mkdir -p "$RESULTS_DIR/logs"

# Optional quick-test override: set MAX_EPOCH=2 for a fast smoke run.
EXTRA_OPTS=()
[[ -n "${MAX_EPOCH:-}" ]] && EXTRA_OPTS+=(OPTIM.MAX_EPOCH "$MAX_EPOCH")

run_seed_job() {
  local dataset="$1" seed="$2"
  local dataset_config="$REPO_ROOT/configs/datasets/${dataset}.yaml"
  local config_file="$REPO_ROOT/configs/trainers/OGKD/base_to_novel/${dataset}.yaml"
  local base_dir="${OUTPUT_BASE}/train_${SUB_BASE}/${dataset}/shots_${SHOTS}/${TRAINER}/seed${seed}"
  local novel_dir="${OUTPUT_BASE}/test_${SUB_NOVEL}/${dataset}/shots_${SHOTS}/${TRAINER}/seed${seed}"
  mkdir -p "$RESULTS_DIR/logs/${dataset}"
  local train_log="$RESULTS_DIR/logs/${dataset}/seed${seed}_train.log"
  local eval_log="$RESULTS_DIR/logs/${dataset}/seed${seed}_novel_eval.log"

  echo "🔄 [${dataset}] train base (seed ${seed})"
  "$PYTHON" "$REPO_ROOT/train.py" \
    --root "$DATA_ROOT" --seed "$seed" --trainer "$TRAINER" \
    --dataset-config-file "$dataset_config" --config-file "$config_file" \
    --output-dir "$base_dir" \
    "${EXTRA_OPTS[@]}" \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES "$SUB_BASE" \
    >"$train_log" 2>&1 || { echo "❌ [$dataset] base train failed (seed $seed); see $train_log"; return 1; }

  echo "🔍 [${dataset}] eval novel (seed ${seed})"
  "$PYTHON" "$REPO_ROOT/train.py" \
    --root "$DATA_ROOT" --seed "$seed" --trainer "$TRAINER" \
    --dataset-config-file "$dataset_config" --config-file "$config_file" \
    --output-dir "$novel_dir" --model-dir "$base_dir" --load-epoch "$LOAD_EPOCH" --eval-only \
    "${EXTRA_OPTS[@]}" \
    DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES "$SUB_NOVEL" \
    >"$eval_log" 2>&1 || { echo "[$dataset] novel eval failed (seed $seed); see $eval_log"; return 1; }
  echo "[${dataset}] done (seed ${seed})"
}

run_with_limit() {
  local -n pids_ref=$1
  while true; do
    local tmp=()
    for pid in "${pids_ref[@]}"; do kill -0 "$pid" 2>/dev/null && tmp+=("$pid"); done
    pids_ref=("${tmp[@]}")
    (( ${#pids_ref[@]} < MAX_WORKERS )) && break
    sleep 2
  done
}

echo "📊 base2novel | datasets: ${DATASETS[*]} | shots: $SHOTS | seeds: ${SEEDS[*]}"
echo "   results: $RESULTS_DIR"
for dataset in "${DATASETS[@]}"; do
  dc="$REPO_ROOT/configs/datasets/${dataset}.yaml"
  cf="$REPO_ROOT/configs/trainers/OGKD/base_to_novel/${dataset}.yaml"
  [[ -f "$dc" && -f "$cf" ]] || { echo "❌ missing config(s) for $dataset"; continue; }
  pids=()
  for seed in "${SEEDS[@]}"; do run_with_limit pids; run_seed_job "$dataset" "$seed" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  {
    echo "=== $dataset ==="
    echo "Base class results:"
    "$PYTHON" "$REPO_ROOT/parse_test_res.py" --directory "${OUTPUT_BASE}/train_${SUB_BASE}/${dataset}/shots_${SHOTS}/${TRAINER}" --test-log
    echo; echo "Novel class results:"
    "$PYTHON" "$REPO_ROOT/parse_test_res.py" --directory "${OUTPUT_BASE}/test_${SUB_NOVEL}/${dataset}/shots_${SHOTS}/${TRAINER}" --test-log
    echo
  } >> "$RESULTS_DIR/detailed_results.txt" 2>&1

  # Free disk: drop this dataset's checkpoints once its results are parsed to save space 
  if [[ "${KEEP_CKPT:-0}" != "1" ]]; then
    rm -rf "${OUTPUT_BASE}/train_${SUB_BASE}/${dataset}" "${OUTPUT_BASE}/test_${SUB_NOVEL}/${dataset}"
  fi
done

echo "writing Excel summary"
"$PYTHON" "$REPO_ROOT/scripts/results_to_excel.py" --mode base2novel \
  --results-dir "$RESULTS_DIR" --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" \
  --shots "$SHOTS" --out "$RESULTS_DIR/results_base2novel.xlsx" || echo "Excel step failed"

echo "Done. Results: $RESULTS_DIR"
