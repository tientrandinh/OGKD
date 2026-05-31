#!/usr/bin/env bash
# Few-shot evaluation 
# Usage:
#   bash scripts/run_fewshot.sh                          # all datasets, seeds 1 2 3
#   bash scripts/run_fewshot.sh --datasets btmri busi    # a subset


set -u
cd "$(dirname "$0")/.."                              # run from the repository root
export PYTHONPATH="$PWD/Dassl.pytorch:$PWD"
export PYTHONWARNINGS=ignore

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset_raw}"
SHOTS="${SHOTS:-16}"
SEEDS="${SEEDS:-1 2 3}"

DATASETS="btmri busi chmnist covid ctkidney dermamnist kneexray kvasir lungcolon retina octmnist"
if [ "${1:-}" = "--datasets" ]; then shift; DATASETS="$*"; fi

stamp="$(date +%Y%m%d_%H%M%S)"
output="${OUTPUT_ROOT:-output}/fewshot/shots_$SHOTS/$stamp"   
results="results/fewshot/shots_$SHOTS/$stamp"                 
mkdir -p "$results/logs"
echo "few-shot | shots: $SHOTS | datasets: $DATASETS | seeds: $SEEDS"

for dataset in $DATASETS; do
    dataset_cfg="configs/datasets/$dataset.yaml"
    trainer_cfg="configs/trainers/OGKD/few_shot/$dataset.yaml"
    mkdir -p "$results/logs/$dataset"

    for seed in $SEEDS; do
        echo "[$dataset seed$seed] ${SHOTS}-shot train + test"
        $PYTHON train.py --root "$DATA_ROOT" --seed "$seed" --trainer OGKD \
            --dataset-config-file "$dataset_cfg" --config-file "$trainer_cfg" \
            --output-dir "$output/$dataset/seed$seed" \
            DATASET.NUM_SHOTS "$SHOTS" ${EXTRA:-} \
            > "$results/logs/$dataset/seed${seed}_train.log" 2>&1
    done

    # result summary
    {
        echo "=== $dataset ==="
        $PYTHON parse_test_res.py --directory "$output/$dataset" --test-log
        echo
    } >> "$results/detailed_results.txt" 2>&1

    [ "${KEEP_CKPT:-0}" = 1 ] || rm -rf "$output/$dataset"
done

echo "writing Excel summary"
$PYTHON scripts/results_to_excel.py --mode fewshot \
    --results-dir "$results" --datasets $DATASETS --seeds $SEEDS \
    --shots "$SHOTS" --out "$results/results_fewshot_${SHOTS}shot.xlsx"
echo "done -> $results"
