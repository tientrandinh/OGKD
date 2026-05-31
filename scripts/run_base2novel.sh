#!/usr/bin/env bash
# Base-to-novel generalization
# For each dataset and seed: train on the base classes (16-shot), then evaluate
# on the held-out novel classes. 
#
# Usage:
#   bash scripts/run_base2novel.sh                          # all datasets, seeds 1 2 3
#   bash scripts/run_base2novel.sh --datasets btmri covid   # a subset



set -u
cd "$(dirname "$0")/.."                              # run from the repository root
export PYTHONPATH="$PWD/Dassl.pytorch:$PWD"
export PYTHONWARNINGS=ignore

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset_raw}"
SHOTS="${SHOTS:-16}"
SEEDS="${SEEDS:-1 2 3}"

DATASETS="btmri chmnist covid ctkidney dermamnist kneexray kvasir lungcolon retina octmnist"
if [ "${1:-}" = "--datasets" ]; then shift; DATASETS="$*"; fi

stamp="$(date +%Y%m%d_%H%M%S)"
output="${OUTPUT_ROOT:-output}/base2novel/$stamp"   
results="results/base2novel/$stamp"                 
mkdir -p "$results/logs"
echo "base-to-novel | datasets: $DATASETS | seeds: $SEEDS | shots: $SHOTS"

for dataset in $DATASETS; do
    dataset_cfg="configs/datasets/$dataset.yaml"
    trainer_cfg="configs/trainers/OGKD/base_to_novel/$dataset.yaml"
    mkdir -p "$results/logs/$dataset"

    for seed in $SEEDS; do
        base_dir="$output/base/$dataset/seed$seed"
        novel_dir="$output/novel/$dataset/seed$seed"

        echo "[$dataset seed$seed] train on base classes"
        $PYTHON train.py --root "$DATA_ROOT" --seed "$seed" --trainer OGKD \
            --dataset-config-file "$dataset_cfg" --config-file "$trainer_cfg" \
            --output-dir "$base_dir" \
            DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES base ${EXTRA:-} \
            > "$results/logs/$dataset/seed${seed}_train.log" 2>&1

        echo "[$dataset seed$seed] evaluate on novel classes"
        $PYTHON train.py --root "$DATA_ROOT" --seed "$seed" --trainer OGKD \
            --dataset-config-file "$dataset_cfg" --config-file "$trainer_cfg" \
            --output-dir "$novel_dir" --model-dir "$base_dir" \
            --load-epoch "${LOAD_EPOCH:-50}" --eval-only \
            DATASET.NUM_SHOTS "$SHOTS" DATASET.SUBSAMPLE_CLASSES new ${EXTRA:-} \
            > "$results/logs/$dataset/seed${seed}_novel_eval.log" 2>&1
    done

    # result summary
    {
        echo "=== $dataset ==="
        echo "base classes:";  $PYTHON parse_test_res.py --directory "$output/base/$dataset"  --test-log
        echo "novel classes:"; $PYTHON parse_test_res.py --directory "$output/novel/$dataset" --test-log
        echo
    } >> "$results/detailed_results.txt" 2>&1

    [ "${KEEP_CKPT:-0}" = 1 ] || rm -rf "$output/base/$dataset" "$output/novel/$dataset"
done

echo "writing Excel summary"
$PYTHON scripts/results_to_excel.py --mode base2novel \
    --results-dir "$results" --datasets $DATASETS --seeds $SEEDS \
    --shots "$SHOTS" --out "$results/results_base2novel.xlsx"
echo "done -> $results"
