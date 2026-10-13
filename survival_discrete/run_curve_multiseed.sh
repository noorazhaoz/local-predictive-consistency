#!/bin/bash
set -e

PYTHON_SCRIPT="./run_survival_inproc_urw.py"

OUTROOT="results_multiseed_curve"

DATASETS=("whas" "rossi")
SEEDS=(0 1 2 42 123)
LAMBDAS=(0 10 300)

DEPTH=1
ITERS=4000
NLOGLR=3

THETA=1e-3
TAU=5.0
MAX_GRAPH_N=5000

NUM_BINS=100
IBS_NUM_TIMES=20
IBS_Q_HIGH=0.95

# Keep aligned with the original experiments
USE_METRIC="--use_metric"
METRIC_RANK=64

echo "============================================"
echo " Multi-seed discrete survival robustness"
echo " Datasets: ${DATASETS[*]}"
echo " Seeds:    ${SEEDS[*]}"
echo " Lambdas:  ${LAMBDAS[*]}"
echo "============================================"

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        # Separate output directory for every seed,
        # because the Python script does not include seed in filenames.
        SEED_OUT="${OUTROOT}/seed${SEED}"

        for LAMBDA in "${LAMBDAS[@]}"; do
            echo "--------------------------------------------"
            echo " dataset=${DATASET}"
            echo " seed=${SEED}"
            echo " lambda=${LAMBDA}"
            echo "--------------------------------------------"

            python "$PYTHON_SCRIPT" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --depth "$DEPTH" \
                --num_iterations "$ITERS" \
                --nloglr "$NLOGLR" \
                --lambda_cons "$LAMBDA" \
                --theta "$THETA" \
                --tau "$TAU" \
                --max_graph_n "$MAX_GRAPH_N" \
                --num_bins "$NUM_BINS" \
                --ibs_num_times "$IBS_NUM_TIMES" \
                --ibs_q_high "$IBS_Q_HIGH" \
                $USE_METRIC \
                --metric_rank "$METRIC_RANK" \
                --out_root "$SEED_OUT" \
                --save_if

        done
    done
done

echo "============================================"
echo " All discrete multi-seed runs finished."
echo " Results: ${OUTROOT}/"
echo "============================================"