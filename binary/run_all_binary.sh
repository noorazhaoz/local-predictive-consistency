#!/bin/bash
# Sweep lambda for Standard Laplacian on Adult + COMPAS

set -e

LAMBDAS=(0.0 0.1 0.3 0.5 1 3 5 10 30 50 100 300 500)
DEPTHS=(1)

DATASETS=("adult" "compas")

DATA_DIR="../data"
SEED=0

mkdir -p results_unified

for DATASET in "${DATASETS[@]}"; do
  for D in "${DEPTHS[@]}"; do
    echo "========================================"
    echo " DATASET=${DATASET} | DEPTH=${D}"
    echo "========================================"

    for L in "${LAMBDAS[@]}"; do
      echo "----------------------------------------"
      echo " Running: dataset=${DATASET} depth=${D} lambda=${L}"
      echo "----------------------------------------"

      python run_binary_inproc_lap.py \
        --dataset "${DATASET}" \
        --data_dir "${DATA_DIR}" \
        --seed "${SEED}" \
        --num_iterations 4000 \
        --nloglr 3 \
        --lambda_cons "${L}" \
        --theta 1e-3 \
        --tau 5.0 \
        --max_graph_n 5000 \
        --svd_rank 1 \
        --use_metric \
        --metric_rank 64 \
        --depth "${D}" \
        --save_if_csv "results_unified/if_${DATASET}_depth${D}_lambda${L}.csv"

    done
  done
done