#!/bin/bash

DATASETS=(flchain)
LAMBDAS=(0.0 0.1 0.3 1.0 3.0 5.0 10.0 30.0 50.0 100.0 300.0)
DEPTHS=(1)

NUM_BINS=100

for DATA in "${DATASETS[@]}"; do

  echo "################################################"
  echo "   Running dataset: $DATA"
  echo "################################################"

  for D in "${DEPTHS[@]}"; do
    echo "========================================"
    echo "   DEPTH = $D"
    echo "========================================"

    for L in "${LAMBDAS[@]}"; do
      echo "----------------------------------------"
      echo "  URW  dataset=$DATA lambda=$L depth=$D"
      echo "----------------------------------------"

      python run_survival_inproc_urw.py \
        --dataset "$DATA" \
        --seed 10 \
        --num_iterations 4000 \
        --nloglr 3 \
        --lambda_cons "$L" \
        --num_bins "${NUM_BINS}" \
        --theta 1e-3 \
        --tau 5.0 \
        --max_graph_n 5000 \
        --svd_rank 1 \
        --use_metric \
        --metric_rank 64 \
        --depth "$D" \
        --out_root results/urw \
        --save_if

      echo "----------------------------------------"
      echo "  LAP  dataset=$DATA lambda=$L depth=$D"
      echo "----------------------------------------"

      python run_survival_inproc_laplacian.py \
        --dataset "$DATA" \
        --seed 10 \
        --num_iterations 4000 \
        --nloglr 3 \
        --lambda_cons "$L" \
        --num_bins "${NUM_BINS}" \
        --theta 1e-3 \
        --tau 5.0 \
        --max_graph_n 5000 \
        --svd_rank 1 \
        --use_metric \
        --metric_rank 64 \
        --depth "$D" \
        --out_root results/lap \
        --save_if

    done
  done
done
