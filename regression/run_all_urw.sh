#!/bin/bash

LAMBDAS=(0.0 0.1 0.3 0.5 1.0 3.0 5.0 10.0 30.0 50.0 100.0 300.0 500.0)
DEPTHS=(1)

DATA_PATH="../data/insurance.csv"

for D in "${DEPTHS[@]}"; do
    echo "========================================"
    echo "   Running experiments for DEPTH = $D   "
    echo "========================================"

    for L in "${LAMBDAS[@]}"; do
        echo "----------------------------------------"
        echo "  Running URW λ = $L , depth = $D"
        echo "----------------------------------------"

        python run_regression_inproc_urw.py \
            --dataset insurance \
            --data_path "$DATA_PATH" \
            --seed 0 \
            --num_iterations 4000 \
            --nloglr 3 \
            --lambda_cons $L \
            --theta 1e-3 \
            --tau 5.0 \
            --max_graph_n 5000 \
            --svd_rank 1 \
            --use_metric \
            --metric_rank 64 \
            --depth $D \
            --save_if_csv results/insurance_urw_depth${D}_lambda${L}.csv
    done
done