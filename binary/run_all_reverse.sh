#!/bin/bash

LAMBDAS=(0.0 0.1 0.3 0.5 1 3 5 10 30 50 100 300 500)
DEPTHS=(1)

for D in "${DEPTHS[@]}"; do
    echo "========================================"
    echo "   Running experiments for DEPTH = $D   "
    echo "========================================"

    for L in "${LAMBDAS[@]}"; do
        echo "----------------------------------------"
        echo "  Running URW λ = $L , depth = $D"
        echo "----------------------------------------"

        python run_binary_inproc_lap.py \
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
            --save_if_csv results_unified/lap_depth${D}_lambda${L}.csv
    done
done
