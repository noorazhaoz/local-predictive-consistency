#!/usr/bin/env bash
set -euo pipefail

PNORMS=(1 2 4 inf)
LAMBDAS=(0.0 0.1 0.3 0.5 1 3 5 10 30)
DEPTHS=(1)

ALPHA_WORST=200
OUTDIR="results_pnorm"
mkdir -p "$OUTDIR"

echo "==========================================="
echo " Running Binary URW (Adult) with p-norm fairness"
echo "==========================================="

for P in "${PNORMS[@]}"; do
  for L in "${LAMBDAS[@]}"; do
    for D in "${DEPTHS[@]}"; do

      echo "-------------------------------------------"
      echo "  Running: p=$P, lambda=$L, depth=$D"
      echo "-------------------------------------------"

      python run_binary_pnorm_urw_adult.py \
        --seed 0 \
        --depth "$D" \
        --p_norm "$P" \
        --lambda_cons "$L" \
        --alpha_worst "$ALPHA_WORST" \
        --num_iterations 4000 \
        --nloglr 3 \
        --theta 1e-3 \
        --tau 5.0 \
        --max_graph_n 5000 \
        --use_metric \
        --metric_rank 32 \
        --save_if_csv "${OUTDIR}/if_adult_p${P}_lambda${L}.csv"

    done
  done
done

echo "==========================================="
echo " All runs finished. Results in $OUTDIR/"
echo "==========================================="
