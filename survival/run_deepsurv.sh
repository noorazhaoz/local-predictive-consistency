#!/usr/bin/env bash
set -e

SCRIPT=run_survival_inproc_deepsurv_urw.py

# -------------------------
# Fixed hyperparameters
# -------------------------
DEPTH=1
SEED=0
NUM_ITERS=4000
NLOGLR=3.0

THETA=1e-3
TAU=5.0
MAX_GRAPH_N=5000
SVD_RANK=1

# -------------------------
# Sweep lists
# -------------------------
DATASETS=("employee" "whas" "rossi" "flchain")


LAMBDAS=(1e-6 1e-5 1e-4 1e-3 1e-2)

# -------------------------
# Run
# -------------------------
for DATASET in "${DATASETS[@]}"; do
  for LAMBDA in "${LAMBDAS[@]}"; do

    echo "================================================"
    echo "Dataset=${DATASET} | lambda_cons=${LAMBDA}"
    echo "================================================"

    python3 ${SCRIPT} \
      --dataset ${DATASET} \
      --depth ${DEPTH} \
      --seed ${SEED} \
      --num_iterations ${NUM_ITERS} \
      --nloglr ${NLOGLR} \
      --lambda_cons ${LAMBDA} \
      --theta ${THETA} \
      --tau ${TAU} \
      --max_graph_n ${MAX_GRAPH_N} \
      --svd_rank ${SVD_RANK} \
      --save_if_csv results/urw/${DATASET}/if_deepsurv_depth${DEPTH}_lambda${LAMBDA}.csv

  done
done

echo "✅ All datasets × lambdas finished."
