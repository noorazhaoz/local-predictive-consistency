##!/usr/bin/env bash
#set -e
#
#SCRIPT=run_survival_inproc_deepsurv_laplacian.py
#
## -------------------------
## Fixed hyperparameters
## -------------------------
#DEPTH=1
#SEED=10
#NUM_ITERS=4000
#NLOGLR=3.0
#
#THETA=1e-3
#TAU=5.0
#MAX_GRAPH_N=5000
#SVD_RANK=1
#
## If you don't use metric learning, keep this OFF
#USE_METRIC=0
#METRIC_RANK=64
#
#TEST_SIZE=0.2
#IBS_NUM_TIMES=20
#
## -------------------------
## Sweep lists
## -------------------------
#DATASETS=("flchain")
#
## example sweep (edit as you like)
#LAMBDAS=(0 1 3 5 10 30 50 100 300)
##LAMBDAS=(50)
## -------------------------
## Output
## -------------------------
#OUT_ROOT="results/lap"
#
## -------------------------
## Run
## -------------------------
#for DATASET in "${DATASETS[@]}"; do
#  for LAMBDA in "${LAMBDAS[@]}"; do
#
#    echo "================================================"
#    echo "Dataset=${DATASET} | lambda_cons=${LAMBDA}"
#    echo "================================================"
#
#    CMD=(python3 ${SCRIPT}
#      --dataset ${DATASET}
#      --depth ${DEPTH}
#      --seed ${SEED}
#      --test_size ${TEST_SIZE}
#      --num_iterations ${NUM_ITERS}
#      --nloglr ${NLOGLR}
#      --lambda_cons ${LAMBDA}
#      --theta ${THETA}
#      --tau ${TAU}
#      --max_graph_n ${MAX_GRAPH_N}
#      --svd_rank ${SVD_RANK}
#      --ibs_num_times ${IBS_NUM_TIMES}
#      --out_root ${OUT_ROOT}
#      --save_if
#    )
#
#    if [[ "${USE_METRIC}" -eq 1 ]]; then
#      CMD+=(--use_metric --metric_rank ${METRIC_RANK})
#    fi
#
#    "${CMD[@]}"
#
#  done
#done
#
#echo "✅ Done"

set -euo pipefail

# -------------------------
# Scripts
# -------------------------
SCRIPT_LAP=run_survival_inproc_deepsurv_laplacian.py
SCRIPT_URW=run_survival_inproc_deepsurv_urw.py

# -------------------------
# Fixed hyperparameters
# -------------------------
DEPTH=1
SEED=10
NUM_ITERS=4000
NLOGLR=3.0

THETA=1e-3
TAU=5.0
MAX_GRAPH_N=5000
SVD_RANK=1

# If you don't use metric learning, keep this OFF
USE_METRIC=0
METRIC_RANK=64

TEST_SIZE=0.2
IBS_NUM_TIMES=20

# -------------------------
# Sweep lists
# -------------------------
DATASETS=("flchain")
LAMBDAS=(200)

# -------------------------
# Output
# -------------------------
OUT_LAP="results/lap"
OUT_URW="results/urw"

# -------------------------
# Helper to run a script
# -------------------------
run_one () {
  local script="$1"
  local dataset="$2"
  local lambda="$3"
  local out_root="$4"
  local tag="$5"

  echo "================================================"
  echo "[${tag}] Dataset=${dataset} | lambda_cons=${lambda} | out_root=${out_root}"
  echo "================================================"

  CMD=(python3 "${script}"
    --dataset "${dataset}"
    --depth "${DEPTH}"
    --seed "${SEED}"
    --test_size "${TEST_SIZE}"
    --num_iterations "${NUM_ITERS}"
    --nloglr "${NLOGLR}"
    --lambda_cons "${lambda}"
    --theta "${THETA}"
    --tau "${TAU}"
    --max_graph_n "${MAX_GRAPH_N}"
    --svd_rank "${SVD_RANK}"
    --ibs_num_times "${IBS_NUM_TIMES}"
    --out_root "${out_root}"
    --save_if
  )

  if [[ "${USE_METRIC}" -eq 1 ]]; then
    CMD+=(--use_metric --metric_rank "${METRIC_RANK}")
  fi

  "${CMD[@]}"
}

# -------------------------
# Run sweep
# -------------------------
for DATASET in "${DATASETS[@]}"; do
  for LAMBDA in "${LAMBDAS[@]}"; do
    run_one "${SCRIPT_LAP}" "${DATASET}" "${LAMBDA}" "${OUT_LAP}" "LAPLACIAN"
    run_one "${SCRIPT_URW}" "${DATASET}" "${LAMBDA}" "${OUT_URW}" "URW"
  done
done

echo "✅ Done (Laplacian + URW)"