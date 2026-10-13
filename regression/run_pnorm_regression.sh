#!/bin/bash
# run_pnorm_reg_urw_aligned_lambda.sh
set -e

OUTDIR="results_pnorm_new"
mkdir -p "$OUTDIR"

SEED=0
DEPTH=1
ITERS=4000
NLOGLR=3

THETA=1e-3
TAU=5.0
MAX_GRAPH_N=5000

USE_METRIC="--use_metric"
METRIC_RANK=32

# -----------------------
# Run all p (ROOT p-norm in python)
# -----------------------
RUN_ONLY_INF=0   # 0 = run all (1,4,inf); 1 = only inf

# aligned lambda grid
LAMBDAS=(0 0.001 0.003 0.01 0.03 0.1 0.3 1 3)

PNORMS=(1 4 inf)
if [[ "$RUN_ONLY_INF" == "1" ]]; then
  PNORMS=(inf)
fi

# -----------------------
# Tail-sensitive p-norm experiment
# -----------------------

# Use the same residual treatment for all p.
# 0 disables residual clipping.
RCLIP=0.0

# Smooth-max parameter for p=inf
INF_ALPHA_MAX=200

# Gradient clipping is optimizer stabilization only
GRAD_CLIP=0.5

echo "==========================================="
echo " Regression URW p-norm runs (ROOT p-norm + p-gain)"
echo " OUTDIR: $OUTDIR"
echo " Lambdas: ${LAMBDAS[*]}"
echo " p: ${PNORMS[*]}"
echo " grad_clip: $GRAD_CLIP"
echo " r_clip finite: $RCLIP_FINITE | r_clip inf: $RCLIP_INF"
echo " p_gain_pow: $P_GAIN_POW | inf_gain: $INF_GAIN"
echo "==========================================="

get_rclip () {
  local P="$1"
  if [[ "$P" == "inf" ]]; then
    echo "$RCLIP_INF"
  else
    echo "$RCLIP_FINITE"
  fi
}

get_inf_extra_args () {
  local args=""
  if [[ "$INF_USE_LOGSUMEXP" == "1" ]]; then
    args="--inf_use_logsumexp --alpha_max ${INF_ALPHA_MAX}"
  else
    args="--inf_topk_frac ${INF_TOPK}"
  fi
  echo "$args"
}

run_one () {
  local P="$1"
  local L="$2"
  local tagP="$3"

  echo "-------------------------------------------"
  echo "  Running: p=$P  lambda=$L"
  echo "-------------------------------------------"

  python ./run_regression_p_norm_urw.py \
    --seed "$SEED" \
    --depth "$DEPTH" \
    --num_iterations "$ITERS" \
    --nloglr "$NLOGLR" \
    --p_norm "$P" \
    --lambda_cons "$L" \
    --theta "$THETA" \
    --tau "$TAU" \
    --max_graph_n "$MAX_GRAPH_N" \
    $USE_METRIC \
    --metric_rank "$METRIC_RANK" \
    --r_clip "$RCLIP" \
    --grad_clip "$GRAD_CLIP" \
    --alpha_max "$INF_ALPHA_MAX" \
    --save_if_csv "${OUTDIR}/urw_p${tagP}_lambda${L}.csv"
}

for P in "${PNORMS[@]}"; do
  for L in "${LAMBDAS[@]}"; do
    run_one "$P" "$L" "$P"
  done
done

echo "==========================================="
echo " All runs finished. Results in $OUTDIR/"
echo "==========================================="
