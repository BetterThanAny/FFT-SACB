#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Base training launcher for SACB-Net (single run, long training).

Usage:
  bash scripts/run_base.sh [options]

Options:
  --dataset <ixi|lpba|abd>         Dataset name. Default: ixi
  --base-dir <path>                Dataset root. Required.
  --gpu <int>                      GPU id. Default: 0
  --max-epoch <int>                Max epochs. Default: 300
  --batch-size <int>               Batch size. Default: 1
  --num-workers <int>              DataLoader workers. Default: 8
  --lp-ratio <value>               Low-pass ratio. Default: 0.15
  --weights <a,b>                  Loss weights. Default: 1,0.3
  --lr <float>                     Learning rate. Default: 1e-4
  --seed <int>                     Random seed. Default: 0
  --save-tag <str>                 Optional save tag. Default: auto-generated
  --python <bin>                   Python executable. Default: python
  --cuda-deterministic             Pass through to train.py
  --cont-training                  Continue training from checkpoint
  --epoch-start <int>              Epoch start when resuming. Default: 0
  --resume-epoch <int>             Fallback resume epoch. Default: 201
  --resume-path <path>             Optional explicit checkpoint path
  --dry-run                        Print command only
  -h, --help                       Show this help

Examples:
  bash scripts/run_base.sh --dataset ixi --base-dir /data --save-tag ixi_base_lp015

  bash scripts/run_base.sh \
    --dataset ixi \
    --base-dir /data \
    --cont-training \
    --epoch-start 120 \
    --resume-path /path/to/model.pth.tar \
    --save-tag ixi_base_lp015
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="ixi"
BASE_DIR=""
GPU=0
MAX_EPOCH=300
BATCH_SIZE=1
NUM_WORKERS=8
LP_RATIO="0.15"
WEIGHTS="1,0.3"
LR="1e-4"
SEED=0
SAVE_TAG=""
PYTHON_BIN="python"
CUDA_DETERMINISTIC=0
CONT_TRAINING=0
EPOCH_START=0
RESUME_EPOCH=201
RESUME_PATH=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --max-epoch)
      MAX_EPOCH="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --lp-ratio)
      LP_RATIO="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --save-tag)
      SAVE_TAG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --cuda-deterministic)
      CUDA_DETERMINISTIC=1
      shift
      ;;
    --cont-training)
      CONT_TRAINING=1
      shift
      ;;
    --epoch-start)
      EPOCH_START="$2"
      shift 2
      ;;
    --resume-epoch)
      RESUME_EPOCH="$2"
      shift 2
      ;;
    --resume-path)
      RESUME_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${DATASET}" in
  ixi|lpba|abd) ;;
  *)
    echo "ERROR: --dataset must be one of: ixi, lpba, abd" >&2
    exit 1
    ;;
esac

if [[ -z "${BASE_DIR}" ]]; then
  echo "ERROR: --base-dir is required." >&2
  usage
  exit 1
fi

if [[ -z "${OMP_NUM_THREADS:-}" ]] || [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ -z "${SAVE_TAG}" ]]; then
  SAVE_TAG="base_${DATASET}_$(date +%Y%m%d_%H%M%S)"
fi

cmd=(
  "${PYTHON_BIN}" train.py
  --dataset "${DATASET}"
  --base-dir "${BASE_DIR}"
  --lp-ratio "${LP_RATIO}"
  --weights "${WEIGHTS}"
  --lr "${LR}"
  --max-epoch "${MAX_EPOCH}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --gpu "${GPU}"
  --seed "${SEED}"
  --save-tag "${SAVE_TAG}"
  --epoch-start "${EPOCH_START}"
  --resume-epoch "${RESUME_EPOCH}"
)

if [[ "${CUDA_DETERMINISTIC}" -eq 1 ]]; then
  cmd+=(--cuda-deterministic)
fi

if [[ "${CONT_TRAINING}" -eq 1 ]]; then
  cmd+=(--cont-training)
  if [[ -n "${RESUME_PATH}" ]]; then
    cmd+=(--resume-path "${RESUME_PATH}")
  fi
fi

printf 'CMD: '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

"${cmd[@]}"
