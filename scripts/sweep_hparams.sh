#!/usr/bin/env bash

set -u
set -o pipefail

usage() {
  cat <<'EOF'
Sweep SACB-Net hyperparameters by launching train.py in a grid.

Usage:
  bash scripts/sweep_hparams.sh --base-dir <DATA_ROOT> [options]

Options:
  --dataset <ixi|lpba|abd>                 Dataset name. Default: ixi
  --base-dir <path>                        Dataset root. Required.
  --gpu <int>                              GPU id. Default: 0
  --max-epoch <int>                        Max epochs per run. Default: 100
  --batch-size <int>                       Batch size. Default: 1
  --num-workers <int>                      DataLoader workers. Default: 8
  --lp-ratios "<a;b;c;d>"                  Semicolon-separated lp ratios.
                                           Each item can be "0.15" or "0.12,0.14,0.16,0.18"
  --weights-list "<a;b;c>"                 Semicolon-separated weights, e.g. "1,0.2;1,0.3"
  --lrs "<a;b;c>"                          Semicolon-separated learning rates.
  --seeds "<a;b;c>"                        Semicolon-separated random seeds.
  --python <bin>                           Python executable. Default: python
  --cuda-deterministic                     Pass through to train.py
  --dry-run                                Print commands only, do not run training
  -h, --help                               Show this help

Note:
  Default grid = 5 lp_ratios x 3 weights x 2 lrs x 3 seeds = 90 runs.
  Use --dry-run to preview all commands before launching.
  Consider narrowing the grid or reducing --max-epoch for initial exploration.

Examples:
  bash scripts/sweep_hparams.sh \
    --dataset ixi \
    --base-dir /data \
    --gpu 0 \
    --max-epoch 120 \
    --lp-ratios "0.10;0.12;0.15;0.18;0.12,0.14,0.16,0.18" \
    --weights-list "1,0.2;1,0.3;1,0.5" \
    --lrs "1e-4;5e-5" \
    --seeds "0;1"
EOF
}

sanitize_tag() {
  echo "$1" | sed 's/,/c/g; s/\./d/g; s/-/m/g; s/+/p/g; s/[^[:alnum:]_]/_/g'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="ixi"
BASE_DIR=""
GPU=0
MAX_EPOCH=100
BATCH_SIZE=1
NUM_WORKERS=8
LP_RATIOS_STR="0.10;0.12;0.15;0.18;0.12,0.14,0.16,0.18"
WEIGHTS_STR="1,0.2;1,0.3;1,0.5"
LRS_STR="1e-4;5e-5"
SEEDS_STR="0;1;2"
PYTHON_BIN="python"
CUDA_DETERMINISTIC=0
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
    --lp-ratios)
      LP_RATIOS_STR="$2"
      shift 2
      ;;
    --weights-list)
      WEIGHTS_STR="$2"
      shift 2
      ;;
    --lrs)
      LRS_STR="$2"
      shift 2
      ;;
    --seeds)
      SEEDS_STR="$2"
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

if [[ -z "${BASE_DIR}" ]]; then
  echo "ERROR: --base-dir is required." >&2
  usage
  exit 1
fi

case "${DATASET}" in
  ixi|lpba|abd) ;;
  *)
    echo "ERROR: --dataset must be one of: ixi, lpba, abd" >&2
    exit 1
    ;;
esac

IFS=';' read -r -a LP_RATIOS <<< "${LP_RATIOS_STR}"
IFS=';' read -r -a WEIGHTS_LIST <<< "${WEIGHTS_STR}"
IFS=';' read -r -a LR_LIST <<< "${LRS_STR}"
IFS=';' read -r -a SEED_LIST <<< "${SEEDS_STR}"

if [[ "${#LP_RATIOS[@]}" -eq 0 || "${#WEIGHTS_LIST[@]}" -eq 0 || "${#LR_LIST[@]}" -eq 0 || "${#SEED_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: sweep lists must be non-empty." >&2
  exit 1
fi

TOTAL=$(( ${#LP_RATIOS[@]} * ${#WEIGHTS_LIST[@]} * ${#LR_LIST[@]} * ${#SEED_LIST[@]} ))
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/sweeps/${DATASET}_${STAMP}"
mkdir -p "${SWEEP_DIR}"

SUMMARY_FILE="${SWEEP_DIR}/summary.tsv"
printf "index\tstatus\tdataset\tlp_ratio\tweights\tlr\tseed\ttag\tlog_file\n" > "${SUMMARY_FILE}"

echo "Sweep directory: ${SWEEP_DIR}"
echo "Total runs: ${TOTAL}"

run_idx=0
ok_count=0
fail_count=0
dry_count=0

for lp in "${LP_RATIOS[@]}"; do
  for weights in "${WEIGHTS_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        run_idx=$((run_idx + 1))

        lp_tag="$(sanitize_tag "${lp}")"
        w_tag="$(sanitize_tag "${weights}")"
        lr_tag="$(sanitize_tag "${lr}")"
        tag="grid_${DATASET}_lp${lp_tag}_w${w_tag}_lr${lr_tag}_s${seed}"
        run_log="${SWEEP_DIR}/run_$(printf '%03d' "${run_idx}")_${tag}.log"

        cmd=(
          "${PYTHON_BIN}" train.py
          --dataset "${DATASET}"
          --base-dir "${BASE_DIR}"
          --lp-ratio "${lp}"
          --weights "${weights}"
          --lr "${lr}"
          --max-epoch "${MAX_EPOCH}"
          --batch-size "${BATCH_SIZE}"
          --num-workers "${NUM_WORKERS}"
          --gpu "${GPU}"
          --seed "${seed}"
          --save-tag "${tag}"
        )
        if [[ "${CUDA_DETERMINISTIC}" -eq 1 ]]; then
          cmd+=(--cuda-deterministic)
        fi

        status="ok"
        echo "[${run_idx}/${TOTAL}] tag=${tag}"
        if [[ "${DRY_RUN}" -eq 1 ]]; then
          status="dryrun"
          dry_count=$((dry_count + 1))
          printf 'CMD: '
          printf '%q ' "${cmd[@]}"
          printf '\n'
        else
          {
            echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
            printf 'CMD: '
            printf '%q ' "${cmd[@]}"
            printf '\n'
          } | tee "${run_log}"

          if "${cmd[@]}" 2>&1 | tee -a "${run_log}"; then
            status="ok"
            ok_count=$((ok_count + 1))
          else
            status="fail"
            fail_count=$((fail_count + 1))
          fi
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${run_idx}" "${status}" "${DATASET}" "${lp}" "${weights}" "${lr}" "${seed}" "${tag}" "${run_log}" \
          >> "${SUMMARY_FILE}"
      done
    done
  done
done

echo "Finished. summary: ${SUMMARY_FILE}"
echo "ok=${ok_count}, fail=${fail_count}, dryrun=${dry_count}"

if [[ "${DRY_RUN}" -eq 0 && "${fail_count}" -gt 0 ]]; then
  exit 1
fi

