#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${ROOT_DIR}/config/col_run.conf}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

require_vars() {
    for var_name in "$@"; do
        if [ -z "${!var_name:-}" ]; then
            echo "${var_name} not set in config" >&2
            exit 1
        fi
    done
}

if [ -z "${MODEL_NAME:-}" ] && [ -n "${MODEL_PATH:-}" ]; then
    MODEL_NAME="$(basename "${MODEL_PATH}")"
fi

LOG_ROOT_DIR="${LOG_PATH_DIR}"
SAVE_ROOT_DIR="${SAVE_DIR_BASE}"

require_vars \
    MODEL_PATH DATASET_PATH SAVE_DIR_BASE LOG_PATH_DIR \
    GPUS BATCH_SIZE N_EPOCHS PLUGIN_NAME LR MICROBATCH_SIZE \
    MAX_LENGTH TPSIZE PPSIZE

# ps aux | grep colai | grep -v grep | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1

NGPUS=$(echo "${GPUS}" | awk -F ',' '{print NF}')
DATE_STAMP=$(date +%Y-%m-%d)
TIME_STAMP=$(date +%H%M%S)
LOG_PATH="${LOG_ROOT_DIR}/${MODEL_NAME}/${DATE_STAMP}"
SAVE_DIR="${SAVE_ROOT_DIR}/${MODEL_NAME}/${DATE_STAMP}/${TIME_STAMP}"
mkdir -p "${LOG_PATH}"
mkdir -p "${SAVE_DIR}"
LOG_FILE="${LOG_PATH}/[${MODEL_NAME}]_${TIME_STAMP}.log"

# # Llama
# export NCCL_DEBUG="TRACE"
# export NCCL_DEBUG_SUBSYS="INIT,GRAPH,ENV"
export NCCL_P2P_LEVEL=NVL
CMD=(
    colossalai run --nproc_per_node "${NGPUS}" --master_port 2955${GPUS:0:1}
    ./training/col_train.py --plugin "${PLUGIN_NAME}"
    --num_epochs "${N_EPOCHS}"
    --model_path "${MODEL_PATH}" --dataset "${DATASET_PATH}"
    --save_dir "${SAVE_DIR}"
    --lr "${LR}" --batch_size "${BATCH_SIZE}" --microbatch_size "${MICROBATCH_SIZE}" -g
    --max_length "${MAX_LENGTH}" --tpsize "${TPSIZE}" --ppsize "${PPSIZE}"
    --mixed_precision bf16 --flash_attention
)

if [ -n "${SAVE_INTERVAL:-}" ]; then
    CMD+=(--save_interval "${SAVE_INTERVAL}")
fi

if [ -n "${DATASET_SPLIT:-}" ]; then
    CMD+=(--split "${DATASET_SPLIT}")
fi

if [ "${USE_WANDB:-0}" = "1" ]; then
    CMD+=(--use_wandb)
    if [ -n "${WANDB_PROJECT:-}" ]; then
        CMD+=(--wandb_project "${WANDB_PROJECT}")
    fi
    if [ -n "${WANDB_ENTITY:-}" ]; then
        CMD+=(--wandb_entity "${WANDB_ENTITY}")
    fi
    if [ -n "${WANDB_RUN_NAME:-}" ]; then
        CMD+=(--wandb_run_name "${WANDB_RUN_NAME}")
    fi
fi

CUDA_VISIBLE_DEVICES="${GPUS}" CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    # --spsize 4 --sp_mode "ring"


printf "%-20s: [%s]\n" "Model" "${MODEL_NAME}"
printf "%-20s: [%s]\n" "Dataset" "${DATASET_PATH}"
printf "%-20s: [%s]\n" "Dataset Split" "${DATASET_SPLIT:-}"
printf "%-20s: [%s]\n" "Savecheckpoint Dir" "${SAVE_DIR}"
printf "%-20s: [%s]\n" "Log" "${LOG_FILE}"
printf "%-20s: [%s]\n" "Wandb" "${USE_WANDB:-0}"
printf "%-20s: [%s]\n" "Log PID" $$:$!
