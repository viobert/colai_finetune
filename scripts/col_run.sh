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

require_vars \
    MODEL_PATH DATASET_PATH SAVE_DIR_BASE LOG_PATH_DIR SAVE_INTERVAL_BASE \
    GPUS BATCH_SIZE N_EPOCHS PLUGIN_NAME LR MICROBATCH_SIZE \
    MAX_LENGTH TPSIZE PPSIZE

# ps aux | grep colai | grep -v grep | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1

NGPUS=$(echo "${GPUS}" | awk -F ',' '{print NF}')
SAVE_INTERVAL=$((SAVE_INTERVAL_BASE / BATCH_SIZE))

DATE_STAMP=$(date +%Y-%m-%d)
TIME_STAMP=$(date +%H%M%S)
LOG_PATH="${LOG_PATH_DIR}/${DATE_STAMP}"
SAVE_DIR="${SAVE_DIR_BASE}/${DATE_STAMP}/${COLLATE_FN}_${MODEL_NAME}_${TIME_STAMP}_$$"
mkdir -p "${LOG_PATH}"
mkdir -p "${SAVE_DIR}"

# # Llama
export NCCL_DEBUG="TRACE"
export NCCL_DEBUG_SUBSYS="INIT,GRAPH,ENV"
export NCCL_P2P_LEVEL=NVL
CUDA_VISIBLE_DEVICES="${GPUS}" CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
    nohup colossalai run --nproc_per_node "${NGPUS}" --master_port 2955${GPUS:0:1} \
    ./training/col_train.py  --plugin "${PLUGIN_NAME}" \
    --num_epochs "${N_EPOCHS}" \
    --model_path "${MODEL_PATH}" --dataset "${DATASET_PATH}" \
    --save_dir "${SAVE_DIR}" --save_interval "${SAVE_INTERVAL}" \
    --lr "${LR}" --batch_size "${BATCH_SIZE}" --microbatch_size "${MICROBATCH_SIZE}" -g \
    --max_length "${MAX_LENGTH}" --tpsize "${TPSIZE}" --ppsize "${PPSIZE}" \
    --mixed_precision bf16 --flash_attention \
    --tensorboard_dir "${LOG_PATH_DIR}/tb_logs" \
    > "${LOG_PATH}/[$$]${MODEL_NAME}.log" 2>&1 &
    # --spsize 4 --sp_mode "ring" 


printf "%-20s: [%s]\n" "Model" "${MODEL_NAME}"
printf "%-20s: [%s]\n" "Dataset" "${DATASET_PATH}"
printf "%-20s: [%s]\n" "Savecheckpoint Dir" "${SAVE_DIR}"
printf "%-20s: [%s]\n" "Log" "${LOG_PATH}"
printf "%-20s: [%s]\n" "Log PID" $$:$!
