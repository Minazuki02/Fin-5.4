#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/swift_final_mixture_exact}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.jsonl}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/qwen25_7b_instruct_lora_mixed}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/.cache/modelscope}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"

MAX_LENGTH="${MAX_LENGTH:-3072}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
PACKING="${PACKING:-false}"
PACKING_LENGTH="${PACKING_LENGTH:-${MAX_LENGTH}}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

unset OMP_NUM_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"
export MODELSCOPE_CACHE
export HF_HOME

if ! command -v swift >/dev/null 2>&1; then
  echo "swift command not found. Run scripts/install_ms_swift_env.sh first." >&2
  exit 1
fi

for f in "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing dataset file: ${f}" >&2
    exit 1
  fi
done

python "${ROOT_DIR}/scripts/validate_swift_dataset.py" "${TRAIN_FILE}" "${VAL_FILE}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}"

CMD=(
  swift sft
  --model "${MODEL_ID}"
  --tuner_type lora
  --dataset "${TRAIN_FILE}"
  --val_dataset "${VAL_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size 2
  --gradient_accumulation_steps "${GRAD_ACC}"
  --num_train_epochs "${EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --learning_rate "${LR}"
  --lr_scheduler_type cosine
  --warmup_ratio "${WARMUP_RATIO}"
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 3
  --logging_steps "${LOGGING_STEPS}"
  --gradient_checkpointing true
  --dataloader_num_workers 4
  --dataset_num_proc 4
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --target_modules all-linear
  --packing "${PACKING}"
  --packing_length "${PACKING_LENGTH}"
)

if [[ -n "${ATTN_IMPL}" ]]; then
  CMD+=(--attn_impl "${ATTN_IMPL}")
fi

"${CMD[@]}"
