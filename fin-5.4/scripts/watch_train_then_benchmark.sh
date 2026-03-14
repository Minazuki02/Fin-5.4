#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
TRAIN_PID="${TRAIN_PID:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
FINQA_PROMPT="${FINQA_PROMPT:-${ROOT_DIR}/prompts/finqa_benchmark_prompt.txt}"
CONVFINQA_PROMPT="${CONVFINQA_PROMPT:-${ROOT_DIR}/prompts/convfinqa_benchmark_prompt.txt}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
POLL_SECS="${POLL_SECS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/.cache/modelscope}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"

export MODELSCOPE_CACHE
export HF_HOME

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "RUN_DIR not found: ${RUN_DIR}" >&2
  exit 1
fi

for required_path in \
  "${FINQA_PROMPT}" \
  "${CONVFINQA_PROMPT}" \
  "${ROOT_DIR}/benchmark_inputs/finqa_test.jsonl" \
  "${ROOT_DIR}/benchmark_inputs/convfinqa_test.jsonl"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required file not found: ${required_path}" >&2
    exit 1
  fi
done

wait_for_training() {
  if [[ -n "${TRAIN_PID}" ]]; then
    echo "Waiting for train pid ${TRAIN_PID} to exit..."
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do
      sleep "${POLL_SECS}"
    done
    return
  fi

  echo "No TRAIN_PID provided; falling back to process-name polling..."
  while pgrep -af "swift sft" | grep -q "${RUN_DIR%/}"; do
    sleep "${POLL_SECS}"
  done
}

wait_for_training

echo "Training finished, starting benchmark for ${RUN_DIR}"
cd "${ROOT_DIR}"
python scripts/run_finqa_convfinqa_benchmark.py \
  --base-model "${BASE_MODEL}" \
  --adapter "${RUN_DIR}" \
  --finqa-data "${ROOT_DIR}/benchmark_inputs/finqa_test.jsonl" \
  --convfinqa-data "${ROOT_DIR}/benchmark_inputs/convfinqa_test.jsonl" \
  --finqa-prompt "${FINQA_PROMPT}" \
  --convfinqa-prompt "${CONVFINQA_PROMPT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUTPUT_DIR}"
