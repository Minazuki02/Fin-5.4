#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT_DIR}/outputs/qwen25_7b_instruct_lora_mixed}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/.cache/modelscope}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"

unset OMP_NUM_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"
export MODELSCOPE_CACHE
export HF_HOME

if ! command -v swift >/dev/null 2>&1; then
  echo "swift command not found. Run scripts/install_ms_swift_env.sh first." >&2
  exit 1
fi

swift infer \
  --model "${MODEL_ID}" \
  --adapters "${ADAPTER_DIR}"
