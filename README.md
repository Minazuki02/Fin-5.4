# Fin-5.4 Data Training Scripts

[![Model](https://img.shields.io/badge/model-Qwen2.5--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Framework](https://img.shields.io/badge/framework-ms--swift-orange)](https://github.com/modelscope/ms-swift)
[![Training](https://img.shields.io/badge/training-LoRA_SFT-green)](#quick-start)

面向 `Qwen2.5-7B-Instruct` 的 LoRA 微调脚本仓库，包含：

- `ms-swift` 训练启动脚本
- 数据集 JSONL 结构校验脚本
- LoRA 推理与权重合并脚本
- FinQA / ConvFinQA 基准评测脚本
- 训练日志增强脚本

这个仓库聚焦于“训练与评测流水线”，数据构建部分在你提供的另一个仓库中完成；这里只假设最终训练数据已经整理成 `messages` 格式。

## Repository Structure

```text
.
├── benchmark_inputs/              # FinQA / ConvFinQA 测试集
├── prompts/                       # Benchmark prompt 模板
├── scripts/
│   ├── install_ms_swift_env.sh    # 环境安装
│   ├── validate_swift_dataset.py  # 数据校验
│   ├── run_qwen25_7b_lora_sft.sh  # LoRA SFT 训练
│   ├── infer_lora_chat.sh         # LoRA 交互推理
│   ├── export_lora_merge.sh       # 合并 LoRA 权重
│   ├── run_finqa_convfinqa_benchmark.py
│   ├── watch_train_then_benchmark.sh
│   └── enrich_training_logging.py
└── requirements-ms-swift.txt
```

## Features

- 开箱即用的 `Qwen2.5-7B-Instruct` LoRA SFT 启动脚本
- 用环境变量覆盖训练参数，方便在 AutoDL / 本地 GPU 机器复用
- 在训练前自动校验 JSONL 数据结构，避免跑到一半才发现格式错误
- 支持 LoRA 适配器推理、导出合并模型、以及 FinQA / ConvFinQA 评测

## Dataset Format

训练数据默认放在 `swift_final_mixture_exact/` 目录下：

```text
swift_final_mixture_exact/
├── train.jsonl
└── val.jsonl
```

每一行应为一个 JSON 对象，至少包含：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ]
}
```

可选字段如 `source`、`dataset`、`metadata` 会被保留，不影响训练。

## Default Training Config

当前训练脚本默认配置如下：

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Training mode | `LoRA SFT` |
| `max_length` | `3072` |
| `per_device_train_batch_size` | `4` |
| `gradient_accumulation_steps` | `8` |
| `num_train_epochs` | `2` |
| `learning_rate` | `1e-4` |
| `lora_rank` | `32` |
| `lora_alpha` | `64` |
| `lora_dropout` | `0.05` |
| `torch_dtype` | `bfloat16` |

单卡有效 batch size 为 `4 x 8 = 32`。

## Quick Start

### 1. Install

```bash
bash scripts/install_ms_swift_env.sh
```

默认会在仓库根目录创建 `.venv` 并安装依赖。

### 2. Validate Dataset

```bash
python scripts/validate_swift_dataset.py \
  swift_final_mixture_exact/train.jsonl \
  swift_final_mixture_exact/val.jsonl
```

### 3. Run LoRA SFT

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen25_7b_lora_sft.sh
```

## Common Overrides

训练脚本支持通过环境变量覆盖默认参数：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=./outputs/qwen25_run1 \
LR=5e-5 \
SAVE_STEPS=200 \
EVAL_STEPS=200 \
bash scripts/run_qwen25_7b_lora_sft.sh
```

切换模型或数据路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
DATA_DIR=./swift_final_mixture_exact \
bash scripts/run_qwen25_7b_lora_sft.sh
```

## Inference And Export

LoRA 交互推理：

```bash
ADAPTER_DIR=./outputs/qwen25_7b_instruct_lora_mixed \
bash scripts/infer_lora_chat.sh
```

合并 LoRA 权重：

```bash
ADAPTER_DIR=./outputs/qwen25_7b_instruct_lora_mixed/checkpoint-xxxx \
MERGED_DIR=./outputs/qwen25_7b_instruct_lora_merged \
bash scripts/export_lora_merge.sh
```

## Benchmark

运行 FinQA + ConvFinQA 基准评测：

```bash
python scripts/run_finqa_convfinqa_benchmark.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter ./outputs/qwen25_7b_instruct_lora_mixed
```

如果希望训练结束后自动开始评测：

```bash
RUN_DIR=./outputs/qwen25_7b_instruct_lora_mixed \
bash scripts/watch_train_then_benchmark.sh
```

## Results

当前以 `execution accuracy` 作为主要 benchmark 指标，对比结果如下：

| Model | FinQA | ConvFinQA |
| --- | ---: | ---: |
| Qwen2.5-7B-Instruct (base) | `60%` | `66%` |
| This LoRA model | `75%` | `77%` |
| GPT-5.4 | `82%` | `86%` |

说明：

- 当前表格展示的是 `FinQA` 和 `ConvFinQA` 的 `execution accuracy`
- `This LoRA model` 指当前仓库对应的微调版本
- 结果用于展示相对提升，后续如补充更完整评测会继续更新

## Visualization

这份 GitHub 上传包刻意排除了训练结果目录，所以当前仓库里没有实际的 `logging.jsonl` / `trainer_state.json`。不过仓库现在已经内置了出图脚本，后续把训练 run 或 benchmark 结果放回目录后，可以直接生成 README 用图。

生成训练 loss / learning rate 曲线：

```bash
python scripts/plot_training_curves.py \
  --run-dir /path/to/your/run_dir \
  --output assets/training_curves.svg \
  --title "Qwen2.5-7B LoRA Training Curves"
```

生成多组权重的 benchmark 对比图：

```bash
python scripts/plot_benchmark_results.py \
  --summary benchmark_results/run_a/summary.json benchmark_results/run_b/summary.json \
  --output assets/benchmark_comparison.svg \
  --title "Adapter Benchmark Comparison"
```

图生成好之后，可以直接把下面两行放进 README 展示：

```markdown
![Training Curves](assets/training_curves.svg)
![Benchmark Comparison](assets/benchmark_comparison.svg)
```

### Current Run Snapshot

基于 `v11_full_experiment_bundle/checkpoints/final_checkpoint-2650/trainer_state.json` 生成的真实训练曲线：

- Final global step: `2650`
- Best checkpoint step: `2350`
- Best eval loss: `0.42924741`

![V11 Final Checkpoint Training Curves](assets/v11_training_curves_final_2650.svg)

## Notes

- 当前仓库不直接包含训练数据；上传 GitHub 时建议只保留脚本、模板、小型 benchmark 输入样例和必要文档。
- `benchmark_inputs/` 与 `prompts/` 已按脚本实际路径组织，可直接用于评测。
- 训练脚本会先执行数据校验，再调用 `swift sft`，这样能更早暴露格式问题。
