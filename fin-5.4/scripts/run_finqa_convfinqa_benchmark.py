#!/usr/bin/env python3
import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>", re.IGNORECASE | re.DOTALL
)


@dataclass
class EvalResult:
    exact_match: bool
    execution_match: bool
    predicted_answer: str
    normalized_prediction: str
    normalized_reference: str
    numeric_prediction: float | None
    numeric_reference: float | None
    execution_target: float | None


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_text(text: str) -> str:
    text = text.strip().replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_single_number(text: str) -> float | None:
    matches = NUMERIC_RE.findall(text.replace("$", ""))
    if len(matches) != 1:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def reference_numeric_and_tolerance(reference: str) -> tuple[float | None, float | None]:
    normalized = normalize_text(reference)
    if normalized in {"yes", "no", "cannot be determined"}:
        return None, None
    value = parse_single_number(reference)
    if value is None:
        return None, None
    percent = "%" in reference
    digits = 0
    num_match = NUMERIC_RE.search(reference)
    if num_match and "." in num_match.group(0):
        digits = len(num_match.group(0).split(".", 1)[1])
    if percent:
        return value / 100.0, 0.5 * (10 ** (-(digits + 2)))
    return value, 0.5 * (10 ** (-digits))


def prediction_numeric_candidates(prediction: str, ref_is_percent: bool) -> list[float]:
    value = parse_single_number(prediction)
    if value is None:
        return []
    if ref_is_percent:
        if "%" in prediction:
            return [value / 100.0, value]
        return [value, value / 100.0]
    return [value]


def parse_execution_target(sample: dict[str, Any]) -> float | None:
    metadata = sample.get("metadata", {})
    gold = metadata.get("gold_answer_info", {}) if isinstance(metadata, dict) else {}
    raw = gold.get("execution_answer") if isinstance(gold, dict) else None
    if raw in (None, ""):
        return None
    try:
        return float(str(raw))
    except ValueError:
        return None


def evaluate_prediction(sample: dict[str, Any], prediction: str) -> EvalResult:
    reference = str(sample.get("reference_answer", "")).strip()
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    exact_match = normalized_prediction == normalized_reference

    execution_target = parse_execution_target(sample)
    ref_numeric, tolerance = reference_numeric_and_tolerance(reference)
    ref_is_percent = "%" in reference
    pred_candidates = prediction_numeric_candidates(prediction, ref_is_percent)

    execution_match = exact_match
    numeric_prediction = pred_candidates[0] if pred_candidates else None

    if not execution_match and normalized_reference in {"yes", "no", "cannot be determined"}:
        execution_match = exact_match
    elif ref_numeric is not None and tolerance is not None:
        for candidate in pred_candidates:
            if math.isclose(candidate, ref_numeric, abs_tol=tolerance):
                execution_match = True
                numeric_prediction = candidate
                break
            if execution_target is not None and math.isclose(candidate, execution_target, abs_tol=tolerance):
                execution_match = True
                numeric_prediction = candidate
                break

    return EvalResult(
        exact_match=exact_match,
        execution_match=execution_match,
        predicted_answer=prediction,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        numeric_prediction=numeric_prediction,
        numeric_reference=ref_numeric,
        execution_target=execution_target,
    )


def resolve_adapter_dir(adapter_or_run_dir: Path) -> Path:
    if (adapter_or_run_dir / "adapter_config.json").exists():
        return adapter_or_run_dir

    checkpoints = sorted(
        (
            path
            for path in adapter_or_run_dir.glob("checkpoint-*")
            if path.is_dir()
        ),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No adapter files or checkpoint-* directories found under {adapter_or_run_dir}"
        )

    latest = checkpoints[-1]
    trainer_state = latest / "trainer_state.json"
    if trainer_state.exists():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if best:
            best_path = Path(best)
            if best_path.exists():
                return best_path
    return latest


def build_history_block(sample: dict[str, Any]) -> str:
    raw_sample = sample.get("metadata", {}).get("raw_sample", {})
    annotation = raw_sample.get("annotation", {}) if isinstance(raw_sample, dict) else {}
    dialogue = annotation.get("dialogue_break") if isinstance(annotation, dict) else None
    if not isinstance(dialogue, list) or len(dialogue) <= 1:
        return "(none)"
    prior_turns = [turn.strip() for turn in dialogue[:-1] if str(turn).strip()]
    if not prior_turns:
        return "(none)"
    return "\n".join(f"{idx + 1}. {turn}" for idx, turn in enumerate(prior_turns))


def format_prompt(template: str, sample: dict[str, Any], dataset_name: str) -> str:
    payload = SafeDict(
        dataset_name=dataset_name,
        question=sample.get("question", ""),
        context=sample.get("context", ""),
        history_block=build_history_block(sample),
    )
    return template.format_map(payload)


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    rendered = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def run_dataset(
    *,
    dataset_name: str,
    samples: list[dict[str, Any]],
    prompt_template: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_new_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    exact_hits = 0
    exec_hits = 0
    rows: list[dict[str, Any]] = []

    prompts = [format_prompt(prompt_template, sample, dataset_name) for sample in samples]
    total_batches = math.ceil(len(samples) / batch_size)

    with output_path.open("w", encoding="utf-8") as sink:
        for batch_index, (sample_batch, prompt_batch) in enumerate(
            zip(batched(samples, batch_size), batched(prompts, batch_size)),
            start=1,
        ):
            generated_batch = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_batch,
                max_new_tokens=max_new_tokens,
            )
            for sample, prompt, generated_text in zip(sample_batch, prompt_batch, generated_batch):
                final_answer = extract_final_answer(generated_text)
                result = evaluate_prediction(sample, final_answer)
                exact_hits += int(result.exact_match)
                exec_hits += int(result.execution_match)
                row = {
                    "id": sample.get("id"),
                    "dataset": dataset_name,
                    "question": sample.get("question"),
                    "reference_answer": sample.get("reference_answer"),
                    "prediction": final_answer,
                    "raw_generation": generated_text,
                    "exact_match": result.exact_match,
                    "execution_match": result.execution_match,
                    "numeric_prediction": result.numeric_prediction,
                    "numeric_reference": result.numeric_reference,
                    "execution_target": result.execution_target,
                }
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{dataset_name}] batch {batch_index}/{total_batches} "
                f"exact={exact_hits}/{len(rows)} exec={exec_hits}/{len(rows)}",
                flush=True,
            )

    total = len(samples)
    return {
        "dataset": dataset_name,
        "samples": total,
        "exact_match": exact_hits,
        "exact_match_pct": round(100.0 * exact_hits / total, 2),
        "execution_match": exec_hits,
        "execution_match_pct": round(100.0 * exec_hits / total, 2),
        "predictions_path": str(output_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FinQA + ConvFinQA benchmark for a local Qwen2.5-7B LoRA adapter."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base Qwen model path or model ID.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA adapter directory or a run directory containing checkpoint-* folders.",
    )
    parser.add_argument(
        "--finqa-data",
        default=str(ROOT_DIR / "benchmark_inputs" / "finqa_test.jsonl"),
        help="FinQA benchmark JSONL path.",
    )
    parser.add_argument(
        "--convfinqa-data",
        default=str(ROOT_DIR / "benchmark_inputs" / "convfinqa_test.jsonl"),
        help="ConvFinQA benchmark JSONL path.",
    )
    parser.add_argument(
        "--finqa-prompt",
        default=str(ROOT_DIR / "prompts" / "finqa_benchmark_prompt.txt"),
        help="Prompt template for FinQA.",
    )
    parser.add_argument(
        "--convfinqa-prompt",
        default=str(ROOT_DIR / "prompts" / "convfinqa_benchmark_prompt.txt"),
        help="Prompt template for ConvFinQA.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-dataset limit for debugging. 0 means full dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow downloading and running custom model code from the Hub. "
        "Only enable this for models you trust. Standard Qwen2.5 models do not need this.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    adapter_input = Path(args.adapter).resolve()
    adapter_dir = resolve_adapter_dir(adapter_input)
    run_name = adapter_input.name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    finqa_samples = load_jsonl(Path(args.finqa_data))
    convfinqa_samples = load_jsonl(Path(args.convfinqa_data))
    if args.limit > 0:
        finqa_samples = finqa_samples[: args.limit]
        convfinqa_samples = convfinqa_samples[: args.limit]

    finqa_prompt = load_prompt(Path(args.finqa_prompt))
    convfinqa_prompt = load_prompt(Path(args.convfinqa_prompt))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    finqa_summary = run_dataset(
        dataset_name="finqa",
        samples=finqa_samples,
        prompt_template=finqa_prompt,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_path=output_dir / "finqa_predictions.jsonl",
    )
    convfinqa_summary = run_dataset(
        dataset_name="convfinqa",
        samples=convfinqa_samples,
        prompt_template=convfinqa_prompt,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_path=output_dir / "convfinqa_predictions.jsonl",
    )

    summary = {
        "created_at_utc": timestamp,
        "base_model": args.base_model,
        "adapter_input": str(adapter_input),
        "adapter_used": str(adapter_dir),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "datasets": {
            "finqa": finqa_summary,
            "convfinqa": convfinqa_summary,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
