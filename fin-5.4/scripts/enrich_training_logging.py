#!/usr/bin/env python3
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer


SOURCE_TO_TASK = {
    'convfinqa': 'reasoning',
    'finqa': 'reasoning',
    'tatqa': 'reasoning',
    'financeiq': 'mcq',
    'finance_alpaca': 'instruction',
    'financial_phrasebank': 'sentiment',
    'fincuge_instruction': 'instruction',
}


def load_run_args(run_dir: Path) -> dict:
    return json.loads((run_dir / 'args.json').read_text(encoding='utf-8'))


def load_dataset_records(dataset_path: Path, tokenizer, max_length: int):
    records = []
    with dataset_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            source = (
                obj.get('source')
                or obj.get('dataset')
                or obj.get('metadata', {}).get('source')
                or 'unknown'
            )
            task_type = SOURCE_TO_TASK.get(source, 'unknown')
            text = tokenizer.apply_chat_template(
                obj['messages'],
                tokenize=False,
                add_generation_prompt=False,
            )
            raw_len = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            records.append({
                'idx': i,
                'source': source,
                'task_type': task_type,
                'raw_len': raw_len,
                'eff_len': min(raw_len, max_length),
            })
    return records


def build_step_stats(records, seed: int, batch_size: int, grad_acc: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(records), generator=generator).tolist()
    step_width = batch_size * grad_acc
    out = {}
    total_steps = len(order) // step_width
    for step_idx in range(total_steps):
        idxs = order[step_idx * step_width:(step_idx + 1) * step_width]
        batch = [records[i] for i in idxs]
        raw_lens = [x['raw_len'] for x in batch]
        eff_lens = [x['eff_len'] for x in batch]
        source_counts = Counter(x['source'] for x in batch)
        task_counts = Counter(x['task_type'] for x in batch)
        dominant_source = source_counts.most_common(1)[0][0]
        dominant_task_type = task_counts.most_common(1)[0][0]
        out[step_idx + 1] = {
            'batch_size_effective': len(batch),
            'batch_avg_raw_len': round(sum(raw_lens) / len(raw_lens), 2),
            'batch_avg_eff_len': round(sum(eff_lens) / len(eff_lens), 2),
            'batch_max_raw_len': max(raw_lens),
            'batch_max_eff_len': max(eff_lens),
            'batch_over_1024': sum(1 for x in raw_lens if x > 1024),
            'batch_over_2048': sum(1 for x in raw_lens if x > 2048),
            'batch_over_3072': sum(1 for x in raw_lens if x > 3072),
            'batch_sources': dict(source_counts),
            'batch_task_types': dict(task_counts),
            'batch_dominant_source': dominant_source,
            'batch_dominant_task_type': dominant_task_type,
            'batch_reasoning_ratio': round(task_counts.get('reasoning', 0) / len(batch), 4),
            'batch_instruction_ratio': round(task_counts.get('instruction', 0) / len(batch), 4),
        }
    return out


def parse_step(obj: dict):
    value = obj.get('global_step/max_steps')
    if not value:
        return None
    try:
        return int(str(value).split('/')[0])
    except (TypeError, ValueError):
        return None


def enrich_existing(log_path: Path, enriched_path: Path, step_stats: dict):
    processed_steps = set()
    if enriched_path.exists():
        with enriched_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    step = parse_step(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if step is not None:
                    processed_steps.add(step)
    with log_path.open('r', encoding='utf-8') as src, enriched_path.open('a', encoding='utf-8') as dst:
        for line in src:
            obj = json.loads(line)
            step = parse_step(obj)
            if step is None or step in processed_steps:
                continue
            obj.update(step_stats.get(step, {}))
            dst.write(json.dumps(obj, ensure_ascii=False) + '\n')
            dst.flush()
            processed_steps.add(step)
    return processed_steps


def follow(log_path: Path, enriched_path: Path, step_stats: dict, interval: float):
    processed_steps = enrich_existing(log_path, enriched_path, step_stats)
    while True:
        time.sleep(interval)
        processed_steps = enrich_existing(log_path, enriched_path, step_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--dataset_path')
    parser.add_argument('--model_path')
    parser.add_argument('--interval', type=float, default=10.0)
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=False,
        help='Allow downloading and running custom model code from the Hub. '
        'Only enable this for models you trust. Standard Qwen2.5 models do not need this.',
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_args = load_run_args(run_dir)
    dataset_path = Path(args.dataset_path or run_args['dataset'][0])
    model_path = args.model_path or 'Qwen/Qwen2.5-7B-Instruct'
    log_path = run_dir / 'logging.jsonl'
    enriched_path = run_dir / 'logging_enriched.jsonl'

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=args.trust_remote_code
    )
    records = load_dataset_records(dataset_path, tokenizer, int(run_args['max_length']))
    step_stats = build_step_stats(
        records,
        seed=int(run_args['seed']),
        batch_size=int(run_args['per_device_train_batch_size']),
        grad_acc=int(run_args['gradient_accumulation_steps']),
    )
    follow(log_path, enriched_path, step_stats, args.interval)


if __name__ == '__main__':
    main()
