#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", {})
    return {
        "label": path.parent.name,
        "path": str(path),
        "finqa_exact": datasets.get("finqa", {}).get("exact_match_pct", 0.0),
        "finqa_exec": datasets.get("finqa", {}).get("execution_match_pct", 0.0),
        "convfinqa_exact": datasets.get("convfinqa", {}).get("exact_match_pct", 0.0),
        "convfinqa_exec": datasets.get("convfinqa", {}).get("execution_match_pct", 0.0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a benchmark comparison chart from summary.json files.")
    parser.add_argument(
        "--summary",
        nargs="+",
        required=True,
        help="One or more summary.json paths produced by run_finqa_convfinqa_benchmark.py.",
    )
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--title", default="Benchmark Comparison", help="Figure title.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = [load_summary(Path(item).expanduser().resolve()) for item in args.summary]
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 5.2), dpi=180)
    fig.patch.set_facecolor("#f6f4ef")
    ax.set_facecolor("#fffdf7")
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar([v - 1.5 * width for v in x], [row["finqa_exact"] for row in rows], width=width, color="#2e86ab", label="FinQA exact")
    ax.bar([v - 0.5 * width for v in x], [row["finqa_exec"] for row in rows], width=width, color="#5fa8d3", label="FinQA exec")
    ax.bar([v + 0.5 * width for v in x], [row["convfinqa_exact"] for row in rows], width=width, color="#d1495b", label="ConvFinQA exact")
    ax.bar([v + 1.5 * width for v in x], [row["convfinqa_exec"] for row in rows], width=width, color="#edae49", label="ConvFinQA exec")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(args.title, fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
